"""Third-round causal input/label contracts and provenance helpers.

Round-2 observation/economic/ridge functions are ported here to avoid a numba
dependency in the PyTorch runtime; numerical parity is checked independently.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / 'research'
V2 = ROOT.parent / 'research_v2'
OUT = ROOT / 'results'
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ['HF_HUB_DISABLE_XET'] = '1'
sys.path.insert(0, str(ROOT / 'vendor_py'))

import numpy as np
import pandas as pd
from scipy import linalg


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def cfg():
    return json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))


def previous_evidence():
    return {str(p.relative_to(ROOT.parent)): sha(p) for folder in [V1, V2]
            for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def observation_table(frame):
    dates = pd.to_datetime(frame.date)
    weekday = dates.dt.dayofweek.to_numpy()
    n = len(frame)
    next_anchor = np.full(n, -1, dtype=int)
    for day in range(5):
        ids = np.flatnonzero(weekday == day)
        next_anchor[ids[:-1]] = ids[1:]
    o, c = frame.open.to_numpy(float), frame.close.to_numpy(float)
    good = frame.valid_ohlc.to_numpy(bool)
    rows = []
    for t in range(124, n - 5):
        e = next_anchor[t] + 1
        if next_anchor[t] < 0 or e >= n:
            continue
        end = max(t + 5, e)
        if not good[t - 124:end + 1].all():
            continue
        rows.append(dict(anchor=t, date=str(frame.date.iloc[t]), weekday=int(weekday[t]),
                         entry=t + 1, exit=e, exit_date=str(frame.date.iloc[e]),
                         close_end=t + 5, completed=str(frame.date.iloc[end]),
                         close_return=c[t + 5] / c[t] - 1,
                         exec_return=o[e] / o[t + 1] - 1, sessions=e - (t + 1)))
    return pd.DataFrame(rows)


def fixed_lengths(n, length):
    return ([n % length] if n % length else []) + [length] * (n // length)


def economic_window(window):
    op, hi, lo, cl, vol = window[['open', 'high', 'low', 'close', 'volume']].to_numpy(float).T
    close_log = np.log(cl)
    ret = np.r_[0., np.diff(close_log)]
    gap = np.r_[0., np.log(op[1:] / cl[:-1])]
    spread = np.log(hi / lo)
    cs = np.r_[0., np.cumsum(vol)]
    scale = np.array([(cs[k + 1] - cs[max(0, k - 19)]) / (k + 1 - max(0, k - 19)) for k in range(len(vol))])
    relative_volume = np.log1p(vol) - np.log1p(scale)
    channels = np.column_stack([ret, gap, np.log(cl / op), spread,
                                np.log(hi / np.maximum(op, cl)),
                                np.log(np.minimum(op, cl) / lo), relative_volume,
                                (2 * cl - hi - lo) / np.maximum(hi - lo, 1e-8)])
    feature = list(channels[-5:].reshape(-1))
    for length in [5, 10, 20, 60, 125]:
        r = ret[-length:]
        feature.extend([r.sum(), r.std(), np.sqrt(np.mean(np.minimum(r, 0) ** 2)),
                        spread[-length:].mean(), relative_volume[-length:].mean(),
                        cl[-1] / cl[-length:].max() - 1,
                        close_log[-1] - close_log[-length:].mean()])
    return channels, np.asarray(feature)


def multi_features(channels, base):
    result = list(base)
    for length in [5, 10, 20]:
        start = 0
        for size in fixed_lengths(125, length):
            segment = channels[start:start + size]
            result.extend([segment[:, 0].sum(), segment[:, 0].std(),
                           segment[:, 3].mean(), segment[:, 6].mean()])
            start += size
    assert len(result) == 255
    return np.asarray(result)


def cross_window(target_close, panel_close):
    """panel_close: seven synchronized indices by 125 historical bars."""
    target = np.r_[0., np.diff(np.log(target_close))]
    panel = np.c_[np.zeros(len(panel_close)), np.diff(np.log(panel_close), axis=1)]
    result = []
    for other in panel:
        for length in [5, 20, 60]:
            a, b = target[-length:], other[-length:]
            corr = np.corrcoef(a, b)[0, 1] if a.std() > 1e-12 and b.std() > 1e-12 else 0.
            result.extend([b.sum() - a.sum(), b.std() - a.std(), corr])
    for length in [5, 20, 60]:
        sums = panel[:, -length:].sum(axis=1)
        result.extend([(sums > 0).mean(), sums.mean(), sums.std()])
    assert len(result) == 72
    return np.asarray(result)


def build_features(frame, obs, panels):
    aligned = [p.set_index('date').reindex(frame.date) for p in panels]
    valid = np.ones(len(frame), dtype=bool)
    for p in aligned:
        valid &= p.valid_ohlc.fillna(False).to_numpy(bool)
        valid &= np.isfinite(p[['open', 'high', 'low', 'close', 'volume']].to_numpy(float)).all(axis=1)
    target, cross, eligible = [], [], []
    close = frame.close.to_numpy(float)
    others = np.array([p.close.to_numpy(float) for p in aligned])
    for t in obs.anchor.to_numpy(int):
        channels, base = economic_window(frame.iloc[t - 124:t + 1])
        target.append(multi_features(channels, base))
        good = valid[t - 124:t + 1].all()
        eligible.append(good)
        cross.append(cross_window(close[t - 124:t + 1], others[:, t - 124:t + 1]) if good else np.full(72, np.nan))
    return np.asarray(target), np.asarray(cross), np.asarray(eligible)


def periods(start_year, end_year):
    for year in range(start_year, end_year + 1):
        for month in [1, 4, 7, 10]:
            start = pd.Timestamp(year, month, 1)
            end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
            yield (start - pd.Timedelta(days=1)).strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def masks(obs, cutoff, end, eligible=None):
    train = (obs.completed <= cutoff).to_numpy()
    if eligible is not None:
        train &= eligible
    test = ((obs.date > cutoff) & (obs.date <= end) & (obs.weekday == 4)).to_numpy()
    assert train.any() and obs.completed[train].max() <= cutoff
    return train, test


def ridge_path(x, y, xt, lambdas):
    mean, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-6] = 1.
    z, zt = (x - mean) / sd, (xt - mean) / sd
    ym = y.mean()
    ev, vec = linalg.eigh(z.T @ z / len(z), check_finite=False)
    rhs = vec.T @ (z.T @ (y - ym) / len(z))
    left = zt @ vec
    return {str(l): left @ (rhs / (np.maximum(ev, 0) + l)) + ym for l in lambdas}


def prediction_row(row, name, prediction, cutoff, train_n):
    return dict(method=name, date=row.date, anchor=int(row.anchor), year=int(row.date[:4]),
                entry=int(row.entry), exit=int(row.exit), exit_date=row.exit_date,
                actual=float(row.exec_return), predicted_return=float(prediction),
                position=int(prediction > 0), correct=int((prediction > 0) == (row.exec_return > 0)),
                cutoff=cutoff, train_n=int(train_n))


def feature_names():
    result = [f'target_multiscale_{i:03d}' for i in range(255)]
    for symbol in cfg()['cross_index_features']['indices']:
        for length in [5, 20, 60]:
            result.extend([f'{symbol}_{length}_{stat}' for stat in ['relative_return', 'vol_difference', 'return_corr']])
    for length in [5, 20, 60]:
        result.extend([f'panel_{length}_{stat}' for stat in ['positive_fraction', 'mean_return', 'return_dispersion']])
    return result

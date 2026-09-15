"""Evaluate all frozen methods, paired uncertainty and implementable-time index proxies."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent


def wilson(k, n):
    z = 1.95996398454
    p = k/n
    mid = (p + z*z/(2*n)) / (1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return mid-half, mid+half


def prediction_metrics(g):
    y, yh = g.actual.to_numpy(), g.prediction.to_numpy()
    a, b = y > 0, yh > 0
    acc = float((a == b).mean())
    ci = wilson(int((a == b).sum()), len(a))
    balanced = ((b[a]).mean() + (~b[~a]).mean()) / 2 if a.any() and (~a).any() else np.nan
    return dict(n=len(g), accuracy=acc, wilson_low=ci[0], wilson_high=ci[1],
                balanced_accuracy=balanced, true_up_rate=float(a.mean()), predicted_up_rate=float(b.mean()),
                return_rmse=float(np.sqrt(np.mean((yh-y)**2))),
                mse_skill_vs_zero=float(1 - np.mean((yh-y)**2)/np.mean(y**2)),
                correlation=float(np.corrcoef(yh, y)[0, 1]) if yh.std() > 1e-12 else None,
                ohlc_invalid_fraction=float(g.ohlc_invalid_bars.mean()/5) if g.ohlc_invalid_bars.notna().any() else None)


def bootstrap_indices(n, block=8, b=10000, seed=20260909):
    rng = np.random.default_rng(seed)
    starts = rng.integers(n, size=(b, int(np.ceil(n/block))))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(b, -1)[:, :n]


def paired(g, ref):
    assert g.date.tolist() == ref.date.tolist()
    n = len(g)
    ids = bootstrap_indices(n)
    d = g.correct.to_numpy() - ref.correct.to_numpy()
    means = d[ids].mean(axis=1)
    # Two-sided centered bootstrap: infer against null difference=0, not bootstrap sign probability.
    centered = (d-d.mean())[ids].mean(axis=1)
    p = (1+np.sum(abs(centered) >= abs(d.mean()))) / (len(ids)+1)
    ga, ra = g.correct.to_numpy(bool), ref.correct.to_numpy(bool)
    win, loss = int((ga & ~ra).sum()), int((~ga & ra).sum())
    e = (g.prediction.to_numpy()-g.actual.to_numpy())**2
    er = (ref.prediction.to_numpy()-ref.actual.to_numpy())**2
    rmse_d = np.sqrt(e[ids].mean(axis=1)) - np.sqrt(er[ids].mean(axis=1))
    return dict(method=g.method.iloc[0], reference=ref.method.iloc[0], n=n,
                accuracy_difference=float(d.mean()), ci95_low=float(np.quantile(means, .025)),
                ci95_high=float(np.quantile(means, .975)), centered_block_bootstrap_p=float(p),
                wins_only=win, losses_only=loss,
                mcnemar_exact_p=float(binomtest(win, win+loss, p=.5).pvalue) if win+loss else 1.,
                return_rmse_difference=float(np.sqrt(e.mean())-np.sqrt(er.mean())),
                rmse_difference_ci95_low=float(np.quantile(rmse_d, .025)),
                rmse_difference_ci95_high=float(np.quantile(rmse_d, .975)))


def backtest(frame, g, bps):
    """Position changes at next open; mark open-to-open; force close at terminal open."""
    changes = dict(zip(g.anchor.astype(int)+1, g.position.astype(int)))
    first, last = min(changes), len(frame)-1
    opens = frame.open.to_numpy(float)
    pos = 0
    wealth = 1.
    turnover = 0
    rows = [dict(date=frame.date.iloc[first], wealth=1., net_return=0., position=0, turnover=0., initial=True)]
    ret, held = [], []
    for i in range(first, last):
        target = changes.get(i, pos)
        change = abs(target-pos)
        gross = target * (opens[i+1]/opens[i]-1)
        growth = (1-change*bps/10000) * (1+gross)
        wealth *= growth
        ret.append(growth-1)
        held.append(target)
        turnover += change
        rows.append(dict(date=frame.date.iloc[i+1], wealth=wealth, net_return=growth-1,
                         position=target, turnover=change, initial=False))
        pos = target
    if pos:
        wealth *= (1-bps/10000)
        rows[-1]["wealth"] = wealth
        ret[-1] = (1+ret[-1])*(1-bps/10000)-1
        rows[-1]["net_return"] = ret[-1]
        rows[-1]["turnover"] += pos
        turnover += pos
    a = np.array(ret)
    values = np.array([r["wealth"] for r in rows])
    maxdd = (values/np.maximum.accumulate(values)-1).min()
    days = (pd.Timestamp(frame.date.iloc[last])-pd.Timestamp(frame.date.iloc[first])).days
    sd = a.std(ddof=1) if len(a)>1 else 0.
    metrics = dict(cost_bps=bps, start=frame.date.iloc[first], end=frame.date.iloc[last],
                   total_return=wealth-1, cagr=wealth**(365.25/days)-1,
                   sharpe_zero_cash=float(a.mean()/sd*np.sqrt(252)) if sd>0 else 0.,
                   max_drawdown=float(maxdd), exposure=float(np.mean(held)),
                   one_way_turnover=int(turnover), observations=len(a))
    return metrics, rows


def main():
    out = ROOT / "results"
    predictions = pd.read_csv(out / "predictions.csv")
    frame = pd.read_csv(ROOT / "data" / "1_000300.csv")
    cfg = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    metrics, yearly, costs, curve_rows = [], [], [], []
    for method, g in predictions.groupby("method", sort=False):
        g = g.sort_values("date")
        metrics.append(dict(method=method, **prediction_metrics(g)))
        for year, sub in g.groupby("year"):
            yearly.append(dict(method=method, year=int(year), **prediction_metrics(sub)))
        for bps in cfg["cost_bps_per_one_way_change"]:
            m, rows = backtest(frame, g, bps)
            costs.append(dict(method=method, **m))
            if bps == 10:
                curve_rows.extend([dict(method=method, **r) for r in rows])
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(yearly).to_csv(out / "yearly_metrics.csv", index=False)
    pd.DataFrame(costs).to_csv(out / "cost_sensitivity.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / "equity_curves_10bps.csv", index=False)
    groups = {m: g.sort_values("date") for m, g in predictions.groupby("method")}
    comparisons = [paired(groups["adaptive_normalized"], groups[r])
                   for r in ["fixed_10", "random_partition", "always_up", "adaptive_raw", "fixed_5", "fixed_20"]]
    (out / "paired_statistics.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    # Fixed block-length sensitivity; does not select a model or change primary inference.
    d = groups["adaptive_normalized"].correct.to_numpy() - groups["fixed_10"].correct.to_numpy()
    sensitivity = []
    for block in [4, 8, 13, 26]:
        z = d[bootstrap_indices(len(d), block=block)].mean(axis=1)
        sensitivity.append(dict(block=block, difference=float(d.mean()),
                                ci95_low=float(np.quantile(z,.025)), ci95_high=float(np.quantile(z,.975))))
    (out / "bootstrap_block_sensitivity.json").write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")
    # Report all realized horizon overlap, rather than treating weekly labels as independent.
    g = groups["adaptive_normalized"]
    overlap = int((g.anchor.to_numpy()[1:] < g.anchor.to_numpy()[:-1]+5).sum())
    audit = dict(n_weekly=len(g), prediction_start=g.date.iloc[0], prediction_end=g.date.iloc[-1],
                 label_end=g.label_end.iloc[-1], overlapping_adjacent_5day_labels=overlap,
                 train_cutoff_before_every_signal=bool((predictions.train_cutoff < predictions.date).all()),
                 all_values_finite=bool(np.isfinite(predictions[["prediction","actual"]].to_numpy()).all()),
                 no_duplicate_predictions=not predictions.duplicated(["method","date"]).any(),
                 paired_identical_test_dates=all(h.date.tolist()==g.date.tolist() for h in groups.values()),
                 inference_no_pretrained_components=True,
                 execution="after signal close, next observed open; index proxy; hypothetical bps; terminal liquidation")
    (out / "evaluation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(pd.DataFrame(metrics)[["method","n","accuracy","return_rmse","ohlc_invalid_fraction"]].to_string(index=False))
    print(json.dumps(comparisons[0], indent=2))
    print(pd.DataFrame(costs).query("cost_bps == 10")[["method","total_return","cagr","max_drawdown","sharpe_zero_cash"]].to_string(index=False))


if __name__ == "__main__":
    main()

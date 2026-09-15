"""Annual walk-forward controlled ablation, executable with Python + numpy/pandas/scipy/numba."""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time
import numpy as np
import pandas as pd
from scipy import linalg
from segmentation import learn_prototypes, make_dataset

ROOT = Path(__file__).resolve().parent
METHODS = ["fixed_5", "fixed_10", "fixed_20", "adaptive_raw", "adaptive_normalized", "random_partition"]


def ridge_path(x, y, xt, alphas):
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    # Padding coordinates are constant on some splits: do not divide by tiny noise.
    sd[sd < 1e-6] = 1.0
    z = (x - mean) / sd
    zt = (xt - mean) / sd
    ym = y.mean(axis=0)
    gram = z.T @ z
    rhs = z.T @ (y - ym)
    ev, vec = linalg.eigh(gram, check_finite=False)
    right = vec.T @ rhs
    left = zt @ vec
    return {str(a): left @ (right / (np.maximum(ev, 0)[:, None] + a)) + ym for a in alphas}


def return_from_y(yhat, mean, sd, close):
    c = mean.shape[1]
    return np.expm1(yhat.reshape(len(yhat), -1, c)[:, -1, 3] * sd[:, 3] + mean[:, 3] - np.log(close))


def fit_dataset(frame, pools, cutoff, cfg):
    eligible = []
    for p in pools:
        past = p.loc[p.date <= cutoff]
        close = past.close.to_numpy(float).copy()
        if "valid_ohlc" in past:
            close[~past.valid_ohlc.to_numpy(dtype=bool)] = np.nan
        eligible.append(close)
    assert all(len(a) >= cfg["prototype_length"] for a in eligible)
    prototypes, info = learn_prototypes(eligible, cfg)
    x, y, mu, sd, anchors, diag = make_dataset(frame, prototypes, cfg, METHODS)
    return x, y, mu, sd, anchors, diag, prototypes, info


def main():
    start = time.time()
    cfg_file = ROOT / "protocol.json"
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    frame = pd.read_csv(ROOT / "data" / (cfg["target"].replace(".", "_") + ".csv"))
    pools = [pd.read_csv(ROOT / "data" / (s.replace(".", "_") + ".csv")) for s in cfg["prototype_assets"]]
    # Freeze code + protocol hashes BEFORE prediction or model selection.
    snapshot = dict(protocol=cfg, protocol_sha256=hashlib.sha256(cfg_file.read_bytes()).hexdigest(),
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")},
                    python=sys.version, executable=sys.executable, platform=platform.platform(),
                    started_utc=pd.Timestamp.now(tz="UTC").isoformat(), numpy=np.__version__, pandas=pd.__version__)
    (out / "run_manifest.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    validation_frame = frame.loc[frame.date <= cfg["validation_end"]].reset_index(drop=True)
    print("Building validation P0, cutoff 2017-12-31", flush=True)
    x, y, mu, sd, anchors, diag, proto, info = fit_dataset(validation_frame, pools, "2017-12-31", cfg)
    dates = validation_frame.date.to_numpy()[anchors]
    labels_end = validation_frame.date.to_numpy()[anchors + cfg["horizon"]]
    weekdays = pd.to_datetime(dates).dayofweek.to_numpy()
    tr = labels_end <= "2017-12-31"
    va = (dates >= cfg["validation_start"]) & (labels_end <= cfg["validation_end"]) & (weekdays == 4)
    actual = validation_frame.close.to_numpy()[anchors + 5] / validation_frame.close.to_numpy()[anchors] - 1
    selected, validation = {}, []
    for method in METHODS:
        path = ridge_path(x[method][tr], y[tr], x[method][va], cfg["ridge_alphas"])
        ranking = []
        for alpha in cfg["ridge_alphas"]:
            r = return_from_y(path[str(alpha)], mu[va], sd[va], validation_frame.close.to_numpy()[anchors[va]])
            mse = float(np.mean((r - actual[va]) ** 2))
            validation.append(dict(method=method, alpha=alpha, n=int(va.sum()), rmse=float(np.sqrt(mse)),
                                   accuracy=float(np.mean((r > 0) == (actual[va] > 0)))))
            ranking.append((mse, -alpha, alpha))
        selected[method] = sorted(ranking)[0][2]
        print(f"validation {method}: alpha={selected[method]}", flush=True)
    pd.DataFrame(validation).to_csv(out / "validation.csv", index=False)
    (out / "selected_alphas.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    np.savez_compressed(out / "validation_prototypes.npz", prototypes=proto)
    records, folds, diag_rows, example_rows = [], [], [], []
    years = sorted(set(int(s[:4]) for s in frame.date if s >= cfg["test_start"]))
    for year in years:
        cutoff = f"{year-1}-12-31"
        # Only five following bars needed to score year-end observations; no future used in features/P0.
        anchor_limit = min(f"{year}-12-31", cfg["test_end"])
        last_anchor = int(np.flatnonzero(frame.date.to_numpy() <= anchor_limit)[-1])
        current = frame.iloc[:min(len(frame), last_anchor + cfg["horizon"] + 1)].copy()
        print(f"{year}: fitting P0 and feature cache", flush=True)
        x, y, mu, sd, anchors, diag, proto, info = fit_dataset(current, pools, cutoff, cfg)
        dates = current.date.to_numpy()[anchors]
        labels_end = current.date.to_numpy()[anchors + cfg["horizon"]]
        weekday = pd.to_datetime(dates).dayofweek.to_numpy()
        tr = labels_end <= cutoff
        te = (dates >= f"{year}-01-01") & (dates <= anchor_limit) & (weekday == cfg["signal_weekday"])
        te &= labels_end <= cfg["test_end"]
        if not te.any():
            continue
        assert labels_end[tr].max() <= cutoff < dates[te].min()
        assert anchors[tr].max() + cfg["horizon"] < anchors[te].min()
        close_now = current.close.to_numpy()[anchors[te]]
        actual = current.close.to_numpy()[anchors[te] + cfg["horizon"]] / close_now - 1
        predictions, y_predictions = {}, {}
        for method in METHODS:
            yhat = ridge_path(x[method][tr], y[tr], x[method][te], [selected[method]])[str(selected[method])]
            predictions[method] = return_from_y(yhat, mu[te], sd[te], close_now)
            y_predictions[method] = yhat
        rng = np.random.default_rng(cfg["prototype_seed"] + year)
        shuffled = y[tr][rng.permutation(tr.sum())]
        yhat = ridge_path(x["adaptive_normalized"][tr], shuffled, x["adaptive_normalized"][te],
                         [selected["adaptive_normalized"]])[str(selected["adaptive_normalized"])]
        predictions["shuffled_labels"] = return_from_y(yhat, mu[te], sd[te], close_now)
        raw_return = current.close.to_numpy()[anchors + 5] / current.close.to_numpy()[anchors] - 1
        predictions["training_mean"] = np.full(te.sum(), raw_return[tr].mean())
        predictions["always_up"] = np.full(te.sum(), 1e-9)
        predictions["zero_return"] = np.zeros(te.sum())
        predictions["momentum_20"] = close_now / current.close.to_numpy()[anchors[te] - 20] - 1
        test_ids = np.flatnonzero(te)
        for method, pred in predictions.items():
            for k, ix in enumerate(test_ids):
                record = dict(method=method, year=year, date=dates[ix], anchor=int(anchors[ix]),
                              label_end=labels_end[ix], train_cutoff=cutoff,
                              prediction=float(pred[k]), actual=float(actual[k]),
                              position=int(pred[k] > 0), correct=int((pred[k] > 0) == (actual[k] > 0)))
                if method in y_predictions:
                    yh = y_predictions[method][k].reshape(5, len(cfg["channels"]))
                    physical = yh * sd[ix] + mu[ix]
                    # Monotone log transforms preserve OHLC/non-negative checks without exp overflow.
                    bad_ohlc = (physical[:, 1] < physical[:, [0, 2, 3]].max(axis=1)) | (physical[:, 2] > physical[:, [0, 1, 3]].min(axis=1))
                    record.update(normalized_5d_mse=float(np.mean((yh.reshape(-1) - y[ix]) ** 2)),
                                  ohlc_invalid_bars=int(bad_ohlc.sum()),
                                  volume_amount_negative_bars=int((physical[:, 4:] < 0).any(axis=1).sum()))
                records.append(record)
        for ix in test_ids:
            for method in ["raw", "normalized"]:
                ls = diag[ix][method]
                diag_rows.append(dict(year=year, date=dates[ix], method=method, patches=len(ls),
                                      min_length=int(min(ls)), max_length=int(max(ls)),
                                      mean_length=float(np.mean(ls)), lengths=json.dumps(ls)))
        example_rows.append(dict(year=year, date=dates[test_ids[-1]], anchor=int(anchors[test_ids[-1]]),
                                 raw=diag[test_ids[-1]]["raw"], normalized=diag[test_ids[-1]]["normalized"]))
        folds.append(dict(year=year, train_cutoff=cutoff, prototype_cutoff=cutoff, scaler_cutoff=cutoff,
                          train_n=int(tr.sum()), test_n=int(te.sum()),
                          last_training_anchor=dates[tr][-1], last_training_label=labels_end[tr][-1],
                          test_start=dates[te][0], test_end=dates[te][-1],
                          prototype_metadata=info))
        np.savez_compressed(out / f"prototypes_{year}.npz", prototypes=proto)
        pd.DataFrame(records).to_csv(out / "predictions.csv", index=False)
        pd.DataFrame(diag_rows).to_csv(out / "segmentation_diagnostics.csv", index=False)
        (out / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
        print(f"{year}: {tr.sum()} train / {te.sum()} test; elapsed {time.time()-start:.1f}s", flush=True)
    (out / "segmentation_examples.json").write_text(json.dumps(example_rows, indent=2), encoding="utf-8")
    snapshot["elapsed_seconds"] = time.time() - start
    snapshot["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    (out / "run_manifest.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print("Finished", flush=True)


if __name__ == "__main__":
    main()

"""Public OHLCV fallback; archive original responses, never synthesize amount."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
import time
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent


def fetch(task):
    secid, year = task
    symbol = ("sh" if secid.startswith("1.") else "sz") + secid.split(".")[1]
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,{year}-01-01,{min(year+5,2027)}-01-01,2000,qfq"
    rawfile = ROOT / "data" / f"{secid.replace('.', '_')}_tencent_{year}.json"
    if rawfile.exists():
        raw = rawfile.read_bytes()
        rows = json.loads(raw)["data"][symbol]["day"]
        return task, rows, dict(url=url, file=rawfile.name, sha256=hashlib.sha256(raw).hexdigest(),
                               retrieved_utc=dt.datetime.fromtimestamp(rawfile.stat().st_mtime, dt.timezone.utc).isoformat())
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            data = r.json()["data"][symbol]
            rows = data["day"]
            assert len(rows) < 2000, "request may have been truncated"
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt+1)
    stem = secid.replace(".", "_")
    rawfile = ROOT / "data" / f"{stem}_tencent_{year}.json"
    rawfile.write_bytes(r.content)
    print(f"download {symbol} {year}: {len(rows)} bars", flush=True)
    return task, rows, dict(url=url, file=rawfile.name, sha256=hashlib.sha256(r.content).hexdigest(),
                           retrieved_utc=dt.datetime.now(dt.timezone.utc).isoformat())


def main():
    cfg = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    assets = [cfg["target"], *cfg["prototype_assets"]]
    groups = {s: [] for s in assets}
    sources = {s: [] for s in assets}
    tasks = [(s, y) for s in assets for y in [2010, 2015, 2020, 2025]]
    for (secid, year), rows, source in ThreadPoolExecutor(3).map(fetch, tasks):
        groups[secid].extend([r[:6] for r in rows])
        sources[secid].append(source)
    manifest = []
    for secid in assets:
        df = pd.DataFrame(groups[secid], columns=["date", "open", "close", "high", "low", "volume"])
        df = df[["date", *cfg["channels"]]]
        df = df.drop_duplicates().sort_values("date").reset_index(drop=True)
        df = df[(df.date >= "2010-01-01") & (df.date <= cfg["test_end"])].reset_index(drop=True)
        for c in cfg["channels"]:
            df[c] = pd.to_numeric(df[c], errors="raise")
        assert df.date.is_unique and df.date.is_monotonic_increasing
        assert df.notna().all().all()
        assert (df.iloc[:, 1:5] > 0).all().all() and (df.volume >= 0).all()
        # Historical index values may be rounded inconsistently by 0.01 point.
        tol = 0.011
        bad = (df.high + tol < df[["open", "close", "low"]].max(axis=1)) | (df.low - tol > df[["open", "close", "high"]].min(axis=1))
        df["valid_ohlc"] = ~bad
        strict = (df.high < df[["open", "close", "low"]].max(axis=1)) | (df.low > df[["open", "close", "high"]].min(axis=1))
        file = out / (secid.replace(".", "_") + ".csv")
        df.to_csv(file, index=False, lineterminator="\n")
        manifest.append(dict(secid=secid, rows=len(df), start=df.date.iloc[0], end=df.date.iloc[-1],
                             sources=sources[secid], csv_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                             duplicate_dates=0, missing_values=0, material_ohlc_violations=int(bad.sum()),
                             invalid_ohlc_dates=df.loc[bad, "date"].tolist(),
                             rounding_ohlc_violations=int(strict.sum()), rounding_tolerance=tol,
                             columns=cfg["channels"], adjustment="index raw day branch returned by qfq endpoint; no constituent-level adjustment applied",
                             volume_unit="provider-native; checked against initially received Eastmoney volume on 2010-01-04: 66101080 for CSI300",
                             amount="unavailable, excluded from model", vintage="historical download, not a point-in-time revision archive"))
        print(f"frozen {secid}: {len(df)} bars, {df.date.iloc[0]} -> {df.date.iloc[-1]}, invalid OHLC: {bad.sum()}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

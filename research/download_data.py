"""Freeze public index bars and their provenance; no account access."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import time
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    cfg = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    manifest = []
    for secid in [cfg["target"], *cfg["prototype_assets"]]:
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = dict(secid=secid, klt="101", fqt="0", beg=cfg["data_start"],
                      end=cfg["data_end"], fields1="f1,f2,f3,f4,f5,f6",
                      fields2="f51,f52,f53,f54,f55,f56,f57", lmt="10000")
        stem = secid.replace(".", "_")
        for attempt in range(5):
            try:
                response = requests.get(url, params=params, timeout=20)
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                print(f"retry {secid}, attempt {attempt+2}", flush=True)
                time.sleep(1 + attempt)
        raw = response.content
        (out / f"{stem}_raw.json").write_bytes(raw)
        payload = response.json()["data"]
        assert payload and payload["code"] == secid.split(".")[1]
        frame = pd.DataFrame([x.split(",") for x in payload["klines"]],
                             columns=["date", "open", "close", "high", "low", "volume", "amount"])
        frame = frame[["date", "open", "high", "low", "close", "volume", "amount"]]
        for c in frame.columns[1:]:
            frame[c] = pd.to_numeric(frame[c], errors="raise")
        assert frame.date.is_unique and frame.date.is_monotonic_increasing
        assert frame.notna().all().all()
        assert (frame.iloc[:, 1:5] > 0).all().all()
        assert (frame.high >= frame[["open", "close", "low"]].max(axis=1)).all()
        assert (frame.low <= frame[["open", "close", "high"]].min(axis=1)).all()
        assert (frame[["volume", "amount"]] >= 0).all().all()
        file = out / f"{stem}.csv"
        frame.to_csv(file, index=False, lineterminator="\n")
        record = dict(secid=secid, name=payload["name"], rows=len(frame),
                      start=frame.date.iloc[0], end=frame.date.iloc[-1],
                      requested_url=response.url, retrieved_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      raw_sha256=hashlib.sha256(raw).hexdigest(),
                      csv_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                      missing_values=0, duplicate_dates=0, ohlc_violations=0,
                      volume_unit="provider native units; no conversion is needed for per-window log normalization",
                      vintage="downloaded historical series; no point-in-time revision archive")
        manifest.append(record)
        print(f"{secid}: {len(frame)} bars, {record['start']} through {record['end']}", flush=True)
        time.sleep(0.15)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

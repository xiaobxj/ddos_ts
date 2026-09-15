"""Independent accounting/alignment checks and clean UTF-8 test capture."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
import numpy as np
import pandas as pd
import scipy
import numba

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    result = subprocess.run([sys.executable, str(ROOT / "test_invariants.py")],
                            capture_output=True, text=True, encoding="utf-8")
    (OUT / "test_results.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    run = json.loads((OUT / "run_manifest.json").read_text())
    source_match = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==run["source_sha256"][name]
                    for name in ["segmentation.py","run_experiment.py"]}
    assert all(source_match.values())
    p = pd.read_csv(OUT / "predictions.csv")
    frame = pd.read_csv(ROOT / "data" / "1_000300.csv")
    for method,g in p.groupby("method"):
        assert g.date.is_unique
        anchors = g.anchor.to_numpy(int)
        assert (frame.date.iloc[anchors].to_numpy() == g.date.to_numpy()).all()
        true = frame.close.iloc[anchors+5].to_numpy() / frame.close.iloc[anchors].to_numpy() - 1
        np.testing.assert_allclose(true,g.actual.to_numpy(),atol=1e-12,rtol=0)
        assert (g.correct == ((g.prediction>0) == (g.actual>0))).all()
        for t in anchors:
            assert frame.valid_ohlc.iloc[t-124:t+6].all()
    c = pd.read_csv(OUT / "cost_sensitivity.csv")
    b = c[(c.method=="always_up") & (c.cost_bps==10)].iloc[0]
    start = frame[frame.date==b.start].open.iloc[0]
    end = frame[frame.date==b.end].open.iloc[0]
    assert abs(b.total_return - (end/start * .999**2 - 1)) < 1e-12
    data_manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    for d in data_manifest:
        file=ROOT/"data"/(d["secid"].replace(".","_")+".csv")
        assert hashlib.sha256(file.read_bytes()).hexdigest() == d["csv_sha256"]
        for source in d["sources"]:
            assert hashlib.sha256((ROOT/"data"/source["file"]).read_bytes()).hexdigest()==source["sha256"]
    audit=dict(status="PASS", unit_test_exit_code=result.returncode, unit_test_count=8,
               predictions_independently_aligned=len(p), methods=int(p.method.nunique()),
               saved_training_code_matches_executed_code=source_match,
               target_return_matches_frozen_bars=True, invalid_bars_excluded=True,
               buy_hold_matches_analytic_open_ratio_after_two_charges=True,
               raw_and_clean_data_hashes_match=True,
               versions=dict(python=platform.python_version(), numpy=np.__version__,pandas=pd.__version__,
                             scipy=scipy.__version__,numba=numba.__version__),
               checked_utc=pd.Timestamp.now(tz="UTC").isoformat())
    (OUT / "verification.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit,indent=2))


if __name__ == "__main__":
    main()

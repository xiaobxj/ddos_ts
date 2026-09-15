# Round 16: annual transfer and fixed simple-feature comparison

Completed comparison of frozen MSE20 features and a fixed 25-coordinate OHLCV summary using the identical ridge-logistic solver from round 15. Neither representation passes the prespecified cross-period screen.

Read the [Chinese report](results/第十六轮测试报告.md), [frozen protocol](protocol.json), [historical date-use audit](results/historical_usage_audit.json), and [verification](results/verification.json).

Six annual folds cover 120 previously inspected signals in 2015-2017 and 141 in 2018-2020. Every backbone and classifier uses only labels completed by its own cutoff. The study design remains retrospective. All later round 15 MSE-probe coefficients and predictions are retained.

Fifteen new heads converge in 55 Newton updates: nine earlier learned-feature heads and six deterministic simple-feature heads. Nine later heads are reused. No neural training, feature/hyperparameter search, threshold fitting or deployment occurs. The simple model has one deterministic fit per cutoff; it is not replicated as three artificial seeds.

Execution order in a separate workspace containing the frozen prior evidence:

```text
prepare16.py
contract16.py
train16.py
score16.py
evaluate16.py
verify16.py
interpret16.py
figures16.py
delivery16.py --freeze
```

Core stages use `research_v4/.venv_gpu/Scripts/python.exe` (Python 3.13, torch 2.6.0+cu124, SciPy 1.18.1). The default Python can render figures and check delivery. Existing one-time results are protected from overwriting.

`interpret16.py` produces explicitly post-evaluation feature-shift and mean-logit accounting descriptions. These never change coefficients, features, forecasts or decisions. The report is written after verification and included in the final delivery hash manifest.

Read-only final check:

```powershell
python research_v16/delivery16.py
```

All evaluated dates have been inspected in prior research. Results do not establish independent confirmation, full paper replication, or trading profitability.

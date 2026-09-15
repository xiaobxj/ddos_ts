# Round 15: frozen-feature logistic probes

Completed controlled comparison of identical ridge-logistic heads on the frozen MSE20 and BCE20 representations. Both families fail the prespecified descriptive screen. The original MSE20 remains the research reference.

Read the [Chinese report](results/第十五轮测试报告.md), [frozen protocol](protocol.json), and [independent verification](results/verification.json).

The 18 final heads use 25 standardized decoded features plus an unpenalized intercept, lambda 0.01, a deterministic float64 Newton solver, and a fixed 0.5 threshold. Training statistics never use validation features. All 18 heads converge in 66 total updates; there are zero neural training steps. No alternative solver coefficients enter forecasts.

The full compatibility preflight is preserved in `../research_v15_preflight`. Disabling parameter gradient flags slightly changes floating-point native outputs under this torch build. Historical forecasts are replayed with their original flags under inference mode; new features remain frozen. All corrected head coefficients match the preflight bit for bit.

In a separate workspace containing the frozen historical evidence, the execution order is:

```text
prepare15.py
contract15.py
extract15.py
fit15.py
score15.py
evaluate15.py
verify15.py
interpret15.py
figures15.py
delivery15.py --freeze
```

Core stages use `research_v4/.venv_gpu/Scripts/python.exe` (Python 3.13, torch 2.6.0+cu124, SciPy 1.18.1). The existing default Python can render the figures and check final delivery. One-time core writers reject existing results. The report is authored after verification; it is checked and hashed by delivery.

For a read-only check of the completed delivery:

```powershell
python research_v15/delivery15.py
```

The same 141 historical weeks have been inspected repeatedly. Results are exploratory and do not establish independent confirmation, full paper replication, or trading profitability.

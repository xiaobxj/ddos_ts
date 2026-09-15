"""Independent provenance, label, rolling-boundary, probability and accounting checks."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from core import ROOT,V1
from run import previous_evidence

OUT=ROOT/'results'


def main():
    tests=subprocess.run([sys.executable,str(ROOT/'test_core.py')],capture_output=True,text=True,encoding='utf-8')
    (OUT/'test_results.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    assert tests.returncode==0,tests.stderr
    run=json.loads((OUT/'run_manifest.json').read_text())
    selected=json.loads((OUT/'selection.json').read_text())
    assert previous_evidence()==run['v1_sha256']
    for name in ['core.py','run.py']:
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==run['code_sha256'][name]
    frame=pd.read_csv(V1/'data/1_000300.csv')
    pred=pd.read_csv(OUT/'predictions.csv')
    obs=pd.read_csv(OUT/'observation_table.csv')
    source=pd.read_csv(V1/'results/predictions.csv')
    anchors=source[source.method=='adaptive_normalized'].anchor.to_numpy(int)
    friday=np.flatnonzero(pd.to_datetime(frame.date).dt.dayofweek.to_numpy()==4)
    next_fri=dict(zip(friday[:-1],friday[1:]))
    for method,g in pred.groupby('method'):
        g=g.sort_values('anchor')
        np.testing.assert_array_equal(g.anchor.to_numpy(int),anchors)
        a=g.anchor.to_numpy(int)
        exits=np.array([next_fri[t]+1 for t in a])
        np.testing.assert_array_equal(g.exit.to_numpy(int),exits)
        np.testing.assert_array_equal(g.entry.to_numpy(int),a+1)
        expected=frame.open.to_numpy()[exits]/frame.open.to_numpy()[a+1]-1
        np.testing.assert_allclose(g.actual,expected,atol=1e-12,rtol=0)
        assert (g.correct.to_numpy()==(g.position.to_numpy()==(expected>0))).all()
        assert np.isfinite(g.predicted_return).all()
        assert (g.cutoff<g.date).all()
        if method.startswith(('k_logit','l_logit')):
            assert g.probability.between(0,1).all()
            np.testing.assert_array_equal(g.position,(g.probability>.5).astype(int))
    folds=json.loads((OUT/'folds.json').read_text())
    for fold in folds:
        assert fold['last_label']<=fold['cutoff']<fold['first_signal']
        assert fold['last_signal']<=fold['period_end']
        train=obs.completed<=fold['cutoff']
        if fold['memory_years']:
            lower=(pd.Timestamp(fold['cutoff'])+pd.Timedelta(days=1)-pd.DateOffset(years=fold['memory_years'])).strftime('%Y-%m-%d')
            assert fold['first_train']>=lower
            train &= obs.date>=lower
        assert train.sum()==fold['train_n']
        assert obs.completed[train].max()==fold['last_label']
    cost=pd.read_csv(OUT/'cost_sensitivity.csv')
    curves=pd.read_csv(OUT/'equity_curves_10bps.csv')
    weekly=pd.read_csv(OUT/'weekly_net_returns_10bps.csv')
    oldcost=pd.read_csv(V1/'results/cost_sensitivity.csv')
    for name,g in cost.groupby('method'):
        g=g.sort_values('cost_bps')
        assert (np.diff(g.total_return)<=1e-12).all()
        daily=curves[curves.method==name]
        net=weekly[weekly.method==name].net_return.to_numpy()
        assert abs(daily.wealth.iloc[-1]-np.prod(1+net))<1e-10
        assert abs(daily.wealth.iloc[-1]-1-g[g.cost_bps==10].total_return.iloc[0])<1e-10
    for name,oldname in [('v1_adaptive_normalized','adaptive_normalized'),('v1_fixed_10','fixed_10'),('always_up','always_up')]:
        new=cost[cost.method==name].sort_values('cost_bps')
        previous=oldcost[oldcost.method==oldname].sort_values('cost_bps')
        np.testing.assert_allclose(new.total_return,previous.total_return,atol=1e-12,rtol=0)
    validation=pd.read_csv(OUT/'validation_predictions.csv')
    summaries=pd.read_csv(OUT/'validation_summary.csv')
    common_dates=None
    for name,choice in selected['selected_lambdas'].items():
        rows=validation[(validation.method==name)&np.isclose(validation.lam,choice['lam'])].sort_values('date')
        assert rows.date.max()<='2020-12-31'
        if common_dates is None:common_dates=rows.date.tolist()
        assert common_dates==rows.date.tolist()
        assert (rows.cutoff<rows.date).all()
        assert len(rows)==141
    chosen=selected['validation_selected_pipeline']
    chosen_record=selected['selected_lambdas'][chosen]
    for name,choice in selected['selected_lambdas'].items():
        if choice['model']=='ridge' and choice['feature']!='legacy':
            assert chosen_record['return_rmse']<=choice['return_rmse']+1e-14
    audit=dict(status='PASS',unit_tests=8,predictions=len(pred),variants=int(pred.method.nunique()),
               model_variants=12,scored_weekly_observations=272,validation_weekly_observations=141,
               refit_folds_checked=len(folds),old_files_preserved=len(run['v1_sha256']),
               executed_core_source_hashes_match=True,labels_recomputed_from_frozen_opens=True,
               all_folds_purged=True,rolling_window_boundaries_match=True,old_backtests_reconciled=True,
               daily_weekly_wealth_reconciled=True,cost_monotonicity=True,
               validation_selection_recomputed=True,
               chosen_pipeline=chosen,checked_utc=pd.Timestamp.now(tz='UTC').isoformat())
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()

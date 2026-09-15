"""Cross-check final report, causal chronology, diagnostics and immutable evidence."""
from pathlib import Path
import argparse,json,hashlib,re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(freeze):
    verification=read(OUT/'verification.json');prep=read(OUT/'preparation_manifest.json')
    assert verification['status']=='PASS' and verification['model_states_replayed']==18
    assert verification['epoch_orders_and_batches']==180 and verification['training_loss_passes']==9
    assert verification['rolling_seed_forecasts']==645 and verification['outer_base_seed_forecasts']==423
    assert verification['outer_transformed_seed_forecasts']==1269 and verification['calibration_memberships']==407
    assert verification['coefficients_recomputed']==3 and verification['leave_year_coefficients_recomputed']==9
    assert verification['primary_comparisons_recomputed']==4
    phases=['preparation','training','rolling','calibration','scoring','evaluation']
    prior_finish=None
    for phase in phases:
        run=read(OUT/f'{phase}_manifest.json')
        assert run['protocol_sha256']==prep['protocol_sha256'] and run['source_sha256']==prep['source_sha256']
        assert run['input_sha256']==prep['input_sha256'] and run['started_utc']<run['finished_utc']
        if prior_finish:assert prior_finish<=run['started_utc']
        prior_finish=run['finished_utc']
        for name,digest in run.get('artifacts',{}).items():assert sha(ROOT/name if name.startswith('results') else OUT/name)==digest,name
    contract=read(OUT/'contract_verification.json');assert contract['status']=='PASS' and len(contract['tests'])==4
    assert contract['completed_utc']<=read(OUT/'training_manifest.json')['started_utc']
    for mapping in ['old_evidence','source_sha256','input_sha256']:
        for name,digest in prep[mapping].items():assert sha(ROOT.parent/name)==digest,name
    for name,digest in prep['local_sha256'].items():assert sha(ROOT/name)==digest,name
    assert len(prep['old_evidence'])==2341 and sha(ROOT/'protocol.json')==prep['protocol_sha256']
    for r in read(OUT/'inner_models.json')+read(OUT/'archived_models.json'):assert sha(ROOT.parent/r['project_file'])==r['sha256']
    budgets=pd.read_csv(OUT/'training_budgets.csv')
    assert budgets.total_steps_all_seeds.sum()==1860 and budgets.presentations_all_seeds.sum()==222060
    member=pd.read_csv(OUT/'calibration_membership.csv')
    assert len(member)==407 and member.row_index.nunique()==214
    for cutoff,g in member.groupby('outer_cutoff'):
        assert g.joint_completed.le(cutoff).all() and g.date.le(cutoff).all() and g.inner_cutoff.lt(g.date).all()
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv').set_index('method')
    ensemble=pd.read_csv(OUT/'outer_ensemble_predictions.csv')
    seeds=pd.read_csv(OUT/'outer_seed_predictions.csv')
    for method,g in ensemble.groupby('method'):
        assert len(g)==141 and g.date.is_unique
        r=metrics.loc[method];y=g.actual.to_numpy();p=g.predicted_return.to_numpy()
        assert abs(r.mse-float(np.mean((p-y)**2)))<1e-14
        assert abs(r.accuracy-float(((p>0)==(y>0)).mean()))<1e-14
        assert int(r.correct_directions)==int(((p>0)==(y>0)).sum())
        expected=seeds[seeds.method.eq(method)].groupby('date').predicted_return.mean()
        np.testing.assert_allclose(expected,g.sort_values('date').predicted_return,rtol=0,atol=1e-14)
    assessment=read(OUT/'assessment.json');assert not assessment['descriptive_screen_pass'] and not any(assessment['flags'].values())
    assert not assessment['independent_confirmation'] and not assessment['strategy_promotion']
    assert abs(assessment['mse_skill_vs_original']-(1-metrics.loc['rolling_shrink','mse']/metrics.loc['archived20','mse']))<1e-12
    diagnostic=read(OUT/'direction_diagnostic.json')
    assert diagnostic['status']=='PASS' and not diagnostic['additional_fitting'] and not diagnostic['new_predictions']
    assert not diagnostic['outer_optimal_alpha_estimated'] and diagnostic['input_sha256']==sha(OUT/'outer_ensemble_predictions.csv')
    for name,digest in diagnostic['files'].items():assert sha(OUT/name)==digest
    changes=pd.read_csv(OUT/'direction_changes.csv');thresholds=pd.read_csv(OUT/'implied_direction_thresholds.csv')
    base=ensemble[ensemble.method.eq('archived20')].set_index('date')
    for r in changes.itertuples():
        g=ensemble[ensemble.method.eq(r.method)].set_index('date')
        if r.year!='all':g=g[g.index.str.startswith(str(r.year))]
        b=base.loc[g.index];correct=(g.predicted_return>0)==(g.actual>0);old=(b.predicted_return>0)==(b.actual>0)
        assert r.n==len(g) and r.correct_to_wrong==int((old&~correct).sum()) and r.wrong_to_correct==int((~old&correct).sum())
        assert r.changed==r.correct_to_wrong+r.wrong_to_correct
        assert r.candidate_correct==r.original_correct-r.correct_to_wrong+r.wrong_to_correct
        assert r.up_to_non_up==0 and r.non_up_to_up==r.changed
    for r in thresholds.itertuples():
        assert r.training_mean>0
        if r.alpha==0:assert pd.isna(r.original_prediction_up_threshold) and r.all_up_due_to_zero_alpha
        else:assert abs(r.original_prediction_up_threshold-r.training_mean*(1-1/r.alpha))<1e-14
    path=OUT/'第十三轮测试报告.md';report=path.read_text(encoding='utf-8')
    assert report.startswith('# 第十三轮方法测试：') and '\ufffd' not in report
    assert '不采用滚动均值收缩' in report and '不能当作独立确认' in report
    for link in re.findall(r'\]\(([^)]+)\)',report):
        if not link.startswith(('http://','https://')):assert (OUT/link).resolve().is_file(),link
    for r in metrics.itertuples():
        for value in [f'{r.accuracy*100:.2f}%',f'{r.rmse*1e4:.2f}',f'{r.mae*1e4:.2f}']:assert value in report,value
    for r in read(OUT/'coefficients.json'):
        assert f"{r['raw_alpha']:.6f}" in report and f"{r['alpha']:.6f}" in report
    for r in read(OUT/'primary_comparisons.json'):
        for value in [f"{r['mse']['holm_adjusted_p']:.4f}",f"{r['mse']['difference']*1e8:+.2f}"]:assert value in report,value
    if freeze:
        assert not (OUT/'delivery_manifest.json').exists(),'Preserve delivered evidence'
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
               if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json'}
        result=dict(experiment=13,report=str(path.relative_to(ROOT)),completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),
            new_fits=9,new_epochs=180,neural_states_replayed=18,outer_validation_weeks=141,coefficients=3,
            previous_files_preserved=2341,descriptive_screen_pass=False,files=files)
        (OUT/'delivery_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    delivered=read(OUT/'delivery_manifest.json')
    for name,digest in delivered['files'].items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(status='PASS',hashed_files=len(delivered['files']),report_characters=len(report),
        previous_files_preserved=2341,neural_states_replayed=18,descriptive_screen_pass=False),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');main(parser.parse_args().freeze)

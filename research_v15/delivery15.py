"""Read-only final evidence checks, plus a one-time delivery manifest freeze."""
from pathlib import Path
import argparse,json,hashlib,re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
PROJECT=ROOT.parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(freeze):
    v=read(OUT/'verification.json');prep=read(OUT/'preparation_manifest.json');protocol=read(ROOT/'protocol.json')
    expected=dict(model_states_replayed=18,training_feature_rows=34584,validation_feature_rows=846,independent_solutions=18,
        newton_solutions_replayed=18,new_seed_predictions=846,archived_seed_predictions=846,ensemble_methods=6,validation_weeks=141,
        primary_comparisons_recomputed=7,reliability_bins_recomputed=25,metrics_independently_verified=36,
        previous_files_preserved=2527,previous_round_files_preserved=2479,preflight_files_preserved=48)
    assert v['status']=='PASS' and all(v[k]==value for k,value in expected.items())
    assert v['preflight_coefficients_bitwise_equal'] and len(prep['old_evidence'])==2527
    assert sha(ROOT/'protocol.json')==prep['protocol_sha256']==v['protocol_sha256']
    for mapping in ['old_evidence','source_sha256','input_sha256']:
        for name,digest in prep[mapping].items():assert sha(PROJECT/name)==digest,name
    for name,digest in prep['local_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in v['artifacts'].items():assert sha(ROOT/name)==digest,name
    assert prep['source_sha256']==v['source_sha256']
    contract=read(OUT/'contract_verification.json');assert contract['status']=='PASS' and len(contract['tests'])==4
    assert contract['protocol_sha256']==prep['protocol_sha256'] and contract['source_sha256']==prep['source_sha256']
    finish=prep['finished_utc'];assert finish<=contract['completed_utc'];finish=contract['completed_utc']
    for phase in ['extraction','fitting','scoring','evaluation']:
        run=read(OUT/f'{phase}_manifest.json');assert finish<=run['started_utc']<run['finished_utc'];finish=run['finished_utc']
        for key in ['protocol_sha256','source_sha256','input_sha256']:assert run[key]==prep[key]
        for name,digest in run['artifacts'].items():assert sha(ROOT/name)==digest,name
    assert finish<=v['completed_utc']
    fitting=read(OUT/'fitting_manifest.json');heads=read(OUT/'heads.json');trace=pd.read_csv(OUT/'solver_trace.csv')
    assert fitting['all_converged'] and fitting['neural_training_steps']==0 and not fitting['validation_used']
    assert fitting['preflight_coefficients_bitwise_equal'] and len(heads)==18 and fitting['primary_heads']==18
    assert sum(h['iterations'] for h in heads)==fitting['total_newton_iterations']==66 and len(trace)==84
    preflight=read(PROJECT/'research_v15_preflight/results/heads.json')
    for h,old in zip(heads,preflight):
        assert all(h[k]==old[k] for k in ['family','cutoff','seed','coefficients','iterations','objective','gradient_inf'])
        assert sha(PROJECT/h['project_file'])==h['sha256'] and len(h['coefficients'])==26
        assert h['gradient_inf']<=protocol['probe']['gradient_infinity_tolerance'] and h['hessian_min_eigenvalue']>0
    independent=pd.read_csv(OUT/'independent_solver_verification.csv');settings=protocol['independent_solver']
    assert len(independent)==18 and independent.alternate_success.all()
    assert independent.objective_absolute_gap.max()<=settings['objective_absolute_tolerance']
    assert independent.maximum_training_probability_gap.max()<=settings['training_probability_absolute_tolerance']
    assert independent.alternate_gradient_inf.max()<=settings['gradient_infinity_tolerance']
    metadata=read(OUT/'training_metadata.json');labels=pd.read_csv(OUT/'training_rows.csv')
    for cutoff,g in labels.groupby('cutoff'):
        assert g.joint_completed.le(cutoff).all() and len(g)==metadata[cutoff]['train_n']
        assert int(g.direction.sum())==metadata[cutoff]['up_count'] and abs(g.direction.mean()-metadata[cutoff]['frequency'])<1e-15
    ensemble=pd.read_csv(OUT/'all_ensemble_predictions.csv');seed=pd.read_csv(OUT/'all_seed_predictions.csv')
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv').set_index('method');assert len(ensemble)==846 and len(seed)==1692 and len(metrics)==6
    for method,g in ensemble.groupby('method'):
        assert len(g)==141 and g.date.is_unique;y=g.actual_up.to_numpy(bool);up=g.direction_up.to_numpy(bool);r=metrics.loc[method]
        assert int((up==y).sum())==r.correct_directions and abs((up==y).mean()-r.accuracy)<1e-14
        assert abs(1-r.accuracy-r.direction_error)<1e-14
        if method=='archived20':
            assert g.probability.isna().all() and pd.isna(r.brier) and pd.isna(r.log_loss);np.testing.assert_array_equal(up,g.score>0)
        else:
            p=g.probability.to_numpy();assert np.all((p>=0)&(p<=1));np.testing.assert_array_equal(up,p>.5)
            assert abs(np.square(p-y.astype(float)).mean()-r.brier)<1e-14
    primary=read(OUT/'primary_comparisons.json');assert len(primary)==7
    for actual,expected_pair in zip(primary,protocol['primary_comparisons']):
        for key,value in expected_pair.items():assert actual[key]==value
        delta=metrics.loc[actual['candidate'],actual['metric']]-metrics.loc[actual['reference'],actual['metric']]
        assert abs(delta-actual['difference'])<1e-14
    assessments=read(OUT/'assessments.json');assert len(assessments)==2
    for result in assessments:
        assert len(result['flags'])==8 and result['descriptive_screen_pass']==all(result['flags'].values())
        assert not result['descriptive_screen_pass'] and not result['independent_confirmation'] and not result['strategy_promotion']
        assert abs(result['accuracy_difference_vs_original']-(metrics.loc[result['method'],'accuracy']-metrics.loc['archived20','accuracy']))<1e-14
        assert abs(result['brier_skill_vs_training_frequency']-(1-metrics.loc[result['method'],'brier']/metrics.loc['training_frequency','brier']))<1e-12
    train=pd.read_csv(OUT/'training_metrics.csv');summary=pd.read_csv(OUT/'training_fold_summary.csv')
    for r in summary.itertuples():
        g=train[train.family.eq(r.family)&train.cutoff.eq(r.cutoff)];assert len(g)==3
        for key in ['log_loss','brier','accuracy','native_accuracy','constant_log_loss','constant_brier']:
            assert abs(g[key].mean()-getattr(r,key))<1e-12
        assert abs((1-g.log_loss/g.constant_log_loss).mean()-r.log_loss_skill)<1e-12
    decomposition=pd.read_csv(OUT/'ensemble_brier_decomposition.csv');assert len(decomposition)==3
    for r in decomposition.itertuples():
        p=seed[seed.method.eq(r.method)].pivot(index='date',columns='seed',values='probability')
        y=ensemble[ensemble.method.eq(r.method)].set_index('date').loc[p.index].actual_up.to_numpy()[:,None]
        mean_seed=float(np.square(p.to_numpy()-y).mean());dispersion=float(p.to_numpy().var(axis=1).mean())
        assert abs(mean_seed-r.mean_seed_brier)<1e-14 and abs(dispersion-r.mean_seed_probability_variance)<1e-14
        assert abs(r.ensemble_brier-metrics.loc[r.method,'brier'])<1e-14
        assert abs(r.mean_seed_brier-r.ensemble_brier-r.mean_seed_probability_variance)<1e-14
    report_path=OUT/'第十五轮测试报告.md';report=report_path.read_text(encoding='utf-8')
    assert report.startswith('# 第十五轮方法测试：') and '\ufffd' not in report and '不能作为独立确认' in report
    assert prep['protocol_sha256'] in report and '3/8，未通过' in report and '2/8，未通过' in report
    for link in re.findall(r'\]\(([^)]+)\)',report):
        if not link.startswith(('http://','https://')):assert (OUT/link).resolve().is_file(),link
    for r in metrics.itertuples():
        assert f'{r.accuracy*100:.2f}%' in report and f'{r.balanced_accuracy*100:.2f}%' in report
        assert f'{r.auroc:.6f}' in report
        for value in [r.brier,r.log_loss]:
            if pd.notna(value):assert f'{value:.6f}' in report
    for r in summary.itertuples():
        assert f'{r.log_loss:.6f}' in report and f'{r.constant_log_loss:.6f}' in report
        if r.family=='probe_bce':assert f'{r.native_log_loss:.6f}' in report and f'{r.accuracy*100:.2f}%' in report
    for r in primary:assert f"{r['p']:.4f}" in report and f"{r['holm_adjusted_p']:.4f}" in report
    target=OUT/'delivery_manifest.json'
    if freeze:
        assert not target.exists(),'Preserve final delivery'
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json'}
        result=dict(experiment=15,completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),report=str(report_path.relative_to(ROOT)),
            primary_heads=18,total_newton_iterations=66,neural_training_steps=0,preflight_refits_preserved=18,
            model_states_replayed=18,independent_solutions=18,validation_weeks=141,previous_files_preserved=2527,
            previous_round_files_preserved=2479,preflight_files_preserved=48,
            descriptive_screen_pass={r['method']:r['descriptive_screen_pass'] for r in assessments},files=files)
        target.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    delivered=read(target)
    for name,digest in delivered['files'].items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(status='PASS',hashed_files=len(delivered['files']),report_characters=len(report),
        model_states_replayed=18,independent_solutions=18,previous_files_preserved=2527,
        descriptive_screen_pass=delivered['descriptive_screen_pass']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');main(parser.parse_args().freeze)

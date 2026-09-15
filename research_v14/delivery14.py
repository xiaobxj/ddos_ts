"""Final evidence, report and controlled-comparison checks."""
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
    v=read(OUT/'verification.json');prep=read(OUT/'preparation_manifest.json');protocol=read(ROOT/'protocol.json')
    assert v['status']=='PASS' and v['model_states_replayed']==18 and v['initial_states_verified']==9
    assert v['training_loss_passes']==81 and v['epoch_orders_and_batches']==180
    assert v['classifier_seed_predictions']==423 and v['archived_seed_predictions']==423
    assert v['ensemble_methods']==4 and v['validation_weeks']==141 and v['primary_comparisons_recomputed']==4
    assert v['reliability_bins_recomputed']==15 and v['previous_files_preserved']==2415
    assert len(prep['old_evidence'])==2415 and sha(ROOT/'protocol.json')==prep['protocol_sha256']
    for mapping in ['old_evidence','source_sha256','input_sha256']:
        for name,digest in prep[mapping].items():assert sha(ROOT.parent/name)==digest,name
    for name,digest in prep['local_sha256'].items():assert sha(ROOT/name)==digest,name
    contract=read(OUT/'contract_verification.json');assert contract['status']=='PASS' and len(contract['tests'])==5
    assert prep['finished_utc']<=contract['completed_utc'];finish=contract['completed_utc']
    for phase in ['training','scoring','evaluation']:
        run=read(OUT/f'{phase}_manifest.json')
        assert finish<=run['started_utc']<run['finished_utc'];finish=run['finished_utc']
        assert run['protocol_sha256']==prep['protocol_sha256'] and run['source_sha256']==prep['source_sha256']
        assert run['input_sha256']==prep['input_sha256']
        for name,digest in run['artifacts'].items():assert sha(OUT/name)==digest,name
    for r in read(OUT/'models.json')+read(OUT/'archived_models.json'):assert sha(ROOT.parent/r['project_file'])==r['sha256']
    budgets=pd.read_csv(OUT/'training_budgets.csv')
    assert budgets.total_steps_all_seeds.sum()==2820 and budgets.presentations_all_seeds.sum()==345840
    meta=read(OUT/'training_metadata.json');labels=pd.read_csv(OUT/'training_rows.csv')
    for cutoff,g in labels.groupby('cutoff'):
        assert g.joint_completed.le(cutoff).all()
        assert int(g.direction.sum())==meta[cutoff]['up_count'] and len(g)==meta[cutoff]['train_n']
        assert abs(g.direction.mean()-meta[cutoff]['frequency'])<1e-15
    ensemble=pd.read_csv(OUT/'all_ensemble_predictions.csv');seed=pd.read_csv(OUT/'all_seed_predictions.csv')
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv').set_index('method');years=pd.read_csv(OUT/'yearly_metrics.csv');seeds=pd.read_csv(OUT/'seed_metrics.csv')
    for method,g in ensemble.groupby('method'):
        assert len(g)==141 and g.date.is_unique
        y=g.actual_up.to_numpy(bool);up=g.direction_up.to_numpy(bool);r=metrics.loc[method]
        assert r.correct_directions==int((up==y).sum()) and abs(r.accuracy-float((up==y).mean()))<1e-14
        assert abs(r.direction_error-(1-r.accuracy))<1e-14
        if method=='archived20':assert g.probability.isna().all() and pd.isna(r.brier) and pd.isna(r.log_loss)
        else:
            p=g.probability.to_numpy();assert np.all((p>=0)&(p<=1))
            assert abs(r.brier-float(np.mean((p-y.astype(int))**2)))<1e-14
            np.testing.assert_array_equal(up,p>.5)
    primary=read(OUT/'primary_comparisons.json');assert len(primary)==4
    for actual,expected in zip(primary,protocol['primary_comparisons']):
        for k,value in expected.items():assert actual[k]==value
        difference=metrics.loc[actual['candidate'],actual['metric']]-metrics.loc[actual['reference'],actual['metric']]
        assert abs(difference-actual['difference'])<1e-14
    assessment=read(OUT/'assessment.json');assert len(assessment['flags'])==8
    assert assessment['descriptive_screen_pass']==all(assessment['flags'].values())
    assert not assessment['independent_confirmation'] and not assessment['strategy_promotion']
    assert abs(assessment['accuracy_difference_vs_original']-(metrics.loc['direction_bce','accuracy']-metrics.loc['archived20','accuracy']))<1e-14
    assert abs(assessment['brier_skill_vs_training_frequency']-(1-metrics.loc['direction_bce','brier']/metrics.loc['training_frequency','brier']))<1e-12
    changes=pd.read_csv(OUT/'direction_changes.csv')
    for r in changes.itertuples():
        g=ensemble[ensemble.method.eq('direction_bce')].set_index('date')
        if r.year!='all':g=g[g.index.str.startswith(str(r.year))]
        old=ensemble[ensemble.method.eq('archived20')].set_index('date').loc[g.index]
        correct=g.direction_up.eq(g.actual_up);prior=old.direction_up.eq(old.actual_up)
        assert r.changed==int(g.direction_up.ne(old.direction_up).sum())
        assert r.correct_to_wrong==int((prior&~correct).sum()) and r.wrong_to_correct==int((~prior&correct).sum())
        assert r.changed==r.correct_to_wrong+r.wrong_to_correct==r.up_to_non_up+r.non_up_to_up
    path=OUT/'第十四轮测试报告.md';report=path.read_text(encoding='utf-8')
    assert report.startswith('# 第十四轮方法测试：') and '\ufffd' not in report
    assert '不能作为独立确认' in report and '不能直接比较数值' in report
    for link in re.findall(r'\]\(([^)]+)\)',report):
        if not link.startswith(('http://','https://')):assert (OUT/link).resolve().is_file(),link
    for r in metrics.itertuples():
        assert f'{r.accuracy*100:.2f}%' in report and f'{r.balanced_accuracy*100:.2f}%' in report
        for value in [r.brier,r.log_loss]:
            if pd.notna(value):assert f'{value:.6f}' in report
    for r in primary:assert f"{r['holm_adjusted_p']:.4f}" in report
    training=pd.read_csv(OUT/'training_summary.csv')
    for mode,g in training.groupby('mode'):
        for key in ['log_loss','constant_log_loss','brier','constant_brier','auxiliary_mse','joint_loss']:
            assert f'{g[key].mean():.6f}' in report
    if freeze:
        assert not (OUT/'delivery_manifest.json').exists(),'Preserve final delivery'
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json'}
        result=dict(experiment=14,report=str(path.relative_to(ROOT)),completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),
            new_fits=9,new_epochs=180,neural_states_replayed=18,validation_weeks=141,
            previous_files_preserved=2415,descriptive_screen_pass=assessment['descriptive_screen_pass'],files=files)
        (OUT/'delivery_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    delivered=read(OUT/'delivery_manifest.json')
    for name,digest in delivered['files'].items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(status='PASS',hashed_files=len(delivered['files']),report_characters=len(report),
        previous_files_preserved=2415,neural_states_replayed=18,descriptive_screen_pass=assessment['descriptive_screen_pass']),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');main(p.parse_args().freeze)

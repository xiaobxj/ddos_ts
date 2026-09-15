"""Final report, timing, pairing, budget and immutable-evidence checks."""
from pathlib import Path
import argparse,hashlib,json,re
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(freeze):
    verification=read(OUT/'verification.json')
    assert verification['status']=='PASS' and verification['model_states_replayed']==36
    assert verification['physical_epoch_orders_and_boundaries']==540 and verification['training_loss_passes_replayed']==243
    assert verification['validation_seed_predictions']==1269 and verification['primary_comparisons_recomputed']==4
    preparation=read(OUT/'preparation_manifest.json');contract=read(OUT/'contract_verification.json')
    prefix=read(OUT/'reconstruction_manifest.json');continuation=read(OUT/'continuation_manifest.json');scoring=read(OUT/'scoring_manifest.json')
    assert contract['completed_utc']<=prefix['started_utc']<prefix['finished_utc']<=continuation['started_utc']<continuation['finished_utc']<=scoring['started_utc']
    assert prefix['exact_matches']==9 and continuation['continuations']==18
    starts=read(OUT/'paired_starting_states.json');prefixes=read(OUT/'prefix_models.json')
    assert len(starts)==18 and len(prefixes)==9
    for p in prefixes:
        group=[r for r in starts if r['cutoff']==p['cutoff'] and r['seed']==p['seed']]
        assert len(group)==2 and {r['schedule'] for r in group}=={'constant40','decay40'}
        for r in group:
            assert r['prefix_project_file']==p['project_file'] and r['prefix_sha256']==p['sha256']
            for key in ['model_sha256','optimizer_sha256','rng_sha256']:assert r['initial_'+key]==p[key]
    for r in pd.read_csv(OUT/'schedule_budgets.csv').itertuples():
        n={'2017-12-31':1678,'2018-12-31':1921,'2019-12-31':2165}[r.cutoff];steps=(n+127)//128
        second=.001 if r.schedule=='constant40' else .0001
        assert r.train_n==n and r.total_epochs==40 and r.steps_per_epoch==steps and r.total_steps==40*steps
        assert r.total_presentations==40*n and r.last_batch_n==n%128 and r.lr1_20==.001 and r.lr21_40==second
        expected=(1-.001*.1)**(20*steps)*(1-second*.1)**(20*steps)
        assert abs(r.nominal_cumulative_decay_factor-expected)<1e-12
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv').set_index('schedule');assessment=read(OUT/'assessment.json')
    assert abs(assessment['mse_skill_vs_constant40']-(1-metrics.loc['decay40','mse']/metrics.loc['constant40','mse']))<1e-12
    assert abs(assessment['mse_skill_vs_archived20']-(1-metrics.loc['decay40','mse']/metrics.loc['archived20','mse']))<1e-12
    primary=read(OUT/'primary_comparisons.json')
    significant=sum(r['candidate']=='decay40' and r['mse']['difference']<0 and r['mse']['holm_adjusted_p']<.05 for r in primary)
    assert significant==assessment['candidate_significant_improvements']
    diagnostic=read(OUT/'moment_diagnostic.json')
    assert diagnostic['status']=='PASS' and not diagnostic['additional_fitting'] and not diagnostic['new_predictions']
    moments=pd.read_csv(OUT/'mse_moment_decomposition.csv').set_index('schedule')
    forecasts=pd.read_csv(OUT/'ensemble_predictions.csv')
    for schedule,g in forecasts.groupby('schedule'):
        p=g.predicted_return.to_numpy(float);y=g.actual.to_numpy(float);r=moments.loc[schedule]
        expected={'squared_bias':(p.mean()-y.mean())**2,'prediction_variance':p.var(),'target_variance':y.var(),
            'negative_twice_covariance':-2*np.mean((p-p.mean())*(y-y.mean()))}
        for key,value in expected.items():assert abs(r[key]-value)<1e-14
        assert abs(sum(expected.values())-metrics.loc[schedule,'mse'])<1e-14
    for r in pd.read_csv(OUT/'mse_change_decomposition.csv').itertuples():
        for key in ['squared_bias','prediction_variance','negative_twice_covariance','target_variance']:
            assert abs(getattr(r,key+'_change')-(moments.loc[r.candidate,key]-moments.loc['archived20',key]))<1e-14
        assert abs(r.mse_change-(metrics.loc[r.candidate,'mse']-metrics.loc['archived20','mse']))<1e-14
    path=OUT/'第十二轮测试报告.md';report=path.read_text(encoding='utf-8')
    assert report.startswith('# 第十二轮方法测试：') and '\ufffd' not in report
    for link in re.findall(r'\]\(([^)]+)\)',report):
        if not link.startswith(('http://','https://')):assert (OUT/link).resolve().is_file(),link
    for r in metrics.itertuples():
        assert f'{r.accuracy*100:.2f}%' in report and f'{r.rmse*1e4:.2f}' in report
    losses=pd.read_csv(OUT/'training_loss_summary.csv')
    for _,g in losses.groupby(['schedule','mode']):
        for key in ['return_mse','auxiliary_mse','joint_mse']:assert f'{g[key].mean():.6f}' in report
    for mapping in ['old_evidence','source_sha256','input_sha256']:
        for name,digest in preparation[mapping].items():assert sha(ROOT.parent/name)==digest,name
    for name,digest in preparation['local_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/'protocol.json')==preparation['protocol_sha256']
    # The preflight revision must leave all research choices except evidence/repair notes unchanged.
    old_protocol=read(ROOT.parent/'research_v12_preflight/protocol.json');new_protocol=read(ROOT/'protocol.json')
    for key,value in old_protocol.items():
        if key!='verification':assert value==new_protocol[key],key
    if freeze:
        assert not (OUT/'delivery_manifest.json').exists(),'Preserve delivered evidence'
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
               if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json'}
        result=dict(experiment=12,report=str(path.relative_to(ROOT)),shared_reconstructions=9,continuations=18,
            complete40_epoch_paths=18,new_research_checkpoints=27,replayed_states=36,validation_weeks=141,
            preserved_preflight_files=25,files=files)
        (OUT/'delivery_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    delivered=read(OUT/'delivery_manifest.json')
    for name,digest in delivered['files'].items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(status='PASS',hashed_files=len(delivered['files']),report_characters=len(report),
        previous_files_preserved=len(preparation['old_evidence']),exact_reconstructions=9,complete40_epoch_paths=18,
        replayed_states=36,descriptive_screen_pass=assessment['descriptive_screen_pass']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');main(parser.parse_args().freeze)

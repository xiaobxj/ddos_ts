"""Verify report arithmetic, links and final evidence hashes before delivery."""
from pathlib import Path
import argparse,hashlib,json,re
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main(freeze):
    v=read(OUT/'verification.json')
    assert v['status']=='PASS' and v['model_states_replayed']==36 and v['dropout_loss_draws_replayed']==288
    assert v['gradient_passes_replayed']==90 and v['epoch_orders_and_boundaries_recomputed']==360
    assert v['new_validation_predictions']==0
    assert read(OUT/'supplementary_verification.json')['status']=='PASS'
    a=read(OUT/'assessment.json')
    assert a['training_only'] and a['new_validation_metrics']==0 and a['new_fits']==18
    p=OUT/'第十一轮测试报告.md'
    report=p.read_text(encoding='utf-8')
    assert report.startswith('# 第十一轮方法测试：') and '\ufffd' not in report
    for link in re.findall(r'\]\(([^)]+)\)',report):
        if not link.startswith(('http://','https://')):
            assert (OUT/link).resolve().is_file(),link
    losses=pd.read_csv(OUT/'training_loss_summary.csv')
    assert len(losses)==72
    for _,g in losses.groupby(['state','mode']):
        assert len(g)==9
        for key in ['return_huber','auxiliary_mse','common_joint','return_mse']:
            assert f'{g[key].mean():.6f}' in report,(key,g[key].mean())
    gradients=pd.read_csv(OUT/'gradient_diagnostics.csv')
    for r in pd.read_csv(OUT/'gradient_summary.csv').to_dict('records'):
        g=gradients[gradients.state.eq(r['state']) & gradients['mode'].eq(r['mode']) &
                    gradients.parameter_group.eq(r['parameter_group'])]
        assert len(g)==r['states']==9
        assert r['negative_cosine_count']==int(g.cosine.lt(-1e-8).sum())
        assert r['full_joint_step_opposes_return_count']==int(g.auxiliary_projection_fraction.lt(-1).sum())
        for key in ['cosine','weighted_auxiliary_norm_ratio','auxiliary_projection_fraction']:
            assert abs(r[key+'_mean']-g[key].mean())<1e-11
        for key in ['cosine','auxiliary_projection_fraction']:
            assert abs(r[key+'_min']-g[key].min())<1e-11 and abs(r[key+'_max']-g[key].max())<1e-11
    prep=read(OUT/'preparation_manifest.json')
    for mapping in ['source_sha256','input_sha256','old_evidence']:
        for name,digest in prep[mapping].items():
            assert sha(ROOT.parent/name)==digest,name
    for name,digest in prep['local_sha256'].items():
        assert sha(ROOT/name)==digest,name
    assert sha(ROOT/'protocol.json')==prep['protocol_sha256']
    if freeze:
        assert not (OUT/'delivery_manifest.json').exists(),'Do not overwrite a delivered round'
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
               if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json'}
        result=dict(experiment=11,report=str(p.relative_to(ROOT)),new_fits=18,archived_states=18,
                    training_only=True,new_validation_predictions=0,new_validation_metrics=0,files=files)
        (OUT/'delivery_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    delivery=read(OUT/'delivery_manifest.json')
    for name,digest in delivery['files'].items():
        assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(status='PASS',hashed_files=len(delivery['files']),report_characters=len(report),
                          previous_files_preserved=len(prep['old_evidence']),new_fits=18,states_replayed=36,
                          new_validation_metrics=0),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    main(parser.parse_args().freeze)

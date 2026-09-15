"""Final file inventory plus independent report and diagnostic checks."""
from pathlib import Path
import json,hashlib,re,argparse
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(freeze):
    verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    assert verification['status']=='PASS' and verification['total_checkpoints_replayed']==18
    assert verification['epoch_order_hashes_recomputed']==180 and verification['boundary_hashes_recomputed']==180
    assert verification['primary_comparisons_recomputed']==3 and verification['training_intercepts_recomputed']==3
    path=OUT/'第十轮测试报告.md';report=path.read_text(encoding='utf-8')
    assert report.startswith('# 第十轮方法测试：') and '\ufffd' not in report
    for link in re.findall(r'\]\(([^)]+)\)',report):
        if not link.startswith(('http://','https://')):assert (OUT/link).resolve().exists(),link
    ensemble=pd.read_csv(OUT/'ensemble_predictions.csv');metrics=pd.read_csv(OUT/'ensemble_metrics.csv')
    assert len(ensemble)==282 and len(metrics)==2
    for r in metrics.itertuples():
        g=ensemble[ensemble.rule==r.rule];assert len(g)==141
        mse=float(np.mean((g.predicted_return-g.actual)**2));accuracy=float(((g.predicted_return>0)==(g.actual>0)).mean())
        assert abs(mse-r.mse)<1e-12 and abs(accuracy-r.accuracy)<1e-12
        assert f'{accuracy*100:.2f}%' in report and f'{mse**.5*1e4:.2f}' in report
    objective=pd.read_csv(OUT/'matched_training_objectives.csv')
    result=json.loads((OUT/'matched_training_objective_summary.json').read_text(encoding='utf-8'))
    assert len(objective)==9 and result['comparisons']==9
    old=objective.scaled_huber_loss+.1*objective.final_model_train_auxiliary_mse_mse
    new=objective.final_model_train_scaled_huber+.1*objective.final_model_train_auxiliary_mse_huber
    np.testing.assert_allclose(old,objective.mse_model_under_huber_joint,rtol=0,atol=1e-12)
    np.testing.assert_allclose(new,objective.huber_model_under_huber_joint,rtol=0,atol=1e-12)
    assert int((new<old).sum())==result['huber_models_with_lower_own_training_objective']
    assert abs(old.mean()-result['mean_mse_model_huber_joint'])<1e-12 and abs(new.mean()-result['mean_huber_model_huber_joint'])<1e-12
    previous=json.loads((OUT/'preparation_manifest.json').read_text(encoding='utf-8'))['old_evidence']
    for name,digest in previous.items():assert sha(ROOT.parent/name)==digest,name
    if freeze:
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
               if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json'}
        manifest=dict(experiment=10,report=str(path.relative_to(ROOT)),new_fits=9,reused_models=9,training_huber_intercepts=3,validation_weeks=141,files=files)
        (OUT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    manifest=json.loads((OUT/'delivery_manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():assert sha(ROOT/name)==digest,name
    assessment=json.loads((OUT/'assessment.json').read_text(encoding='utf-8'))
    print(json.dumps(dict(status='PASS',hashed_files=len(manifest['files']),report_characters=len(report),previous_files_preserved=len(previous),
        new_fits=9,checkpoints_replayed=18,training_intercepts=3,descriptive_screen_pass=assessment['descriptive_screen_pass'],
        flags=assessment['flags'],huber_models_with_lower_own_training_objective=result['huber_models_with_lower_own_training_objective']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');main(parser.parse_args().freeze)

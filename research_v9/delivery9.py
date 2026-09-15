"""Final file inventory and independent report/metric/link checks."""
from pathlib import Path
import json,hashlib,re,argparse
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(freeze):
    audit=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    assert audit['status']=='PASS' and audit['total_checkpoints_replayed']==18
    assert audit['epoch_order_hashes_recomputed']==180 and audit['boundary_hashes_recomputed']==180
    report_path=OUT/'第九轮测试报告.md';report=report_path.read_text(encoding='utf-8')
    assert report.startswith('# 第九轮方法测试：') and '\ufffd' not in report
    for target in re.findall(r'\]\(([^)]+)\)',report):
        if not target.startswith(('http://','https://')):assert (OUT/target).resolve().exists(),target
    predictions=pd.read_csv(OUT/'ensemble_predictions.csv');metrics=pd.read_csv(OUT/'ensemble_metrics.csv')
    assert len(predictions)==282 and len(metrics)==2
    for r in metrics.itertuples():
        g=predictions[predictions.rule==r.rule];assert len(g)==141
        mse=float(np.mean((g.predicted_return-g.actual)**2));accuracy=float(((g.predicted_return>0)==(g.actual>0)).mean())
        assert abs(mse-r.mse)<1e-12 and abs(accuracy-r.accuracy)<1e-12
        assert f'{accuracy*100:.2f}%' in report and f'{mse**.5*1e4:.2f}' in report
    previous=json.loads((OUT/'preparation_manifest.json').read_text(encoding='utf-8'))['old_evidence']
    for name,expected in previous.items():assert digest(ROOT.parent/name)==expected,name
    if freeze:
        files={str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.rglob('*'))
               if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json'}
        data=dict(experiment=9,report=str(report_path.relative_to(ROOT)),new_fits=9,reused_models=9,validation_weeks=141,files=files)
        (OUT/'delivery_manifest.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    delivery=json.loads((OUT/'delivery_manifest.json').read_text(encoding='utf-8'))
    for name,expected in delivery['files'].items():assert digest(ROOT/name)==expected,name
    assessment=json.loads((OUT/'assessment.json').read_text(encoding='utf-8'))
    print(json.dumps(dict(status='PASS',hashed_files=len(delivery['files']),report_characters=len(report),previous_files_preserved=len(previous),
        new_fits=9,checkpoints_replayed=18,descriptive_screen_pass=assessment['descriptive_screen_pass'],flags=assessment['flags']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');main(parser.parse_args().freeze)

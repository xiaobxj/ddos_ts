"""Read-only delivery checksum, link and headline evidence check."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent;out=root/'results'
manifest=json.loads((out/'delivery_manifest.json').read_text(encoding='utf-8'))
for name,digest in manifest['files'].items():
    with (root/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,name
report=(root/manifest['report']).read_text(encoding='utf-8')
assert report.startswith('# 第五轮方法测试：') and '\ufffd' not in report and 'NotImplementedError' not in report
for target in re.findall(r'\]\(([^)]+)\)',report):
    if not target.startswith(('https://','http://')):assert (out/target).resolve().exists(),target
audit=json.loads((out/'verification.json').read_text(encoding='utf-8'));assert audit['status']=='PASS'
capacity=json.loads((out/'capacity_summary.json').read_text(encoding='utf-8'))
selection=json.loads((out/'selection.json').read_text(encoding='utf-8'))
metrics=pd.read_csv(out/'validation_metrics.csv')
ensemble=pd.read_csv(out/'validation_ensemble_predictions.csv')
for r in metrics.itertuples():
    g=ensemble[(ensemble.variant==r.variant)&(ensemble.epoch==r.epoch)]
    assert abs(np.sqrt(np.mean((g.predicted_return-g.actual)**2))-r.rmse)<1e-12
    assert abs(((g.predicted_return>0)==(g.actual>0)).mean()-r.accuracy)<1e-12
reference=json.loads((out/'frozen_round4_reference.json').read_text(encoding='utf-8'))
prior=pd.read_csv(root.parent/'research_v4/results/validation_seed_predictions.csv')
prior=prior[prior.epoch==reference['epochs']].groupby('date').agg(p=('predicted_return','mean'),y=('actual','first'))
assert abs(np.sqrt(np.mean((prior.p-prior.y)**2))-reference['rmse'])<1e-12
print(json.dumps(dict(status='PASS',hashed_files=len(manifest['files']),report_characters=len(report),
                      capacity_fits=len(capacity),capacity_passes=sum(r['capacity_pass'] for r in capacity),
                      validation_candidates=len(metrics),validation_weeks=selection['validation_weeks'],
                      capacity_checkpoints_replayed=audit['real_and_permuted_capacity_checkpoints_replayed'],
                      validation_checkpoints_replayed=audit['validation_checkpoints_replayed'],
                      selected_variant=selection['selected']['variant'],selected_epochs=selection['selected']['epoch'],
                      selected_validation_rmse=selection['selected']['rmse']),indent=2))

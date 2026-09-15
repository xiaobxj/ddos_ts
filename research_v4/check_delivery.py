"""Read-only final delivery integrity and evidence checks."""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd

root=Path(__file__).resolve().parent;out=root/'results'
manifest=json.loads((out/'delivery_manifest.json').read_text(encoding='utf-8'))
for name,digest in manifest['files'].items():
    with (root/name).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,name
report=(root/manifest['report']).read_text(encoding='utf-8')
assert report.startswith('# 第四轮方法测试：')
assert '\ufffd' not in report and 'NotImplementedError' not in report
for target in re.findall(r'\]\(([^)]+)\)',report):
    if not target.startswith(('https://','http://')):
        assert (out/target).resolve().exists(),target
verification=json.loads((out/'verification.json').read_text(encoding='utf-8'))
assert verification['status']=='PASS'
config=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
arms=[a['name'] for a in config['arms']]
metrics=pd.read_csv(out/'metrics.csv').query('window=="common"').set_index('method')
assert (metrics.loc[arms,'return_rmse']>metrics.loc['training_mean','return_rmse']).all()
pred=pd.read_csv(out/'predictions.csv')
assert pred[pred.method=='kronos_permuted'].position.eq(1).all()
pairs=json.loads((out/'paired_comparisons.json').read_text(encoding='utf-8'))
common=[p for p in pairs if p['window']=='common']
assert len(common)==7 and min(p['mse']['holm_adjusted_p'] for p in common)>.05
influence=json.loads((out/'influence_diagnostics.json').read_text(encoding='utf-8'))
contrast=next(r for r in influence if r['window']=='common' and r['method']=='kronos_adaptive'
              and r['reference']=='random_tokenizer_adaptive')
print(json.dumps(dict(status='PASS',files_hashed=len(manifest['files']),
                      report_characters=len(report),development_checkpoints_replayed=verification['development_checkpoints_replayed'],
                      all_seven_model_rmse_above_training_mean=True,permuted_tokenizer_positions_identical_to_buyhold=True,
                      minimum_primary_holm_p=min(p['mse']['holm_adjusted_p'] for p in common),
                      pretrained_vs_random_top_week_share=contrast['largest_positive_log_contribution']/contrast['total_log_wealth_gap']),indent=2))

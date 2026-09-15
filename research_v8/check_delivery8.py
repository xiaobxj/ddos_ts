"""Read-only final artifact, linkage and headline-number audit."""
from pathlib import Path
import json
import hashlib
import re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'


def main():
    delivery=json.loads((OUT/'delivery_manifest.json').read_text(encoding='utf-8'))
    for name,digest in delivery['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    report=(ROOT/delivery['report']).read_text(encoding='utf-8')
    assert report.startswith('# 第八轮方法测试：') and '\ufffd' not in report
    for link in re.findall(r'\]\(([^)]+)\)',report):
        if not link.startswith(('http://','https://')):assert (OUT/link).resolve().exists(),link
    audit=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    assert audit['status']=='PASS' and audit['total_neural_checkpoints_replayed']==54 and audit['linear_models_replayed']==18
    assert audit['training_order_hashes_recomputed']==900 and audit['primary_comparisons_recomputed']==10
    c=pd.read_csv(OUT/'policy_comparisons.csv');all_predictions=pd.read_csv(OUT/'ensemble_predictions.csv')
    assert len(c)==6 and len(all_predictions)==1692
    for r in c.itertuples():
        g=all_predictions[(all_predictions.model=='neural')&(all_predictions.policy==r.policy)]
        assert len(g)==141
        mse=float(np.mean((g.predicted_return-g.actual)**2))
        assert abs(mse**.5-r.rmse)<1e-12
        assert abs(((g.predicted_return>0)==(g.actual>0)).mean()-r.accuracy)<1e-12
        assert f'{r.accuracy*100:.2f}%' in report and f'{r.rmse*1e4:.2f}' in report
        common_mse=float(np.mean((g.full_training_mean-g.actual)**2))
        assert abs(1-mse/common_mse-r.mse_skill_vs_common_mean)<1e-12
    archive=json.loads((ROOT/'preflight_archive.json').read_text(encoding='utf-8'))
    for name,digest in archive['files'].items():assert hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()==digest,name
    old=json.loads((OUT/'preparation_manifest.json').read_text(encoding='utf-8'))['old_evidence']
    for name,digest in old.items():assert hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()==digest,name
    status=json.loads((OUT/'assessment.json').read_text(encoding='utf-8'))
    print(json.dumps(dict(status='PASS',hashed_delivery_files=len(delivery['files']),report_characters=len(report),
        formal_new_neural_fits=45,withdrawn_preflight_neural_fits=4,neural_checkpoints_replayed=54,linear_models_replayed=18,
        preserved_previous_files=1794,preserved_preflight_files=len(archive['files']),
        policy_screens=status['policy_screens'],disjoint_family_screen=status['disjoint_family_screen']),indent=2))


if __name__=='__main__':main()

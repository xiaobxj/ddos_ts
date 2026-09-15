"""Run frozen R39 checks with explicit None/NaN equivalence for empty metric columns.

No research source, parameter, prediction, numeric tolerance or gate is changed.
The first failed verification log remains intact. Only three all-missing floating
metric columns are normalized on copied frames before the original assertion.
"""
from pathlib import Path
import sys,json,hashlib,time
root=Path('D:/ddos_v3/research_v39');out=root/'results'
sys.path.insert(0,str(root));import verify39 as verification
pd=verification.pd;np=verification.np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
failure=out/'logs/verify.log';assert 'None != nan' in failure.read_text(encoding='utf-8')
assert not (out/'verification.json').exists();assert not (out/'verification_compatibility.json').exists()
before={str(f.relative_to(root)):sha(f) for phase in ['fitting','validation','scoring','evaluation'] for f in [root/n for n in read(out/f'{phase}_manifest.json')['artifacts']]}
original=pd.testing.assert_frame_equal;normalizations=[]
def compatible(left,right,*args,**kwargs):
    a=left.copy();b=right.copy()
    for name in ['brier_difference','baseline_brier','candidate_brier']:
        if name in a and name in b and a[name].isna().all() and b[name].isna().all():
            a[name]=a[name].astype(float);b[name]=b[name].astype(float);normalizations.append(dict(column=name,rows=len(a)))
    return original(a,b,*args,**kwargs)
# Verify that only the representation of undefined values is accepted.
compatible(pd.DataFrame({'brier_difference':[None]}),pd.DataFrame({'brier_difference':[np.nan]}),check_dtype=False)
for left,right in [(None,0.),(.1,.2)]:
    try:compatible(pd.DataFrame({'brier_difference':[left]}),pd.DataFrame({'brier_difference':[right]}),check_dtype=False)
    except AssertionError:pass
    else:raise AssertionError('Compatibility must not mask missing-vs-number or unequal finite metrics')
normalizations.clear();pd.testing.assert_frame_equal=compatible
started=verification.now();verification.main();pd.testing.assert_frame_equal=original
assert all(sha(root/n)==d for n,d in before.items())
copy=out/'verification_null_compat.py.txt';copy.write_bytes(Path(__file__).read_bytes())
record=dict(status='PASS',started_utc=started,finished_utc=verification.now(),reason='Pandas3 empty-column representation: None versus NaN in zero-validation warmup cells. Original numeric tolerances and every substantive check retained.',normalizations=normalizations,original_failure_log_sha256=sha(failure),original_verifier_sha256=sha(root/'verify39.py'),helper_sha256=sha(copy),research_artifacts_unchanged=before,verification_sha256=sha(out/'verification.json'))
(out/'verification_compatibility.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8');print('Verification compatibility audit PASS; all fits, gates and forecasts unchanged.',flush=True)

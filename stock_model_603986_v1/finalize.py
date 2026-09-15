"""Seal verified stock delivery and independently check preserved index evidence."""
from stock import *

def main():
    check_freeze();assert read(OUT/'verification.json')['status']=='PASS'
    assert (ROOT/'建模结果.md').is_file() and (ROOT/'603986_滚动检验.png').is_file()
    import torch
    save(OUT/'runtime_environment.json',dict(recorded_at_delivery_utc=now(),executable=sys.executable,python=sys.version,
        numpy=np.__version__,pandas=pd.__version__,torch=str(torch.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),training_settings_source='Frozen research_v6/common6.py initialize and stock protocol.json'))
    prior=PROJECT/'research_v51';old=read(prior/'results/freeze.json')['old_evidence'].copy()
    delivery=read(prior/'results/delivery_manifest.json')
    old.update({relative(prior/n):h for n,h in delivery['files'].items()})
    old[relative(prior/'results/delivery_manifest.json')]=sha(prior/'results/delivery_manifest.json')
    assert len(old)==6296
    for name,digest in old.items():assert sha(PROJECT/name)==digest,name
    for name,digest in delivery['real_runtime_files'].items():assert sha(PROJECT/name)==digest,name
    events=sorted((PROJECT/'prospective_r49/events').glob('*.json'));assert len(events)==2
    save(OUT/'index_preservation.json',dict(status='PASS',checked_utc=now(),index_evidence_files=6296,
        real_index_journal_events=2,index_journal_predictions_added=0,index_journal_labels_added=0))
    paths=sorted(p for p in ROOT.rglob('*') if p.is_file())
    save(OUT/'delivery_manifest.json',dict(status='PASS',completed_utc=now(),freeze_sha256=sha(OUT/'freeze.json'),
        final_verification_sha256=sha(OUT/'verification.json'),files={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    print('Stock delivery sealed; all 6296 pre-existing index evidence files preserved.',flush=True)

if __name__=='__main__':main()

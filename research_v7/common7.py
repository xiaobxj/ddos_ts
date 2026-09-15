from pathlib import Path
import sys
import json
import hashlib
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
V5 = PROJECT / 'research_v5'
V6 = PROJECT / 'research_v6'
OUT = ROOT / 'results'
CACHE = ROOT / 'cache'
sys.path.insert(0, str(V6))
import models6 as legacy
import numpy as np
import pandas as pd
torch = legacy.torch


def cfg():
    return json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def old_evidence():
    prior = json.loads((V6 / 'results/preparation_manifest.json').read_text(encoding='utf-8'))['old_evidence']
    delivery = json.loads((V6 / 'results/delivery_manifest.json').read_text(encoding='utf-8'))
    result = dict(prior)
    for name, digest in delivery['files'].items():
        result[str((V6 / name).relative_to(PROJECT))] = digest
    result[str((V6 / 'results/delivery_manifest.json').relative_to(PROJECT))] = sha(V6 / 'results/delivery_manifest.json')
    for name, digest in result.items():
        assert sha(PROJECT / name) == digest, name
    return result


def source_hashes():
    files = [ROOT / n for n in ['common7.py', 'prepare7.py', 'train7.py']]
    files += [V6 / n for n in ['common6.py', 'models6.py', 'train6.py']]
    old = PROJECT / 'research_v4'
    files += [old / 'architecture.py']
    files += list((old / 'vendor/Crossformer/cross_models').rglob('*.py'))
    return {str(p.relative_to(PROJECT)): sha(p) for p in files}


def input_hashes():
    files = [V5 / 'cache/packed_raw.npz', V5 / 'cache/targets.npz', V6 / 'results/observation_table.csv',
             V6 / 'protocol.json', V6 / 'results/main_seed_predictions.csv', V6 / 'results/main_checkpoint_manifest.json',
             V6 / 'results/blocks_seed_predictions.csv', V6 / 'results/blocks_checkpoint_manifest.json']
    files += list((V6 / 'cache').glob('masks_*.npz'))
    return {str(p.relative_to(PROJECT)): sha(p) for p in files}


def manifest(phase):
    return dict(phase=phase, started_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                protocol_sha256=sha(ROOT / 'protocol.json'), source_sha256=source_hashes(),
                input_sha256=input_hashes(), old_evidence=old_evidence(), executable=sys.executable,
                python=sys.version, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                device=torch.cuda.get_device_name())


def role(arm, epoch):
    return next(r['name'] for r in cfg()['roles'] if r['arm'] == arm and r['epoch'] == epoch)


def fit_plan():
    return [dict(arm=arm, history=history, cutoff=fold['cutoff'], end=fold['end'], seed=seed)
            for arm in cfg()['training']['arms'] for fold in cfg()['folds']
            for history in cfg()['histories'][1:] for seed in cfg()['training']['seeds']]


def order_hash(rows):
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()

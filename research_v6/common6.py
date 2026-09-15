from pathlib import Path
import os
import sys
import json
import hashlib
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
V3 = PROJECT / 'research_v3'
V4 = PROJECT / 'research_v4'
V5 = PROJECT / 'research_v5'
OUT = ROOT / 'results'
CACHE = ROOT / 'cache'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(V4))
import numpy as np
import pandas as pd


def cfg():
    return json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def old_evidence():
    prior = json.loads((V5 / 'results/validation_manifest.json').read_text(encoding='utf-8'))['old_evidence']
    delivery = json.loads((V5 / 'results/delivery_manifest.json').read_text(encoding='utf-8'))
    result = dict(prior)
    for name, digest in delivery['files'].items():
        result[str((V5 / name).relative_to(PROJECT))] = digest
    path = V5 / 'results/delivery_manifest.json'
    result[str(path.relative_to(PROJECT))] = sha(path)
    for name, digest in result.items():
        assert sha(PROJECT / name) == digest, name
    return result


def initialize():
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    return torch


def input_hashes():
    files = [V5 / 'cache/packed_raw.npz', V5 / 'cache/targets.npz',
             V5 / 'results/observation_table.csv', V5 / 'protocol.json']
    return {str(p.relative_to(PROJECT)): sha(p) for p in files}


def source_hashes():
    files = [ROOT / n for n in ['common6.py', 'models6.py', 'prepare6.py', 'train6.py']]
    files += [V4 / 'architecture.py']
    files += list((V4 / 'vendor/Crossformer/cross_models').glob('*.py'))
    files += list((V4 / 'vendor/Crossformer/cross_models').glob('*/*.py'))
    return {str(p.relative_to(PROJECT)): sha(p) for p in files}


def start_manifest(phase):
    import torch
    return dict(phase=phase, started_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                protocol_sha256=sha(ROOT / 'protocol.json'), source_sha256=source_hashes(),
                input_sha256=input_hashes(), old_evidence=old_evidence(), python=sys.version,
                executable=sys.executable, torch=torch.__version__, numpy=np.__version__,
                pandas=pd.__version__, device=torch.cuda.get_device_name())


def fold_indices(obs, fold):
    training = np.flatnonzero((obs.joint_completed <= fold['cutoff']).to_numpy())
    testing = np.flatnonzero(((obs.date > fold['cutoff']) & (obs.date <= fold['end']) &
                             (obs.weekday == 4) & (obs.joint_completed <= '2020-12-31')).to_numpy())
    blocks = np.array_split(training, 5)
    histories = {'full': training}
    for name, block in [('drop_early', 0), ('drop_middle', 2), ('drop_recent', 4)]:
        histories[name] = np.setdiff1d(training, blocks[block], assume_unique=True)
    return histories, testing


def epoch_order(count, presentations, rng):
    """Same full-history routine as round 5; retained-only permutations otherwise."""
    assert presentations >= count > 0
    if count == presentations:
        return rng.permutation(count)
    chunks = []
    remaining = presentations
    while remaining:
        part = rng.permutation(count)[:remaining]
        chunks.append(part)
        remaining -= len(part)
    return np.concatenate(chunks)


def stats(pred, y):
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    mse = float(np.mean((pred - y) ** 2))
    return dict(mse=mse, rmse=float(np.sqrt(mse)),
                accuracy=float(((pred > 0) == (y > 0)).mean()),
                correlation=float(np.corrcoef(pred, y)[0, 1]) if pred.std() > 1e-12 else None,
                forecast_std=float(pred.std()))

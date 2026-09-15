from pathlib import Path
import sys
import json
import hashlib
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
V5 = PROJECT / 'research_v5'
V6 = PROJECT / 'research_v6'
V7 = PROJECT / 'research_v7'
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
    result = dict(json.loads((V7 / 'results/preparation_manifest.json').read_text(encoding='utf-8'))['old_evidence'])
    prior = json.loads((V7 / 'results/delivery_manifest.json').read_text(encoding='utf-8'))
    for name, digest in prior['files'].items():
        result[str((V7 / name).relative_to(PROJECT))] = digest
    result[str((V7 / 'results/delivery_manifest.json').relative_to(PROJECT))] = sha(V7 / 'results/delivery_manifest.json')
    result.update(json.loads((ROOT / 'preflight_archive.json').read_text(encoding='utf-8'))['files'])
    for name, digest in result.items():
        assert sha(PROJECT / name) == digest, name
    return result


def source_hashes():
    files = [ROOT / n for n in ['common8.py', 'prepare8.py', 'train8.py', 'audit8.py', 'linear8.py', 'evaluate8.py', 'contract8.py']]
    files += [V6 / n for n in ['common6.py', 'models6.py', 'evaluate6.py']]
    old = PROJECT / 'research_v4'
    files += [old / 'architecture.py'] + list((old / 'vendor/Crossformer/cross_models').rglob('*.py'))
    return {str(p.relative_to(PROJECT)): sha(p) for p in files}


def input_hashes():
    files = [ROOT / 'preflight_archive.json', V5 / 'cache/packed_raw.npz', V5 / 'cache/targets.npz', V6 / 'protocol.json',
             V7 / 'results/observation_table.csv', V7 / 'results/reused_full_predictions.csv',
             V7 / 'results/reused_checkpoint_manifest.json']
    files += list((V7 / 'cache').glob('masks_*.npz'))
    return {str(p.relative_to(PROJECT)): sha(p) for p in files}


def manifest(phase):
    return dict(phase=phase, started_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                protocol_sha256=sha(ROOT / 'protocol.json'), source_sha256=source_hashes(),
                input_sha256=input_hashes(), old_evidence=old_evidence(), executable=sys.executable,
                python=sys.version, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                device=torch.cuda.get_device_name())


def check_frozen(require_contract=True):
    prep = json.loads((OUT / 'preparation_manifest.json').read_text(encoding='utf-8'))
    assert prep['protocol_sha256'] == sha(ROOT / 'protocol.json')
    assert prep['source_sha256'] == source_hashes() and prep['input_sha256'] == input_hashes()
    for name, digest in prep['local_sha256'].items():
        assert sha(ROOT / name) == digest, name
    if require_contract:
        contract = json.loads((OUT / 'contract_verification.json').read_text(encoding='utf-8'))
        assert contract['status'] == 'PASS' and contract['source_sha256'] == source_hashes()
    return prep


def fit_plan():
    jobs = [dict(policy=policy, cutoff=f['cutoff'], end=f['end'], seed=seed)
            for f in cfg()['folds'] for policy in cfg()['policies'][1:] for seed in cfg()['training']['seeds']]
    return [dict(**job, job_index=i, worker=i % 2) for i, job in enumerate(jobs)]


def order_hash(rows):
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()


def define_policy(obs, full, name):
    tr = np.asarray(full, dtype=np.int64)
    weights = np.ones(len(tr), dtype=float)
    if name.startswith('disjoint'):
        phase = int(name[-1]); chosen = []; last = -1
        candidates = tr[((obs.anchor.iloc[tr].to_numpy()-int(obs.anchor.iloc[full[0]])) % 6)==phase]
        for i in candidates:
            row = obs.iloc[i]
            if int(row.entry) > last:
                chosen.append(i)
                last = max(int(row.exit), int(row.anchor)+5)
        tr = np.asarray(chosen, dtype=np.int64)
        weights = np.ones(len(tr), dtype=float)
    elif name == 'recent3y':
        # The cutoff is inferred only from the supplied full fold's mature year.
        year = int(obs.joint_completed.iloc[full].str[:4].max())
        tr = tr[obs.date.iloc[tr].ge(f'{year-2}-01-01').to_numpy()]
        weights = np.ones(len(tr), dtype=float)
    elif name == 'quarter_equal':
        quarters = pd.to_datetime(obs.date.iloc[tr]).dt.to_period('Q').astype(str)
        counts = quarters.value_counts()
        weights = len(tr) / (len(counts)*quarters.map(counts).to_numpy(dtype=float))
    else:
        assert name == 'full'
    return tr, weights


def load_policy(cutoff, policy):
    with np.load(CACHE / f'policies_{cutoff}.npz') as data:
        return data[policy].copy(), data[policy+'_weights'].copy(), data['testing'].copy(), len(data['full'])


def target_arrays(indices, weights):
    labels = {}; scales = {}
    with np.load(V5 / 'cache/targets.npz') as data:
        for key in ['returns', 'auxiliary']:
            y = data[key][indices].astype(float)
            if np.array_equal(weights, np.ones(len(weights))):
                mean = y.mean(axis=0); sd = y.std(axis=0)
            else:
                mean = np.average(y, axis=0, weights=weights)
                sd = np.sqrt(np.average((y-mean)**2, axis=0, weights=weights))
            sd = np.maximum(sd, 1e-6)
            labels[key] = (y-mean)/sd
            scales[key+'_mean'] = np.asarray(mean).tolist()
            scales[key+'_sd'] = np.asarray(sd).tolist()
    return labels, scales


def targets(indices, weights):
    labels, scales = target_arrays(indices, weights)
    return {k: torch.tensor(v, dtype=torch.float32, device='cuda') for k, v in labels.items()}, scales


def joint_loss(output, auxiliary, labels, batch, weights):
    # Keep legacy reduction order for exactly equal weights, allowing tensor-exact checks.
    if weights is None:
        return ((output-labels['returns'][batch])**2).mean() + .1*((auxiliary-labels['auxiliary'][batch])**2).mean()
    return (((output-labels['returns'][batch])**2)*weights[batch]).mean() + .1*(
        ((auxiliary-labels['auxiliary'][batch])**2)*weights[batch, None]).mean()

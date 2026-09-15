from pathlib import Path
from contextlib import contextmanager
import sys, json, hashlib, time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
V5 = PROJECT / 'research_v5'
V6 = PROJECT / 'research_v6'
V10 = PROJECT / 'research_v10'
OUT = ROOT / 'results'
CACHE = ROOT / 'cache'
sys.path.insert(0, str(V6))
import models6 as legacy
import numpy as np
import pandas as pd
torch = legacy.torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def cfg():
    return read(ROOT / 'protocol.json')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype='<i8').tobytes()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def old_evidence():
    result = dict(read(V10 / 'results/preparation_manifest.json')['old_evidence'])
    for name, digest in read(V10 / 'results/delivery_manifest.json')['files'].items():
        result[str((V10 / name).relative_to(PROJECT))] = digest
    p = V10 / 'results/delivery_manifest.json'
    result[str(p.relative_to(PROJECT))] = sha(p)
    for name, digest in result.items():
        assert sha(PROJECT / name) == digest, name
    assert len(result) == 2106, len(result)
    return result


def source_hashes():
    files = [ROOT / n for n in ['common11.py', 'prepare11.py', 'contract11.py', 'probe11.py',
                                'train11.py', 'analyze11.py', 'verify11.py']]
    files += [V6 / 'common6.py', V6 / 'models6.py', PROJECT / 'research_v4/architecture.py']
    files += sorted((PROJECT / 'research_v4/vendor/Crossformer/cross_models').rglob('*.py'))
    return {str(p.relative_to(PROJECT)): sha(p) for p in files}


def input_hashes():
    files = [V5 / 'cache/packed_raw.npz', V5 / 'cache/targets.npz', V6 / 'protocol.json',
             V10 / 'results/observation_table.csv', V10 / 'results/mse_checkpoint_manifest.json',
             V10 / 'results/huber_checkpoint_manifest.json', V10 / 'results/matched_training_objectives.csv']
    files += sorted((V10 / 'cache').glob('masks_*.npz'))
    for rule in ['mse', 'huber']:
        files += [PROJECT / r['project_file'] for r in read(V10 / f'results/{rule}_checkpoint_manifest.json')]
    return {str(p.relative_to(PROJECT)): sha(p) for p in files}


def manifest(phase):
    return dict(phase=phase, started_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                protocol_sha256=sha(ROOT / 'protocol.json'), source_sha256=source_hashes(),
                input_sha256=input_hashes(), executable=sys.executable, python=sys.version,
                torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                device=torch.cuda.get_device_name(), training_only=True)


def check_frozen(contract=True):
    prep = read(OUT / 'preparation_manifest.json')
    assert prep['protocol_sha256'] == sha(ROOT / 'protocol.json')
    assert prep['source_sha256'] == source_hashes() and prep['input_sha256'] == input_hashes()
    for name, digest in prep['local_sha256'].items():
        assert sha(ROOT / name) == digest, name
    if contract:
        test = read(OUT / 'contract_verification.json')
        assert test['status'] == 'PASS' and test['source_sha256'] == source_hashes()
    return prep


def load_training(cutoff):
    # Experiment forward/backward passes can only use these physically restricted arrays.
    with np.load(CACHE / f'training_{cutoff}.npz') as data:
        assert set(data.files) == {'patches', 'geometry', 'valid', 'returns', 'auxiliary', 'row_index'}
        values = {k: torch.from_numpy(data[k].copy()).cuda() for k in ['patches', 'geometry', 'valid']}
        labels = {k: torch.from_numpy(data[k].copy()).cuda() for k in ['returns', 'auxiliary']}
        indices = data['row_index'].copy()
    scales = read(OUT / 'training_scales.json')[cutoff]
    return values, labels, indices, scales


def load_model(ref):
    p = PROJECT / ref['project_file']
    assert sha(p) == ref['sha256']
    state = torch.load(p, map_location='cpu', weights_only=True)
    model = legacy.make_model('combined', ref['seed'])
    model.load_state_dict(state['state_dict'])
    assert sum(p.numel() for p in model.parameters()) == 38551
    return model, state


def batches(order, size=128):
    return [order[i:i + size] for i in range(0, len(order), size)]


def robust_loss(errors):
    return torch.where(errors.abs() <= 1., errors.square(), 2. * errors.abs() - 1.)


def components(model, values, labels, b):
    p, a = legacy.predict(model, {k: t[b] for k, t in values.items()}, True)
    return robust_loss(p - labels['returns'][b]), (a - labels['auxiliary'][b]).square().mean(dim=1)


@contextmanager
def audit_mode(model, stochastic=False, seed=None):
    modes = [(m, m.training) for m in model.modules()]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model.train(stochastic)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        try:
            yield
        finally:
            for module, training in modes:
                module.training = training


def loss_pass(model, values, labels, stochastic=False, seed=None):
    outputs, auxiliaries = [], []
    n = len(labels['returns'])
    with audit_mode(model, stochastic, seed), torch.inference_mode():
        for lo in range(0, n, 128):
            v = {k: t[lo:lo + 128] for k, t in values.items()}
            p, a = legacy.predict(model, v, True)
            outputs.append(p.cpu().numpy().copy())
            auxiliaries.append((a - labels['auxiliary'][lo:lo + 128]).square().mean(dim=1).cpu().numpy().copy())
    p = np.concatenate(outputs)
    a = np.concatenate(auxiliaries)
    e = p.astype(float) - labels['returns'].cpu().numpy().astype(float)
    huber = np.where(np.abs(e) <= 1., e * e, 2. * np.abs(e) - 1.)
    stats = dict(return_mse=float(np.mean(e * e)), return_huber=float(huber.mean()),
                 auxiliary_mse=float(a.astype(float).mean()), linear_fraction=float(np.mean(np.abs(e) > 1.)))
    stats['common_joint'] = stats['return_huber'] + .1 * stats['auxiliary_mse']
    return stats, dict(predicted_standardized=p, auxiliary_mse=a, return_huber=huber)


def loss_audit(model, values, labels):
    settings = cfg()['frozen_state_loss_audit']
    stats, outputs = loss_pass(model, values, labels)
    rows = [dict(mode='eval', draw=-1, audit_seed=-1, **stats)]
    for draw in range(settings['dropout_passes']):
        seed = settings['dropout_seed_base'] + draw
        stats, _ = loss_pass(model, values, labels, True, seed)
        rows.append(dict(mode='dropout', draw=draw, audit_seed=seed, **stats))
    return rows, outputs


def gradient_layout(model):
    return [dict(name=k, count=p.numel(), shape=list(p.shape),
                 shared=not k.startswith(('return_head.', 'auxiliary_head.')))
            for k, p in model.named_parameters()]


def flat_gradient(gradients, parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for g, p in zip(gradients, parameters)])


def gradient_pass(model, values, labels, stochastic=False, seed=None, batch_size=128):
    parameters = tuple(model.parameters())
    count = sum(p.numel() for p in parameters)
    total_return = torch.zeros(count, device='cuda', dtype=torch.float64)
    total_auxiliary = torch.zeros_like(total_return)
    n = len(labels['returns'])
    with audit_mode(model, stochastic, seed):
        for lo in range(0, n, batch_size):
            b = slice(lo, lo + batch_size)
            r, a = components(model, values, labels, b)
            gr = torch.autograd.grad(r.sum() / n, parameters, retain_graph=True, allow_unused=True)
            ga = torch.autograd.grad(a.sum() / n, parameters, allow_unused=True)
            total_return += flat_gradient(gr, parameters).detach().double()
            total_auxiliary += flat_gradient(ga, parameters).detach().double()
    return total_return.cpu().numpy(), total_auxiliary.cpu().numpy()


def gradient_stats(gr, ga, layout):
    shared = np.concatenate([np.full(r['count'], r['shared'], dtype=bool) for r in layout])
    rows = []
    for group, mask in [('shared', shared), ('all', np.ones_like(shared))]:
        r, a = gr[mask], ga[mask]
        nr, na, dot = np.linalg.norm(r), np.linalg.norm(a), float(np.dot(r, a))
        rows.append(dict(parameter_group=group, parameter_count=int(mask.sum()),
            return_gradient_norm=float(nr), auxiliary_gradient_norm=float(na), dot_return_auxiliary=dot,
            cosine=float(dot / (nr * na)) if nr * na > 0 else 0.,
            weighted_auxiliary_norm_ratio=float(.1 * na / nr) if nr > 0 else None,
            auxiliary_projection_fraction=float(.1 * dot / (nr * nr)) if nr > 0 else None,
            common_joint_gradient_norm=float(np.linalg.norm(r + .1 * a))))
    return rows


def update_step(model, optimizer, values, labels, positions, auxiliary_weight):
    b = torch.as_tensor(positions, dtype=torch.int64, device='cuda')
    optimizer.zero_grad(set_to_none=True)
    r, a = components(model, values, labels, b)
    loss = r.mean() + auxiliary_weight * a.mean()
    assert torch.isfinite(loss)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(r.detach().mean()), float(a.detach().mean()), float(norm)


def reference_id(ref):
    return f"{ref['state']}_{ref['cutoff']}_{ref['seed']}"


def metadata(ref):
    return {k: ref[k] for k in ['state', 'cutoff', 'seed']}

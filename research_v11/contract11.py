"""Causal subsets, gradients, and stochastic observation isolation on the real model."""
from common11 import *


def main():
    legacy.initialize()
    check_frozen(contract=False)
    checks = []
    obs = pd.read_csv(V10 / 'results/observation_table.csv')
    for cutoff in cfg()['cutoffs']:
        values, labels, tr, scales = load_training(cutoff)
        np.testing.assert_array_equal(tr, np.flatnonzero(obs.joint_completed.le(cutoff).to_numpy()))
        expected, expected_scales = legacy.standardized_targets(tr)
        assert scales == expected_scales
        assert all(torch.equal(labels[k], expected[k]) for k in labels)
        old_values = legacy.batch_tensors(tr)
        assert all(torch.equal(values[k], old_values[k]) for k in values)
        assert len(batches(tr)) == (len(tr) + 127) // 128
        del values, labels, expected, old_values
    checks.append('Exact causal training-only arrays, target scales and original batch budgets')
    e = torch.tensor([-8., -1.3, -1., -.2, 0., .3, 1., 1.4, 8.], dtype=torch.float64, device='cuda', requires_grad=True)
    expected = 2 * torch.nn.functional.huber_loss(e, torch.zeros_like(e), reduction='none', delta=1.)
    torch.testing.assert_close(robust_loss(e), expected, rtol=0, atol=0)
    derivative = torch.autograd.grad(robust_loss(e).sum(), e)[0]
    torch.testing.assert_close(derivative, 2 * e.detach().clamp(-1, 1), rtol=0, atol=0)
    checks.append('Scaled Huber formula and derivative against installed PyTorch')
    ref = next(r for r in read(OUT / 'archived_models.json') if r['state'] == 'archived_huber')
    model, state = load_model(ref)
    values, labels, _, _ = load_training(ref['cutoff'])
    values = {k: v[:129] for k, v in values.items()}
    labels = {k: v[:129] for k, v in labels.items()}
    r, a = gradient_pass(model, values, labels, batch_size=128)
    rr, aa = gradient_pass(model, values, labels, batch_size=129)
    errors = [float(np.linalg.norm(x-y)/max(np.linalg.norm(y), 1e-12)) for x, y in [(r, rr), (a, aa)]]
    assert max(errors) < 3e-5, errors
    with audit_mode(model):
        lr, la = components(model, values, labels, slice(None))
        joint = torch.autograd.grad(lr.mean() + .1 * la.mean(), tuple(model.parameters()), allow_unused=True)
        g = flat_gradient(joint, tuple(model.parameters())).detach().cpu().numpy().astype(float)
    identity_error = float(np.linalg.norm(g-(rr+.1*aa))/np.linalg.norm(g))
    assert identity_error < 3e-5
    checks.append('Real-model row-weighted gradient partition and joint-gradient identities')
    model.train()
    model.return_head.eval()
    modes = [m.training for m in model.modules()]
    before = state_hash(model)
    cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    first, _ = loss_pass(model, values, labels, True, 20261100)
    gradient_pass(model, values, labels, True, 20261100)
    second, _ = loss_pass(model, values, labels, True, 20261100)
    assert first == second and modes == [m.training for m in model.modules()]
    assert before == state_hash(model)
    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(cuda, torch.cuda.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    checks.append('Dropout losses/gradients deterministic and preserve RNG, module modes, weights and .grad')
    hashes = []
    for add_audit in [False, True]:
        candidate, _ = load_model(ref)
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=.0001, weight_decay=.1)
        torch.manual_seed(ref['seed'] + 110000)
        torch.cuda.manual_seed_all(ref['seed'] + 110000)
        order = np.random.default_rng(ref['seed'] + 110000).permutation(129)
        candidate.train()
        for b in batches(order):
            if add_audit:
                loss_pass(candidate, values, labels, True, 20261101)
            update_step(candidate, optimizer, values, labels, b, .1)
        hashes.append(state_hash(candidate))
        del candidate, optimizer
    assert hashes[0] == hashes[1]
    checks.append('Paired stochastic updates identical even when fixed-state audits are inserted')
    check_frozen(contract=False)
    result = dict(status='PASS', checks=checks, checks_passed=len(checks),
                  gradient_partition_relative_errors=errors, joint_gradient_relative_error=identity_error,
                  source_sha256=source_hashes(), protocol_sha256=sha(ROOT/'protocol.json'),
                  completed_utc=pd.Timestamp.now(tz='UTC').isoformat(), training_only=True)
    save(OUT / 'contract_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

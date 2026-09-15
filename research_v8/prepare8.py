"""Freeze date-only training policies and already archived full-history references."""
from common8 import *


def main():
    legacy.initialize(); OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    assert not (OUT / 'preparation_manifest.json').exists(), 'Preparation is already frozen'
    run = manifest('preparation')
    (OUT / 'observation_table.csv').write_bytes((V7 / 'results/observation_table.csv').read_bytes())
    obs = pd.read_csv(OUT / 'observation_table.csv')
    definitions = []; scales = []; masses = []; intersections = []
    for fold in cfg()['folds']:
        with np.load(V7 / 'cache' / f'masks_{fold["cutoff"]}.npz') as data:
            full = data['full'].copy(); te = data['testing'].copy()
        assert obs.joint_completed.iloc[full].le(fold['cutoff']).all()
        assert int(obs.joint_completed.iloc[full].str[:4].max()) == int(fold['cutoff'][:4])
        arrays = dict(testing=te)
        for policy in cfg()['policies']:
            tr, weights = define_policy(obs, full, policy)
            arrays[policy] = tr; arrays[policy+'_weights'] = weights
            _, s = target_arrays(tr, weights)
            scales.append(dict(cutoff=fold['cutoff'], policy=policy, **s))
            definitions.append(dict(cutoff=fold['cutoff'], policy=policy, train_n=len(tr), full_train_n=len(full),
                validation_n=len(te), first_anchor=obs.date.iloc[tr[0]], last_anchor=obs.date.iloc[tr[-1]],
                unique_fraction=len(tr)/len(full), weight_min=float(weights.min()), weight_max=float(weights.max()),
                sum_weights=float(weights.sum()), presentations_per_epoch=len(full), presentations_per_unique_row=len(full)/len(tr),
                optimizer_steps_per_epoch=int(np.ceil(len(full)/cfg()['training']['batch_size']))))
            selected = obs.iloc[tr][['date']].copy(); selected['weight'] = weights
            selected['quarter'] = pd.to_datetime(selected.date).dt.to_period('Q').astype(str)
            for quarter, g in selected.groupby('quarter'):
                masses.append(dict(cutoff=fold['cutoff'], policy=policy, quarter=quarter, rows=len(g),
                                   sample_share=len(g)/len(tr), weight_mass=float(g.weight.sum()/weights.sum())))
        for first, second in [('disjoint0','disjoint1'),('disjoint0','disjoint2'),('disjoint1','disjoint2')]:
            count = len(np.intersect1d(arrays[first], arrays[second]))
            assert count == 0
            intersections.append(dict(cutoff=fold['cutoff'],first=first,second=second,shared_anchors=count,
                                      union_anchors=len(np.union1d(arrays[first],arrays[second]))))
        np.savez_compressed(CACHE / f'policies_{fold["cutoff"]}.npz', **arrays)
    pd.DataFrame(definitions).to_csv(OUT / 'policy_definitions.csv', index=False)
    pd.DataFrame(masses).to_csv(OUT / 'quarter_masses.csv', index=False)
    pd.DataFrame(intersections).to_csv(OUT / 'phase_mask_overlap.csv', index=False)
    save(OUT / 'policy_scales.json', scales)
    save(OUT / 'fit_plan.json', fit_plan())
    previous = pd.read_csv(V7 / 'results/reused_full_predictions.csv')
    reused = previous[previous.role.eq('candidate20')].copy()
    reused['policy'] = 'full'; reused['source'] = 'reused_round6_full'
    reused['training_mean'] = [next(s['returns_mean'] for s in scales if s['cutoff']==c and s['policy']=='full') for c in reused.cutoff]
    assert len(reused) == 423
    reused.to_csv(OUT / 'reused_full_predictions.csv', index=False)
    refs = json.loads((V7 / 'results/reused_checkpoint_manifest.json').read_text(encoding='utf-8'))
    selected = [dict(r, policy='full') for r in refs if r['role']=='candidate20']
    assert len(selected) == 9
    for r in selected:
        assert sha(PROJECT / r['project_file']) == r['sha256']
    save(OUT / 'reused_checkpoint_manifest.json', selected)
    frozen = list(CACHE.glob('*.npz')) + [OUT / n for n in ['observation_table.csv', 'policy_definitions.csv',
        'quarter_masses.csv', 'phase_mask_overlap.csv', 'policy_scales.json', 'fit_plan.json', 'reused_full_predictions.csv', 'reused_checkpoint_manifest.json']]
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(), planned_fits=len(fit_plan()),
               reused_checkpoints=len(selected), reused_predictions=len(reused),
               local_sha256={str(p.relative_to(ROOT)):sha(p) for p in frozen})
    save(OUT / 'preparation_manifest.json', run)
    print(pd.DataFrame(definitions)[['cutoff','policy','train_n','weight_min','weight_max']].to_string(index=False))
    print(json.dumps(dict(new_fits=45, reused_checkpoints=9, old_evidence_files=len(run['old_evidence']))))


if __name__ == '__main__':
    main()

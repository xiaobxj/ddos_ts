"""Freeze masks, one candidate and paired legacy evidence before fitting."""
from common9 import *


def main():
    legacy.initialize();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    assert not (OUT/'preparation_manifest.json').exists()
    run=manifest('preparation')
    (OUT/'observation_table.csv').write_bytes((V8/'results/observation_table.csv').read_bytes())
    rows=[];exposures=[]
    for fold in cfg()['folds']:
        with np.load(V8/'cache'/f'policies_{fold["cutoff"]}.npz') as data:
            tr=data['full'].copy();te=data['testing'].copy()
        np.savez_compressed(CACHE/f'masks_{fold["cutoff"]}.npz',training=tr,testing=te)
        for rule in ['legacy','balanced']:
            chunks=batches(np.arange(len(tr)),rule);sizes=[len(b) for b in chunks]
            rows.append(dict(cutoff=fold['cutoff'],validation_year=int(fold['cutoff'][:4])+1,rule=rule,training_n=len(tr),
                validation_n=len(te),steps=len(chunks),batch_sizes=sizes,minimum_batch=min(sizes),maximum_batch=max(sizes),
                final_batch=sizes[-1],loss_scale_min=min(batch_factor(k,len(tr),rule) for k in sizes),
                loss_scale_max=max(batch_factor(k,len(tr),rule) for k in sizes)))
            for seed in cfg()['training']['seeds']:
                rng=np.random.default_rng(seed);coeff=np.zeros(len(tr));draws=np.zeros(len(tr),dtype=int)
                for _ in range(20):
                    order=legacy.epoch_order(len(tr),len(tr),rng)
                    for b in batches(order,rule):
                        np.add.at(coeff,b,batch_factor(len(b),len(tr),rule)/len(b))
                        np.add.at(draws,b,1)
                ratio=coeff/coeff.sum()*len(tr)
                assert np.all(draws==20)
                exposures.append(dict(cutoff=fold['cutoff'],seed=seed,rule=rule,min_presentations=int(draws.min()),max_presentations=int(draws.max()),
                    nominal_coefficient_ratio_min=float(ratio.min()),nominal_coefficient_ratio_max=float(ratio.max()),
                    nominal_coefficient_ratio_sd=float(ratio.std()),actual_parameter_influence=False))
    pd.DataFrame(rows).to_csv(OUT/'batch_definitions.csv',index=False)
    pd.DataFrame(exposures).to_csv(OUT/'nominal_exposure.csv',index=False)
    save(OUT/'fit_plan.json',fit_plan())
    old=pd.read_csv(V8/'results/reused_full_predictions.csv');assert len(old)==423
    old['rule']='legacy';old['source']='reused_round6_full'
    old.to_csv(OUT/'legacy_predictions.csv',index=False)
    refs=json.loads((V8/'results/reused_checkpoint_manifest.json').read_text(encoding='utf-8'))
    assert len(refs)==9
    for r in refs:
        assert sha(PROJECT/r['project_file'])==r['sha256']
        r['rule']='legacy'
    save(OUT/'legacy_checkpoint_manifest.json',refs)
    curves=pd.read_csv(V6/'results/main_training_curves.csv')
    curves=curves[(curves.arm=='combined')&(curves.history=='full')].copy()
    assert len(curves)==180
    curves['rule']='legacy';curves.to_csv(OUT/'legacy_training_curves.csv',index=False)
    files=list(CACHE.glob('*.npz'))+[OUT/n for n in ['observation_table.csv','batch_definitions.csv','nominal_exposure.csv',
        'fit_plan.json','legacy_predictions.csv','legacy_checkpoint_manifest.json','legacy_training_curves.csv']]
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),planned_new_fits=9,reused_models=9,
               local_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    save(OUT/'preparation_manifest.json',run)
    print(pd.DataFrame(rows).to_string(index=False));print(f'Preserved {len(run["old_evidence"])} prior files; 9 planned new fits.')


if __name__=='__main__':main()

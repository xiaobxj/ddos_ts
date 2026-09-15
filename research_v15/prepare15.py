from common15 import *

def main():
    assert not (OUT/'preparation_manifest.json').exists(),'Preserve preparation'
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(exist_ok=True)
    run=manifest('preparation');run['old_evidence']=old_evidence();local=[]
    mapping={n:n for n in ['observation_table.csv','training_rows.csv','training_metadata.json','classification_baselines.csv']}
    mapping.update({'all_seed_predictions.csv':'archived_seed_predictions.csv','all_ensemble_predictions.csv':'archived_ensemble_predictions.csv'})
    for old,new in mapping.items():shutil.copyfile(V14/'results'/old,OUT/new);local.append(OUT/new)
    refs=[]
    for family in cfg()['families']:
        name='archived_models.json' if family['source']=='archived20' else 'models.json'
        for old in read(V14/'results'/name):
            r=dict(old,family=family['name'],source=family['source']);assert sha(PROJECT/r['project_file'])==r['sha256'];refs.append(r)
    assert len(refs)==18 and len({identity(r) for r in refs})==18
    save(OUT/'backbones.json',refs);local.append(OUT/'backbones.json')
    obs=pd.read_csv(OUT/'observation_table.csv');metadata=read(OUT/'training_metadata.json')
    with np.load(V5/'cache/targets.npz') as raw:
        for fold,n_expected,v_expected in zip(cfg()['folds'],[1678,1921,2165],[49,46,46]):
            cutoff=fold['cutoff'];tr=np.flatnonzero(obs.joint_completed.le(cutoff).to_numpy())
            te=np.flatnonzero((obs.date.gt(cutoff)&obs.date.le(fold['end'])&obs.weekday.eq(4)&obs.joint_completed.le('2020-12-31')).to_numpy())
            assert len(tr)==n_expected and len(te)==v_expected and not len(np.intersect1d(tr,te))
            with np.load(V14/'cache'/f'training_{cutoff}.npz') as d:
                np.testing.assert_array_equal(d['row_index'],tr);np.testing.assert_array_equal(d['direction'],(raw['returns'][tr]>0).astype(float))
            with np.load(V14/'cache'/f'validation_{cutoff}.npz') as d:np.testing.assert_array_equal(d['row_index'],te)
            assert metadata[cutoff]['frequency']==float((raw['returns'][tr]>0).mean())
    run.update(finished_utc=now(),frozen_backbones=18,primary_heads=18,neural_training_steps=0,training_feature_rows=34584,
        validation_weeks=141,local_sha256={str(p.relative_to(ROOT)):sha(p) for p in local})
    save(OUT/'preparation_manifest.json',run)
    print(json.dumps(dict(status='FROZEN',old_files=len(run['old_evidence']),backbones=18,heads=18,
        protocol_sha256=run['protocol_sha256'])),flush=True)

if __name__=='__main__':main()

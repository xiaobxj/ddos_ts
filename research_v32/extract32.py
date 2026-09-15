from common32 import *
def main():
    legacy.initialize();check_frozen();assert not (OUT/'extraction_manifest.json').exists();run=manifest('extraction');members=csv('bank_membership');models=read(OUT/'source_models.json');heads=read(OUT/'source_heads.json');tests=read(OUT/'source_testing_features.json');records=[];files=[]
    for modelref in models:
        annual=modelref['cutoff'];seed=modelref['seed'];model,state=load_model(modelref);plan=members[members.encoder_cutoff.eq(annual)];rows=plan.row_index.to_numpy(int);missing=plan[plan.source.eq(2)].row_index.to_numpy(int)
        h=next(h for h in heads if h['cutoff']==annual and h['seed']==seed and h['method']=='learned_market');t=next(t for t in tests if t['cutoff']==annual and t['seed']==seed);d=arrays(h);td=arrays(t);values=legacy.batch_tensors(missing);f,z=neural.extract_features(model,values);assert len(missing)>0
        lookup={int(i):x for i,x in zip(d['row_index'],d['features'])};lookup.update({int(i):x for i,x in zip(td['row_index'],td['features'])});assert not set(missing)&set(lookup);lookup.update({int(i):x for i,x in zip(missing,f)});features=np.stack([lookup[int(i)] for i in rows])
        p=CACHE/f'bank_{annual}_{seed}.npz';np.savez_compressed(p,row_index=rows,features=features,source=plan.source.to_numpy(int),missing_rows=missing,missing_features=f,missing_native_standardized=z);files.append(p)
        records.append(dict(cutoff=annual,seed=seed,bank_rows=len(rows),new_rows=len(missing),annual_training_source=h,annual_testing_source=t,model_project_file=modelref['project_file'],**ref(p)))
        assert training.object_hash(model.state_dict())==modelref['model_sha256'] and not any(p.requires_grad or p.grad is not None for p in model.parameters());save(OUT/'feature_banks.json',records);del model,state,values;torch.cuda.empty_cache();print(f'Frozen annual feature bank {len(records)}/18: {annual}, seed {seed}, {len(missing)} missing daily rows.',flush=True)
    files.append(OUT/'feature_banks.json');check_frozen();finish(run,files,annual_networks=18,new_neural_fits=0,new_head_fits=0,extension_feature_rows=sum(r['new_rows'] for r in records),new_bank_extension_passes=18,weekly_scoring=False)
if __name__=='__main__':main()

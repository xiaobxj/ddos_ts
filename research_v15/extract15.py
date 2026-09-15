from common15 import *

def main():
    legacy.initialize();check_frozen();path=OUT/'extraction_manifest.json';assert not path.exists(),'Preserve extraction'
    run=manifest('extraction');save(path,run);start=time.time();records=[];diagnostics=[];files=[];rows=0
    for ref in read(OUT/'backbones.json'):
        model,state=load_backbone(ref);values,tr,y=training_data(ref['cutoff']);f,native=extract_features(model,values)
        x,mean,sd=normalize_train(f,cfg()['standardization']['sd_floor']);raw_sd=f.astype(float).std(axis=0)
        p=CACHE/f'training_{identity(ref)}.npz'
        weights=model.return_head.weight.detach().cpu().numpy().astype(float).flatten();bias=float(model.return_head.bias.detach().cpu()[0])
        np.savez_compressed(p,features=f,standardized=x,mean=mean,sd=sd,direction=y,row_index=tr,native_output=native,
            native_weights=weights,native_bias=np.array(bias))
        singular=np.linalg.svd(x,compute_uv=False)
        r=dict(ref,feature_file=str(p.relative_to(ROOT)),feature_sha256=sha(p),train_n=len(tr));records.append(r);files.append(p)
        diagnostics.append(dict(family=ref['family'],cutoff=ref['cutoff'],seed=ref['seed'],train_n=len(tr),feature_dimensions=25,
            minimum_raw_sd=float(raw_sd.min()),maximum_raw_sd=float(raw_sd.max()),floored_coordinates=int((raw_sd<1e-6).sum()),
            rank_tolerance=1e-8,numerical_rank=int((singular>1e-8).sum()),maximum_singular_value=float(singular.max()),minimum_singular_value=float(singular.min()),
            unchanged_model_sha256=object_hash(model.state_dict())))
        rows+=len(tr);assert sha(PROJECT/ref['project_file'])==ref['sha256']
        save(OUT/'feature_models.json',records);pd.DataFrame(diagnostics).to_csv(OUT/'feature_diagnostics.csv',index=False)
        print(f"Training feature extraction {len(records)}/18: {identity(ref)}; {time.time()-start:.1f}s",flush=True)
        del model,values;torch.cuda.empty_cache()
    assert len(records)==18 and rows==34584;files += [OUT/'feature_models.json',OUT/'feature_diagnostics.csv']
    run.update(finished_utc=now(),elapsed_seconds=time.time()-start,backbones=18,training_rows=rows,validation_features_extracted=0,
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in files});save(path,run)
    print('Training feature extraction complete; no validation feature extraction.',flush=True)

if __name__=='__main__':main()

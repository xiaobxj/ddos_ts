from common29 import *

def main():
    legacy.initialize();check_frozen();assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');obs,price,_=data();missing=csv('extension_membership');heads=read(OUT/'source_heads.json');models=read(OUT/'source_models.json');scales=read(V28/'results/training_scales.json');extra=[];files=[];testrefs=[]
    for cutoff,g in missing.groupby('cutoff',sort=True):
        rows=g.sort_values('row_index').row_index.to_numpy(int);tr=training_rows(obs,cutoff)
        for seed in cfg()['seeds']:
            r=next(r for r in models if r['cutoff']==cutoff and r['seed']==seed);model,_=load_model(r);hs=[next(h for h in heads if h['cutoff']==cutoff and h['seed']==seed and h['method']==m) for m in LEARNED]
            f,native=forecasts(model,rows,scales[f'rolling5_{cutoff}']);mf=market(price,obs,rows);u=gates(price,obs,rows);xx=apply_pipeline(f,mf,u,arrays(hs[0]))
            path=CACHE/f'extension_{cutoff}_{seed}.npz';np.savez_compressed(path,features=f,native=native,market_features=mf,gate=u,row_index=rows,x18=xx[0],x19=xx[1],x23=xx[2]);files.append(path);testrefs.append(dict(cutoff=cutoff,seed=seed,**ref(path)))
            for h,x in zip(hs,xx):
                p=probability(design(x)@np.asarray(h['coefficients']));extra.append(predict_rows(obs,rows,cutoff,h['method'],seed,p,p))
            extra.append(predict_rows(obs,rows,cutoff,'native_mse',seed,native))
            if seed==cfg()['seeds'][0]:
                p=np.full(len(rows),float((obs.exec_return.iloc[tr]>0).mean()));extra.append(predict_rows(obs,rows,cutoff,'training_frequency',-1,p,p))
            assert training.object_hash(model.state_dict())==r['model_sha256'];del model;torch.cuda.empty_cache();print(f'Extended frozen {cutoff} seed {seed}, {len(rows)} previously unscored weeks.',flush=True)
    archive=archive_predictions();extended=pd.concat(extra,ignore_index=True) if extra else archive.iloc[:0].copy();assert len(extended)==16*len(missing)
    unique=pd.concat([archive,extended],ignore_index=True);assert not unique.duplicated(['history','cutoff','method','seed','date']).any();candidate=route_predictions(unique);models=pd.concat([csv('baseline_model_predictions'),candidate],ignore_index=True);ensemble=ensemble_from(models)
    assert len(models)==17408 and len(ensemble)==6528;pd.testing.assert_frame_equal(ensemble[ensemble.history.ne('state90')].reset_index(drop=True),csv('baseline_ensemble_predictions'),check_dtype=False,atol=1e-14,rtol=0)
    extended.to_csv(OUT/'extension_predictions.csv',index=False);candidate.to_csv(OUT/'state_predictions.csv',index=False);models.to_csv(OUT/'model_predictions.csv',index=False);ensemble.to_csv(OUT/'ensemble_predictions.csv',index=False);save(OUT/'testing_features.json',testrefs)
    files += [OUT/n for n in ['extension_predictions.csv','state_predictions.csv','model_predictions.csv','ensemble_predictions.csv','testing_features.json']]
    check_frozen();finish(run,files,new_neural_fits=0,new_head_fits=0,extension_networks=len(testrefs),extension_model_rows=len(extended),reused_state_prediction_rows=len(candidate)-len(extended),weeks=272,model_rows=len(models),ensemble_rows=len(ensemble));print('Scoring PASS: frozen baselines preserved; state schedule scored.',flush=True)
if __name__=='__main__':main()

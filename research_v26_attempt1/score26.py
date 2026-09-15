from common26 import *
def main():
    legacy.initialize();check_frozen();check_phase('training');assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');obs,price,_=data();scales=read(OUT/'training_scales.json');heads=read(OUT/'heads.json');preds=[];files=[];features_refs=[]
    for r in read(OUT/'models.json'):
        fold=next(f for f in cfg()['folds'] if f['cutoff']==r['cutoff']);tr,te=indices(obs,fold);model,_=load_model(r);f,native=forecasts(model,te,scales[r['cutoff']]);mf=market(price,obs,te);u=gates(price,obs,te)
        hs=[next(h for h in heads if h['cutoff']==r['cutoff'] and h['seed']==r['seed'] and h['method']==m) for m in LEARNED];d=arrays(hs[0]);xx=apply_pipeline(f,mf,u,d)
        p=CACHE/f"testing_{r['cutoff']}_{r['seed']}.npz";np.savez_compressed(p,features=f,native=native,market_features=mf,gate=u,row_index=te,x18=xx[0],x19=xx[1],x23=xx[2]);files.append(p);features_refs.append(dict(cutoff=r['cutoff'],seed=r['seed'],**ref(p)))
        for h,x in zip(hs,xx):
            z=design(x)@np.asarray(h['coefficients']);p=probability(z);preds.append(rows_for_predictions(obs,te,r['cutoff'],h['method'],r['seed'],p,p))
        preds.append(rows_for_predictions(obs,te,r['cutoff'],'native_mse',r['seed'],native))
        if r['seed']==cfg()['seeds'][0]:
            rate=float((obs.exec_return.iloc[tr]>0).mean());p=np.full(len(te),rate);preds.append(rows_for_predictions(obs,te,r['cutoff'],'training_frequency',-1,p,p))
        del model;torch.cuda.empty_cache();print(f"Scored {len(features_refs)}/18: {r['cutoff']} seed {r['seed']}",flush=True)
    models=pd.concat(preds,ignore_index=True);ensemble=ensemble_from(models);assert len(models)==4352 and len(ensemble)==1632
    models.to_csv(OUT/'model_predictions.csv',index=False);ensemble.to_csv(OUT/'ensemble_predictions.csv',index=False);save(OUT/'testing_features.json',features_refs)
    files += [OUT/n for n in ['model_predictions.csv','ensemble_predictions.csv','testing_features.json']];finish(run,files,weeks=272,models=4352,ensembles=1632,neural_test_observations=816)
if __name__=='__main__':main()

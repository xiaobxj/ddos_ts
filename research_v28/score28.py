from common28 import *
def main():
    legacy.initialize();check_frozen();check_phase('training');assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');obs,price,_=data();scales=read(OUT/'training_scales.json');heads=read(OUT/'heads.json');preds=[];files=[];feature_refs=[]
    for r in read(OUT/'models.json'):
        history=r['history'];cutoff=r['cutoff'];fold=next(f for f in cfg()['folds'] if f['cutoff']==cutoff);tr,te=indices(obs,fold,history);model,_=load_model(r);f,native=forecasts(model,te,scales[f'{history}_{cutoff}']);mf=market(price,obs,te);u=gates(price,obs,te)
        hs=[next(h for h in heads if h['history']==history and h['cutoff']==cutoff and h['seed']==r['seed'] and h['method']==m) for m in LEARNED];d=arrays(hs[0]);xx=apply_pipeline(f,mf,u,d)
        p=CACHE/f"testing_{history}_{cutoff}_{r['seed']}.npz";np.savez_compressed(p,features=f,native=native,market_features=mf,gate=u,row_index=te,x18=xx[0],x19=xx[1],x23=xx[2]);files.append(p);feature_refs.append(dict(history=history,cutoff=cutoff,seed=r['seed'],**ref(p)))
        for h,x in zip(hs,xx):
            p=probability(design(x)@np.asarray(h['coefficients']));preds.append(predict_rows(obs,te,history,cutoff,h['method'],r['seed'],p,p))
        preds.append(predict_rows(obs,te,history,cutoff,'native_mse',r['seed'],native))
        if r['seed']==cfg()['seeds'][0]:
            p=np.full(len(te),float((obs.exec_return.iloc[tr]>0).mean()));preds.append(predict_rows(obs,te,history,cutoff,'training_frequency',-1,p,p))
        del model;torch.cuda.empty_cache();print(f"Scored {len(feature_refs)}/138: {history} {cutoff} seed {r['seed']}",flush=True)
    unique=pd.concat(preds,ignore_index=True);assert len(unique)==cfg()['budget']['unique_model_prediction_rows'];unique.to_csv(OUT/'unique_model_predictions.csv',index=False)
    models=pd.concat([csv('baseline_model_predictions'),route_predictions(unique)],ignore_index=True);ensemble=ensemble_from(models);assert len(models)==30464 and len(ensemble)==11424
    pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),csv('baseline_ensemble_predictions'),check_dtype=False,atol=1e-14,rtol=0)
    models.to_csv(OUT/'model_predictions.csv',index=False);ensemble.to_csv(OUT/'ensemble_predictions.csv',index=False);save(OUT/'testing_features.json',feature_refs)
    files += [OUT/n for n in ['unique_model_predictions.csv','model_predictions.csv','ensemble_predictions.csv','testing_features.json']];finish(run,files,weeks=272,new_model_rows=17408,reused_model_rows=13056,ensemble_rows=11424,unique_neural_test_observations=sum(f['test_n'] for f in cfg()['folds'])*6)
if __name__=='__main__':main()

from common35 import *
def main():
    legacy.initialize();check_frozen();assert not (OUT/'fitting_manifest.json').exists();run=manifest('fitting');obs,price,targets=data();ids=np.arange(len(obs));mf=market(price,obs,ids);u=gates(price,obs,ids);y=(targets['returns']>0).astype(float)
    banks=read(OUT/'feature_banks.json');annuals=read(OUT/'source_heads.json');heads=[];traces=[];training_refs=[];files=[]
    for cutoff in cfg()['decision_dates']:
        annual=annual_for(cutoff)
        if cutoff==annual:continue
        sets=member_sets(obs,cutoff)
        for seed in cfg()['seeds']:
            bank=arrays(next(b for b in banks if b['cutoff']==annual and b['seed']==seed));hs=[next(h for h in annuals if h['cutoff']==annual and h['seed']==seed and h['method']==m) for m in LEARNED];d=arrays(hs[0])
            for arm in ['add','remove']:
                rows=sets[arm];xx,labels=training_interface(bank,rows,mf,u,y,d);ts,tt,ms=fit_coefficients(xx,labels);path=CACHE/f'training_{arm}_{cutoff}_{seed}.npz';np.savez_compressed(path,row_index=rows,direction=labels,x18=xx[0],x19=xx[1],x23=xx[2]);files.append(path);training_refs.append(dict(arm=arm,cutoff=cutoff,encoder_cutoff=annual,seed=seed,**ref(path)))
                for m,h,t,trace,met in zip(LEARNED,hs,ts,tt,ms):
                    job=f'{arm}_{m}_{cutoff}_{seed}';heads.append(dict(arm=arm,history=ARMS[arm],method=m,cutoff=cutoff,encoder_cutoff=annual,transform_cutoff=annual,seed=seed,job=job,train_n=len(rows),coefficients=t.tolist(),iterations=trace[-1]['iteration'],cache_file=h['cache_file'],cache_sha256=h['cache_sha256'],annual_job=h['job'],**met));traces.extend(dict(job=job,**r) for r in trace)
            print(f'Fitted {len(heads)}/408 addition/removal heads; {cutoff} seed {seed}; no weekly scoring.',flush=True)
    assert len(heads)==408 and len(training_refs)==102
    for n,v in [('heads.json',heads),('training_designs.json',training_refs)]:p=OUT/n;save(p,v);files.append(p)
    p=OUT/'solver_trace.csv';pd.DataFrame(traces).to_csv(p,index=False);files.append(p);check_frozen();finish(run,files,new_neural_fits=0,new_feature_inference=0,new_transform_fits=0,new_head_fits=408,all_converged=True,weekly_scoring_during_fit=False)
if __name__=='__main__':main()

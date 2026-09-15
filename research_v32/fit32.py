from common32 import *
def main():
    legacy.initialize();check_frozen();check_phase('extraction');assert not (OUT/'fitting_manifest.json').exists();run=manifest('fitting');obs,price,targets=data();allrows=np.arange(len(obs));mf=market(price,obs,allrows);u=gates(price,obs,allrows);y=(targets['returns']>0).astype(float);banks=read(OUT/'feature_banks.json');jobs=csv('jobs');heads=[];traces=[];metrics=[];files=[]
    for j in jobs[jobs.refit].itertuples():
        rows=training_rows(obs,j.head_cutoff)
        for seed in cfg()['seeds']:
            bankref=next(b for b in banks if b['cutoff']==j.encoder_cutoff and b['seed']==seed);bank=arrays(bankref);f,m,v,labels=training_interface(bank,rows,mf,u,y);d,coefs,ts,ms=fit_pipeline(f,m,v,labels);d['row_index']=rows;p=CACHE/f'pipeline_{j.head_cutoff}_{seed}.npz';np.savez_compressed(p,**d);files.append(p)
            for method,theta,trace,metric in zip(LEARNED,coefs,ts,ms):
                job=f'{method}_{j.head_cutoff}_{seed}';h=dict(history=NEW[0],method=method,cutoff=j.head_cutoff,encoder_cutoff=j.encoder_cutoff,seed=seed,job=job,train_n=len(rows),coefficients=theta.tolist(),iterations=trace[-1]['iteration'],model_project_file=bankref['model_project_file'],**ref(p),**metric);heads.append(h);traces.extend(dict(job=job,**t) for t in trace);metrics.append(dict(job=job,method=method,head_cutoff=j.head_cutoff,encoder_cutoff=j.encoder_cutoff,seed=seed,train_n=len(rows),**metric))
            save(OUT/'heads.json',heads);pd.DataFrame(traces).to_csv(OUT/'solver_trace.csv',index=False);print(f'Fitted {len(heads)}/204 quarter heads: {j.head_cutoff}, annual encoder {j.encoder_cutoff}, seed {seed}; no weekly scoring.',flush=True)
    assert len(heads)==204;pd.DataFrame(metrics).to_csv(OUT/'training_metrics.csv',index=False);files += [OUT/n for n in ['heads.json','solver_trace.csv','training_metrics.csv']];check_frozen();finish(run,files,new_neural_fits=0,head_jobs=51,new_head_fits=204,new_projections=102,newton_iterations=sum(h['iterations'] for h in heads),all_converged=True,weekly_scoring_during_fit=False)
if __name__=='__main__':main()

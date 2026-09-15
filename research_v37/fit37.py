from common37 import *
def main():
    check_frozen();assert not (OUT/'fitting_manifest.json').exists();run=manifest('fitting');obs,price,targets=data();allrows=np.arange(len(obs));mf=prior.market(price,obs,allrows);u=prior.gates(price,obs,allrows);y=(targets['returns']>0).astype(float);states=csv('state_observations');banks=read(OUT/'feature_banks.json');annuals=read(OUT/'annual_heads.json');heads=[];refs=[];flat=[];files=[]
    for cutoff in cfg()['decision_dates']:
        annual=annual_for(cutoff);rows=new_rows(obs,cutoff)
        for seed in cfg()['seeds']:
            hs=[next(h for h in annuals if h['cutoff']==annual and h['seed']==seed and h['method']==m) for m in LEARNED]
            if len(rows):
                bank=arrays(next(b for b in banks if b['cutoff']==annual and b['seed']==seed));d=training_interface(rows,annual,bank,mf,u,y,states,hs);p=CACHE/f'correction_input_{cutoff}_{seed}.npz';np.savez_compressed(p,**d);files.append(p);refs.append(dict(cutoff=cutoff,encoder_cutoff=annual,seed=seed,**ref(p)))
            else:d=dict(annual_logits=np.empty((0,4)),direction=np.empty(0),state_index=np.empty(0,int))
            for j,m in enumerate(LEARNED):
                r=corrections(d['annual_logits'][:,j],d['direction'],d['state_index'],cfg()['correction']);key=dict(cutoff=cutoff,encoder_cutoff=annual,seed=seed,method=m);heads.append(dict(**key,annual_job=hs[j]['job'],new_n=len(rows),**r))
                flat.extend(dict(**key,kind='state',**item) for item in r['states']);flat.append(dict(**key,kind='global',state='eligible_shared',eligible=r['eligible_states']>0,**r['shared']))
        print(f'Fitted corrections at {cutoff}; mature new members={len(rows)}; no weekly scoring.',flush=True)
    assert len(heads)==276 and len(refs)==51
    for n,v in [('correction_heads.json',heads),('training_inputs.json',refs)]:p=OUT/n;save(p,v);files.append(p)
    table=pd.DataFrame(flat);p=OUT/'correction_parameters.csv';table.to_csv(p,index=False);files.append(p);fit_count=int(table.eligible.sum());check_frozen();finish(run,files,new_scalar_fits=fit_count,state_fits=int((table.eligible&table.kind.eq('state')).sum()),shared_fits=int((table.eligible&table.kind.eq('global')).sum()),new_neural_fits=0,new_transform_fits=0,new_feature_inference=0,weekly_scoring=False,all_kkt_pass=True)
    print(f'Frozen all {fit_count} active scalar correction solutions.',flush=True)
if __name__=='__main__':main()

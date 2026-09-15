from common29 import *

def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run);files=[OUT/'initial_freeze.json']
    obs,price,_=data();mf=market(price,obs,np.arange(len(obs)));events,states=build_policy(obs[['date','joint_completed']],mf)
    path=CACHE/'observed_market.npz';np.savez_compressed(path,row_index=np.arange(len(obs)),features=mf);files.append(path);refs=[]
    for cutoff,d in states.items():
        path=CACHE/f'state_{cutoff}.npz';np.savez_compressed(path,**d);files.append(path);refs.append(dict(cutoff=cutoff,**ref(path)))
    events.to_csv(OUT/'events.csv',index=False);route=route_from_events(obs,events);route.to_csv(OUT/'routing.csv',index=False);missing=required_extensions(obs,events);missing.to_csv(OUT/'extension_membership.csv',index=False)
    models=[r for r in read(V28/'results/models.json') if r['history']=='rolling5'];heads=[h for h in read(V28/'results/heads.json') if h['history']=='rolling5'];assert len(models)==69 and len(heads)==276
    selected=sorted(route.cutoff.unique());used_models=[r for r in models if r['cutoff'] in selected];used_heads=[h for h in heads if h['cutoff'] in selected]
    save(OUT/'source_models.json',used_models);save(OUT/'source_heads.json',used_heads);save(OUT/'state_refs.json',refs)
    budgets=[];oldbudget=pd.read_csv(V28/'results/budgets.csv');oldbudget=oldbudget[oldbudget.history.eq('rolling5')]
    for policy,cutoffs in [('rolling5_annual20',[c for c in cfg()['decision_dates'] if c.endswith('12-31')]),('rolling5_quarterly20',cfg()['decision_dates']),('state90',selected)]:
        g=oldbudget[oldbudget.cutoff.isin(cutoffs)];budgets.append(dict(history=policy,refit_dates=len(cutoffs),neural_fits_if_run_online=3*len(cutoffs),head_fits_if_run_online=12*len(cutoffs),optimizer_steps_if_run_online=int(g.steps_per_seed.sum())*3,sample_presentations_if_run_online=int(g.total_presentations_per_seed.sum())*3,actual_new_neural_fits=0,actual_new_head_fits=0))
    pd.DataFrame(budgets).to_csv(OUT/'budgets.csv',index=False)
    # The complete target-free schedule is generated before copying archived scores.
    base,ensemble=baselines();base.to_csv(OUT/'baseline_model_predictions.csv',index=False);ensemble.to_csv(OUT/'baseline_ensemble_predictions.csv',index=False)
    files += [OUT/n for n in ['events.csv','routing.csv','extension_membership.csv','source_models.json','source_heads.json','state_refs.json','budgets.csv','baseline_model_predictions.csv','baseline_ensemble_predictions.csv']]
    finish(run,files,old_files=len(run['old_evidence']),optional_checks=17,triggered_refits=int(events[~events.mandatory].refit.sum()),mandatory_refits=6,selected_model_cutoffs=selected,reused_selected_networks=len(used_models),new_neural_fits=0,new_head_fits=0,extra_inference_weeks=len(missing),extra_inference_networks=int(missing.cutoff.nunique())*3,new_model_prediction_rows=len(missing)*16,weeks=272,new_scoring=False)
    print(events[['cutoff','previous_model_cutoff','selected_model_cutoff','mandatory','refit','ratio']].to_string(index=False),flush=True);print(f'Policy frozen: {len(selected)} refit dates, {len(missing)} missing model-weeks; no new neural training.',flush=True)
if __name__=='__main__':main()

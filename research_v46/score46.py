from common46 import *
def main():
    check_frozen();check_phase('validation');assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');bank=csv('weekly_signal_bank');base=csv('baseline_model_predictions');be=csv('baseline_ensemble_predictions');heads=read(OUT/'correction_heads.json');hi={(h['cutoff'],h['method'],h['seed'],h['family'],h['component']):h for h in heads};gates=csv('gate_decisions').set_index(['cutoff','method','family','state']);route=csv('routing').set_index('row_index');rows=[];details=[]
    for b in bank.itertuples(index=False):
        r=b._asdict();cutoff=route.loc[b.row_index,'head_cutoff']
        for family,history in POLICIES.items():
            group=component(b.state,family);g=gates.loc[(cutoff,b.method,family,b.state)];h=hi[(cutoff,b.method,b.seed,family,group)];off=h['offset'] if g.accepted else 0.;p=float(corrected(b.annual_logit,b.probability,off));out={k:r[k] for k in base.columns};out.update(history=history,cutoff=cutoff,probability=p,score=p,direction_up=int(p>.5));rows.append(out)
            details.append(dict(history=history,cutoff=cutoff,method=b.method,seed=b.seed,row_index=b.row_index,date=b.date,state=b.state,family=family,component=group,fit_eligible=bool(h['fit_eligible']),validation_accepted=bool(g.accepted),gate_reason=g.reason,offset=off,annual_probability=b.probability,probability=p,actual_up=int(b.actual_up)))
    parts=[pd.DataFrame(rows)]
    for history in NEW:
        g=base[base.history.eq(ANNUAL)&~base.method.isin(METHODS)].copy();g['history']=history;parts.append(g)
    new=pd.concat(parts,ignore_index=True);assert len(new)==8704;models=pd.concat([base,new],ignore_index=True);assert len(models)==117504 and not models.duplicated(['history','method','seed','row_index']).any()
    # Match the inherited common26 per-group Series.mean reduction exactly.
    ensemble_rows=[]
    for (history,method,date),g in new.groupby(['history','method','date'],sort=False):
        assert len(g)==(1 if method=='training_frequency' else 3)
        r=g.iloc[0].to_dict();r.pop('seed');r['score']=float(g.score.mean());r['probability']=float(g.probability.mean()) if method!='native_mse' else np.nan;r['direction_up']=int(r['score']>0 if method=='native_mse' else r['probability']>.5);ensemble_rows.append(r)
    en=pd.DataFrame(ensemble_rows)[be.columns];assert len(en)==3264;ensemble=pd.concat([be,en],ignore_index=True);assert len(ensemble)==44064 and ensemble.history.nunique()==27
    files=[]
    for name,g in [('model_predictions',models),('ensemble_predictions',ensemble),('seed_routing',pd.DataFrame(details))]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_learned_seed_rows=6528,new_model_rows=8704,old_histories_preserved=25,gate_changes_during_scoring=0);print('Two fixed quarterly policies scored on272weeks;25oldhistories preserved.',flush=True)
if __name__=='__main__':main()

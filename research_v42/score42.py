from common42 import *

def main():
    check_frozen();check_phase('routing');assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');bank=csv('weekly_signal_bank');heads=read(OUT/'correction_heads.json');hi={(h['cutoff'],h['method'],h['seed'],h['component']):h for h in heads};decisions=csv('retention_decisions').set_index(['cutoff','method','state']);base=csv('baseline_model_predictions');obs,_,_=data();parts=[];details=[]
    for cutoff,route in csv('routing').groupby('head_cutoff',sort=True):
        ids=route.row_index.tolist()
        for method in LEARNED:
            for seed in cfg()['seeds']:
                rows=v40.input_rows(bank,ids,method,seed);offsets=[]
                for r in rows.itertuples():
                    d=decisions.loc[(cutoff,method,r.state)];used=usable(d.source_cutoff,d.expiry,r.date);off=hi[(d.source_cutoff,method,seed,r.state)]['offset'] if used else 0.;offsets.append(off)
                    if used:assert d.source_encoder_cutoff==r.encoder_cutoff and d.source_cutoff<=cutoff<r.date<=d.expiry
                    details.append(dict(history=NEW[0],cutoff=cutoff,method=method,seed=seed,row_index=r.row_index,date=r.date,state=r.state,action=d.action,original_accepted=bool(d.original_accepted),original_reason=d.original_reason,retention_reason=d.retention_reason,source_cutoff=d.source_cutoff if used else None,expiry=d.expiry if used else None,source_encoder_cutoff=d.source_encoder_cutoff if used else None,source_age_days=int((pd.Timestamp(r.date)-pd.Timestamp(d.source_cutoff)).days) if used else None,used_retained_source=bool(used and d.action=='carry'),offset=off,annual_probability=r.probability))
                probs=v40.corrected_probability(rows.annual_logit.to_numpy(),rows.probability.to_numpy(),np.array(offsets));g=prior.rows_for_predictions(obs,np.array(ids),cutoff,method,seed,probs,probs);g.insert(0,'history',NEW[0]);parts.append(g)
    control=base[base.history.eq(ANNUAL)&~base.method.isin(LEARNED)].copy();control['history']=NEW[0];parts.append(control);new=pd.concat(parts,ignore_index=True);assert len(new)==4352;models=pd.concat([base,new],ignore_index=True);assert len(models)==100096 and not models.duplicated(['history','method','seed','date']).any();ensemble=v37.ensemble_from(models);assert len(ensemble)==37536
    pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),csv('baseline_ensemble_predictions'),check_dtype=False,atol=1e-14,rtol=0);effects=pd.DataFrame(details).merge(new[['history','method','seed','row_index','probability','actual_up']],on=['history','method','seed','row_index'],validate='one_to_one');files=[]
    for name,g in [('model_predictions',models),('ensemble_predictions',ensemble),('seed_routing',effects)]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_model_rows=4352,new_learned_seed_rows=3264,gate_changes_during_scoring=0);print('One bounded-retention candidate scored on272weeks; old22histories unchanged.',flush=True)
if __name__=='__main__':main()

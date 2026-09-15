from common39 import *
def main():
    check_frozen();check_phase('validation');assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');bank=csv('weekly_signal_bank');heads=read(OUT/'correction_heads.json');gate=csv('gate_decisions').set_index(['cutoff','method','family','component']);base=csv('baseline_model_predictions');obs,_,_=data();parts=[];effects=[]
    for cutoff,g in csv('routing').groupby('head_cutoff',sort=True):
        ids=g.row_index.tolist()
        for m in LEARNED:
            for seed in cfg()['seeds']:
                rows=input_rows(bank,ids,m,seed)
                for policy,history in POLICIES.items():
                    family=policy.split('_')[0];gated=policy.endswith('gated');deltas=[]
                    for r in rows.itertuples():
                        component=r.state if family=='state' else 'all';h=next(h for h in heads if h['cutoff']==cutoff and h['method']==m and h['seed']==seed and h['family']==family and h['component']==component);decision=gate.loc[(cutoff,m,family,component)];off=h['offset'] if not gated or decision.accepted else 0.;deltas.append(off);effects.append(dict(history=history,cutoff=cutoff,method=m,seed=seed,row_index=r.row_index,date=r.date,state=r.state,family=family,component=component,fit_eligible=h['fit_eligible'],validation_accepted=bool(decision.accepted),gate_reason=decision.reason,offset=off,annual_probability=r.probability))
                    p=corrected_probability(rows.annual_logit.to_numpy(),rows.probability.to_numpy(),np.array(deltas));pred=prior.rows_for_predictions(obs,np.array(ids),cutoff,m,seed,p,p);pred.insert(0,'history',history);parts.append(pred)
    for history in NEW:
        g=base[base.history.eq(ANNUAL)&~base.method.isin(LEARNED)].copy();g['history']=history;parts.append(g)
    new=pd.concat(parts,ignore_index=True);assert len(new)==17408;models=pd.concat([base,new],ignore_index=True);assert len(models)==87040 and not models.duplicated(['history','method','seed','date']).any();ensemble=v37.ensemble_from(models);assert len(ensemble)==32640
    pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),csv('baseline_ensemble_predictions'),check_dtype=False,atol=1e-14,rtol=0);details=pd.DataFrame(effects);details=details.merge(new[['history','method','seed','row_index','probability','actual_up']],on=['history','method','seed','row_index'],validate='one_to_one');files=[]
    for name,g in [('model_predictions',models),('ensemble_predictions',ensemble),('seed_routing',details)]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_model_rows=17408,new_learned_seed_rows=13056,gate_changes_during_scoring=0);print('Four fixed weekly calibration policies scored on272weeks; old16histories preserved.',flush=True)
if __name__=='__main__':main()

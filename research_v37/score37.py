from common37 import *
def main():
    check_frozen();check_phase('fitting');assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');obs,price,_=data();route=csv('routing');states=csv('state_observations');banks=read(OUT/'feature_banks.json');annuals=read(OUT/'annual_heads.json');heads=read(OUT/'correction_heads.json');base=csv('baseline_model_predictions');parts=[];details=[];maxgap=0.
    for cutoff in cfg()['decision_dates']:
        annual=annual_for(cutoff);rows=route[route.head_cutoff.eq(cutoff)].row_index.to_numpy(int);s=state_ids(rows,annual,states)
        for seed in cfg()['seeds']:
            ah=[next(h for h in annuals if h['cutoff']==annual and h['seed']==seed and h['method']==m) for m in LEARNED];bank=arrays(next(b for b in banks if b['cutoff']==annual and b['seed']==seed));xx=apply_pipeline(bank_slice(bank,rows),prior.market(price,obs,rows),prior.gates(price,obs,rows),arrays(ah[0]))
            for m,x,h0 in zip(LEARNED,xx,ah):
                z=design(x)@np.array(h0['coefficients']);old=base[base.history.eq(ANNUAL)&base.method.eq(m)&base.seed.eq(seed)&base.row_index.isin(rows)].sort_values('row_index');np.testing.assert_array_equal(rows,old.row_index);p0=old.probability.to_numpy();gap=float(abs(probability(z)-p0).max());maxgap=max(maxgap,gap);assert gap<1e-12
                h=next(h for h in heads if h['cutoff']==cutoff and h['seed']==seed and h['method']==m)
                for kind,history in POLICIES.items():
                    delta=get_delta(h,kind,s);p=probability(z+delta);p=np.where(delta==0,p0,p);g=prior.rows_for_predictions(obs,rows,cutoff,m,seed,p,p);g.insert(0,'history',history);parts.append(g)
                    for i,row in enumerate(rows):details.append(dict(history=history,method=m,seed=seed,row_index=int(row),date=obs.date.iloc[row],head_cutoff=cutoff,encoder_cutoff=annual,state=STATES[s[i]],new_state_n=h['states'][s[i]]['n'],eligible=h['states'][s[i]]['eligible'],annual_logit=float(z[i]),annual_probability=float(p0[i]),logit_offset=float(delta[i]),probability=float(p[i]),probability_change=float(p[i]-p0[i])))
    for history in NEW:
        g=base[base.history.eq(ANNUAL)&~base.method.isin(LEARNED)].copy();g['history']=history;parts.append(g)
    new=pd.concat(parts,ignore_index=True);assert len(new)==13056;models=pd.concat([base,new],ignore_index=True);assert len(models)==69632 and not models.duplicated(['history','method','seed','date']).any();ensemble=ensemble_from(models);assert len(ensemble)==26112
    pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),csv('baseline_ensemble_predictions'),check_dtype=False,atol=1e-14,rtol=0)
    files=[]
    for name,g in [('model_predictions',models),('ensemble_predictions',ensemble),('seed_correction_effects',pd.DataFrame(details))]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_learned_seed_predictions=9792,new_total_seed_rows=13056,maximum_annual_replay_gap=maxgap,annual_fallback_exact=True,neural_inference=0);print('Scored3 fixed correction policies,13 unchanged baselines,272weeks.',flush=True)
if __name__=='__main__':main()

from common39 import *
def validate(bank,heads,schedule,members):
    decisions=[];records=[]
    for c in schedule.itertuples():
        ids=members[members.cutoff.eq(c.cutoff)&members.role.eq('validation')].row_index.tolist()
        for m in LEARNED:
            for family,components in [('state',STATES),('global',['all'])]:
                for component in components:
                    probs=[];bases=[];selected=None;train_n=None
                    for seed in cfg()['seeds']:
                        g=input_rows(bank,ids,m,seed);g=g if component=='all' else g[g.state.eq(component)];h=next(h for h in heads if h['cutoff']==c.cutoff and h['method']==m and h['seed']==seed and h['family']==family and h['component']==component);train_n=h['n'];selected=g
                        p0=g.probability.to_numpy();p=corrected_probability(g.annual_logit.to_numpy(),p0,h['offset']);probs.append(p);bases.append(p0)
                        records.extend(dict(cutoff=c.cutoff,family=family,component=component,method=m,seed=seed,row_index=int(row.row_index),date=row.date,joint_completed=row.joint_completed,annual_probability=float(q0),candidate_probability=float(q),actual_up=int(row.actual_up),offset=h['offset']) for row,q0,q in zip(g.itertuples(),p0,p))
                    p=np.mean(probs,axis=0);p0=np.mean(bases,axis=0);decision=gate_decision(c.mode,family,train_n,len(selected),p0,p,selected.actual_up.to_numpy());decisions.append(dict(cutoff=c.cutoff,fit_cutoff=c.fit_cutoff,mode=c.mode,family=family,component=component,method=m,training_n=train_n,**decision))
    return pd.DataFrame(decisions),pd.DataFrame(records)
def main():
    check_frozen();check_phase('fitting');assert not (OUT/'validation_manifest.json').exists();run=manifest('validation');g,v=validate(csv('weekly_signal_bank'),read(OUT/'correction_heads.json'),csv('schedule'),csv('split_membership'));assert len(g)==460;files=[]
    for name,t in [('gate_decisions',g),('validation_predictions',v)]:p=OUT/f'{name}.csv';t.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,gate_cells=len(g),accepted_cells=int(g.accepted.sum()),parameters_refit_after_validation=0,future_scoring=False);print(f'Validation complete: {g.accepted.sum()}/460 cells accepted; all gates frozen before future scoring.',flush=True)
if __name__=='__main__':main()

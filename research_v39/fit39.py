from common39 import *
def main():
    check_frozen();assert not (OUT/'fitting_manifest.json').exists();run=manifest('fitting');bank=csv('weekly_signal_bank');members=csv('split_membership');heads=[];inputs=[]
    for c in csv('schedule').itertuples():
        ids=members[members.cutoff.eq(c.cutoff)&members.role.eq('training')].row_index.tolist()
        for m in LEARNED:
            for seed in cfg()['seeds']:
                train=input_rows(bank,ids,m,seed);hs=fitted(train,c.mode)
                for h in hs:heads.append(dict(cutoff=c.cutoff,fit_cutoff=c.fit_cutoff,mode=c.mode,method=m,seed=seed,**h))
                if c.mode=='ready':
                    g=train[['row_index','date','joint_completed','state','annual_logit','probability','actual_up']].copy();g.insert(0,'seed',seed);g.insert(0,'method',m);g.insert(0,'cutoff',c.cutoff);inputs.append(g)
        print(f'Offsets frozen for {c.cutoff}: {c.mode}; validation labels not used.',flush=True)
    files=[];p=OUT/'correction_heads.json';save(p,heads);files.append(p);p=OUT/'training_interface.csv';pd.concat(inputs,ignore_index=True).to_csv(p,index=False);files.append(p);p=OUT/'correction_parameters.csv';pd.DataFrame(heads).to_csv(p,index=False);files.append(p);check_frozen();finish(run,files,scalar_fits=sum(h['fit_eligible'] for h in heads),state_fits=sum(h['fit_eligible'] and h['family']=='state' for h in heads),global_fits=sum(h['fit_eligible'] and h['family']=='global' for h in heads),new_neural_fits=0,new_feature_inference=0,heldout_labels_used=False);print('All training-only calibration parameters frozen.',flush=True)
if __name__=='__main__':main()

from common47 import *
def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True);assert not (OUT/'preparation_manifest.json').exists();assert len(list(ROOT.glob('*.py')))==11
    r=manifest('preparation');r['old_evidence']=old_evidence();assert read(PREV/'results/verification.json')['status']=='PASS' and read(V28/'results/verification.json')['status']=='PASS';save(OUT/'initial_freeze.json',r);files=[OUT/'initial_freeze.json']
    for src,name in COPIES:p=OUT/name;p.write_bytes(src.read_bytes());files.append(p)
    heads,models,tests=source_inventory()
    for name,data in [('source_heads',heads),('source_models',models),('source_testing_features',tests)]:p=OUT/f'{name}.json';save(p,data);files.append(p)
    obs=csv('observation_table');members=[];summary=[]
    for cutoff in cfg()['decision_dates']:
        ids=training_rows(obs,cutoff);raw,w,age=weights_for(obs.date.iloc[ids],cutoff);y=obs.exec_return.iloc[ids].gt(0).to_numpy(float);total=float(raw.sum())
        for i,rw,nw,days in zip(ids,raw,w,age):members.append(dict(cutoff=cutoff,row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i],age_days=float(days),raw_weight=float(rw),weight=float(nw),actual_up=int(obs.exec_return.iloc[i]>0)))
        summary.append(dict(cutoff=cutoff,n=len(ids),raw_weight_sum=total,weight_sum=float(w.sum()),minimum_weight=float(w.min()),maximum_weight=float(w.max()),kish_effective_n=float(1/(w@w)),uniform_up_frequency=float(y.mean()),weighted_up_frequency=float(w@y),recent_two_year_mass=float(w[age<=730.5].sum()),maximum_joint_completed=obs.joint_completed.iloc[ids].max()))
    for name,data in [('training_weights',members),('weight_summary',summary)]:p=OUT/f'{name}.csv';pd.DataFrame(data).to_csv(p,index=False);files.append(p)
    finish(r,files,old_files=6056,new_neural_fits=0,annual_jobs=18,candidate_head_fits=72);print('R47:11sources,protocol,weights,inputs and6056oldfiles frozen.',flush=True)
if __name__=='__main__':main()

from common26 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    run=manifest('preparation');run['old_evidence']=old_evidence();obs,price,targets=data();files=[];members=[];scales={};budgets=[]
    assert len(obs)==3780 and len(price)==4046 and price.date.iloc[-1]=='2026-08-31'
    with np.load(PROJECT/'research_v5/cache/packed_raw.npz') as packed:
        for fold in cfg()['folds']:
            tr,te=indices(obs,fold);labels,s=scales_for(targets,tr);cutoff=fold['cutoff'];scales[cutoff]=s
            path=CACHE/f'training_{cutoff}.npz';np.savez_compressed(path,**{k:packed[k][tr] for k in ['patches','geometry','valid']},**labels,row_index=tr);files.append(path)
            for split,ids in [('training',tr),('testing',te)]:
                for i in ids:members.append(dict(cutoff=cutoff,split=split,row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i]))
            budgets.append(dict(**fold,neural_fits=3,epochs=60,optimizer_steps=60*((len(tr)+127)//128),presentations=60*len(tr),head_fits=12,projections=6))
    save(OUT/'training_scales.json',scales);pd.DataFrame(members).to_csv(OUT/'membership.csv',index=False);pd.DataFrame(budgets).to_csv(OUT/'budgets.csv',index=False)
    files += [OUT/n for n in ['training_scales.json','membership.csv','budgets.csv']]
    finish(run,files,old_files=len(run['old_evidence']),neural_fits=18,head_fits=72,projections=36,weeks=272,neural_execution=False)
    print(json.dumps(dict(status='FROZEN',protocol=run['protocol_sha256'],old_files=len(run['old_evidence']),neural_fits=18,weeks=272)),flush=True)
if __name__=='__main__':main()

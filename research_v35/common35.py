"""Four causal training-member sets, with unchanged annual representations and coordinates."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache'
V34=PROJECT/'research_v34';V32=PROJECT/'research_v32';V31=PROJECT/'research_v31';V28=PROJECT/'research_v28'
sys.path.insert(0,str(V34));import common34 as previous_round
base32=previous_round.base32;prior=previous_round.prior
np=prior.np;pd=prior.pd;torch=prior.torch;legacy=prior.legacy;neural=prior.neural
read=prior.read;save=prior.save;sha=prior.sha;metric=prior.metric;data=prior.data;statistics=prior.statistics;arrays=prior.arrays;ref=prior.ref;market=prior.market;gates=prior.gates
apply_pipeline=prior.apply_pipeline;design=prior.design;probability=prior.probability
METHODS=prior.METHODS;LEARNED=prior.LEARNED;PRIMARY=prior.PRIMARY
ARMS={'annual':'rolling5_annual20','add':'fixed_transform_add_only','remove':'fixed_transform_remove_only','both':'fixed_transform_step100'}
NEW=[ARMS['add'],ARMS['remove']];HISTORIES=previous_round.HISTORIES+NEW
training_rows=previous_round.training_rows;ensemble_from=previous_round.ensemble_from;annual_for=previous_round.annual_for;bank_slice=previous_round.bank_slice
training_interface=previous_round.training_interface;fit_coefficients=previous_round.fit_coefficients
EFFECTS=['add_without_remove','add_after_remove','remove_without_add','remove_after_add','nonadditive','add_effect','remove_effect','total_delta']
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def source_hashes():
    r=previous_round.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.rglob('*.py')});return r
def input_hashes():
    r=previous_round.input_hashes();paths=[V34/'protocol.json']+[V34/'results'/n for n in ['delivery_manifest.json','preparation_manifest.json','verification.json','结果解读与下一步.md','endpoints.json','training_designs.json','policy_heads.json','model_predictions.csv','ensemble_predictions.csv','primary_comparisons.json','feature_banks.json']]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V34/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V34/'results/delivery_manifest.json')['files'].items():r[str((V34/n).relative_to(PROJECT))]=d
    p=V34/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==5063
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    r=read(OUT/'preparation_manifest.json');assert r['protocol_sha256']==sha(ROOT/'protocol.json') and r['source_sha256']==source_hashes() and r['input_sha256']==input_hashes()
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:
        c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and c['source_sha256']==r['source_sha256'] and c['protocol_sha256']==r['protocol_sha256']
        for n,d in c['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def baselines():return tuple(pd.read_csv(V34/'results'/n,float_precision='round_trip') for n in ['model_predictions.csv','ensemble_predictions.csv'])
def member_sets(obs,cutoff):
    a=training_rows(obs,annual_for(cutoff));q=training_rows(obs,cutoff)
    return {'annual':a,'add':np.union1d(a,q),'remove':np.intersect1d(a,q),'both':q}
def membership_tables():
    obs,_,targets=data();summaries=[];members=[];changes=[]
    for cutoff in cfg()['decision_dates']:
        sets=member_sets(obs,cutoff);a,q=sets['annual'],sets['both'];added=np.setdiff1d(q,a);removed=np.setdiff1d(a,q);common=np.intersect1d(a,q)
        r=dict(cutoff=cutoff,encoder_cutoff=annual_for(cutoff),annual_n=len(a),both_n=len(q),retained_n=len(common),added_n=len(added),removed_n=len(removed))
        for role,ids in [('annual',a),('both',q),('added',added),('removed',removed),('retained',common)]:r[role+'_up_fraction']=float((targets['returns'][ids]>0).mean()) if len(ids) else np.nan
        changes.append(r)
        for arm,ids in sets.items():
            summaries.append(dict(arm=arm,history=ARMS[arm],cutoff=cutoff,encoder_cutoff=annual_for(cutoff),train_n=len(ids),minimum_date=obs.date.iloc[ids].min(),maximum_date=obs.date.iloc[ids].max(),maximum_maturity=obs.joint_completed.iloc[ids].max(),up_fraction=float((targets['returns'][ids]>0).mean()),new_fit=arm in ['add','remove'] and cutoff!=annual_for(cutoff)))
            for i in ids:members.append(dict(arm=arm,cutoff=cutoff,row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i]))
    return pd.DataFrame(summaries),pd.DataFrame(members),pd.DataFrame(changes)
def effects(p0,pa,pr,pb):
    da=pa-p0;dar=pb-pr;dr=pr-p0;dra=pb-pa
    return dict(add_without_remove=da,add_after_remove=dar,remove_without_add=dr,remove_after_add=dra,nonadditive=pb-pa-pr+p0,add_effect=.5*(da+dar),remove_effect=.5*(dr+dra),total_delta=pb-p0)
def candidate_predictions(obs,heads,banks,route,base):
    parts=[];sources=[];_,price,targets=data()
    for arm in ['add','remove']:
        history=ARMS[arm]
        for cutoff in cfg()['decision_dates']:
            rows=route[route.head_cutoff.eq(cutoff)].row_index.to_numpy(int);annual=annual_for(cutoff)
            if cutoff==annual:
                g=base[base.history.eq(ARMS['annual'])&base.method.isin(LEARNED)&base.row_index.isin(rows)].copy();g['history']=history;parts.append(g)
            else:
                for seed in cfg()['seeds']:
                    hs=[next(h for h in heads if h['arm']==arm and h['cutoff']==cutoff and h['seed']==seed and h['method']==m) for m in LEARNED]
                    bank=arrays(next(b for b in banks if b['cutoff']==annual and b['seed']==seed));f=bank_slice(bank,rows);xx=apply_pipeline(f,market(price,obs,rows),gates(price,obs,rows),arrays(hs[0]))
                    for h,x in zip(hs,xx):
                        p=probability(design(x)@np.asarray(h['coefficients']));g=prior.rows_for_predictions(obs,rows,cutoff,h['method'],seed,p,p);g.insert(0,'history',history);parts.append(g)
            train=member_sets(obs,cutoff)[arm];frequency=float((targets['returns'][train]>0).mean());p=np.full(len(rows),frequency);g=prior.rows_for_predictions(obs,rows,cutoff,'training_frequency',-1,p,p);g.insert(0,'history',history);parts.append(g)
            for seed in cfg()['seeds']:sources.append(dict(history=history,arm=arm,head_cutoff=cutoff,encoder_cutoff=annual,transform_cutoff=annual,seed=seed,test_n=len(rows),train_n=len(train),head_source='annual_reuse' if cutoff==annual else 'member_ablation_refit'))
        g=base[base.history.eq(ARMS['annual'])&base.method.eq('native_mse')].copy();g['history']=history;parts.append(g)
    r=pd.concat(parts,ignore_index=True);assert len(r)==8704 and not r.duplicated(['history','method','seed','date']).any();return r,pd.DataFrame(sources)

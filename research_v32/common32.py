"""Freeze annual neural networks and update the downstream pipeline quarterly."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache';V31=PROJECT/'research_v31';V28=PROJECT/'research_v28'
sys.path.insert(0,str(V31));import common31 as previous_round
prior=previous_round.prior;np=prior.np;pd=prior.pd;torch=prior.torch;legacy=prior.legacy;neural=prior.neural;training=prior.training
read=prior.read;save=prior.save;sha=prior.sha;metric=prior.metric;data=prior.data;statistics=prior.statistics;arrays=prior.arrays;ref=prior.ref;market=prior.market;gates=prior.gates;load_model=prior.load_model
fit_pipeline=prior.fit_pipeline;apply_pipeline=prior.apply_pipeline;design=prior.design;probability=prior.probability
METHODS=prior.METHODS;LEARNED=prior.LEARNED;PRIMARY=prior.PRIMARY;NEW=['annual_net_quarter_head'];HISTORIES=previous_round.HISTORIES+NEW
training_rows=previous_round.training_rows;ensemble_from=previous_round.ensemble_from
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def source_hashes():
    r=previous_round.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.rglob('*.py')});return r
def input_hashes():
    r=previous_round.input_hashes();paths=[V31/'protocol.json']+[V31/'results'/n for n in ['delivery_manifest.json','preparation_manifest.json','verification.json','model_predictions.csv','ensemble_predictions.csv','budgets.csv']]+[V28/'results'/n for n in ['models.json','heads.json','testing_features.json','training_scales.json']]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V31/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V31/'results/delivery_manifest.json')['files'].items():r[str((V31/n).relative_to(PROJECT))]=d
    p=V31/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==4759
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json');assert p['protocol_sha256']==sha(ROOT/'protocol.json') and p['source_sha256']==source_hashes() and p['input_sha256']==input_hashes()
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:assert read(OUT/'contract_verification.json')['status']=='PASS'
    return p
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def baselines():return tuple(pd.read_csv(V31/'results'/n,float_precision='round_trip') for n in ['model_predictions.csv','ensemble_predictions.csv'])
def annual_for(cutoff):return cutoff if cutoff.endswith('12-31') else f'{int(cutoff[:4])-1}-12-31'
def canonical(obs):return np.flatnonzero(obs.date.ge('2021-01-01')&obs.weekday.eq(4)&obs.joint_completed.le(cfg()['label_end']))
def routing(obs):
    rows=[]
    for i in canonical(obs):
        date=obs.date.iloc[i];head=max(d for d in cfg()['decision_dates'] if d<date);encoder=max(d for d in cfg()['decision_dates'] if d<date and d.endswith('12-31'));rows.append(dict(row_index=int(i),date=date,head_cutoff=head,encoder_cutoff=encoder))
    return pd.DataFrame(rows)
def model_sources():return [r for r in read(V28/'results/models.json') if r['history']=='rolling5' and r['cutoff'].endswith('12-31')]
def head_sources():return [r for r in read(V28/'results/heads.json') if r['history']=='rolling5' and r['cutoff'].endswith('12-31')]
def test_sources():return [r for r in read(V28/'results/testing_features.json') if r['history']=='rolling5' and r['cutoff'].endswith('12-31')]
def memberships(obs):
    route=routing(obs);members=[];jobs=[]
    for cutoff in cfg()['decision_dates']:
        tr=training_rows(obs,cutoff);te=route[route.head_cutoff.eq(cutoff)].row_index.to_numpy(int);annual=annual_for(cutoff)
        jobs.append(dict(head_cutoff=cutoff,encoder_cutoff=annual,train_n=len(tr),test_n=len(te),refit=not cutoff.endswith('12-31')))
        for split,ids in [('training',tr),('testing',te)]:
            for i in ids:members.append(dict(head_cutoff=cutoff,encoder_cutoff=annual,split=split,row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i]))
    return pd.DataFrame(jobs),pd.DataFrame(members),route
def bank_memberships(obs,members):
    out=[];heads=head_sources();tests=test_sources()
    for annual in sorted({r['cutoff'] for r in heads}):
        h=next(r for r in heads if r['cutoff']==annual and r['seed']==cfg()['seeds'][0]);t=next(r for r in tests if r['cutoff']==annual and r['seed']==cfg()['seeds'][0]);tr=set(arrays(h)['row_index'].tolist());te=set(arrays(t)['row_index'].tolist());assert not tr&te
        need=set(members[members.encoder_cutoff.eq(annual)].row_index);ids=sorted(tr|te|need)
        for i in ids:out.append(dict(encoder_cutoff=annual,row_index=int(i),date=obs.date.iloc[i],source=0 if i in tr else 1 if i in te else 2))
    return pd.DataFrame(out)
def bank_slice(bank,rows):
    positions=np.searchsorted(bank['row_index'],rows);np.testing.assert_array_equal(bank['row_index'][positions],rows);return bank['features'][positions]
def training_interface(bank,rows,mf,u,y):
    # This is the complete data interface received by the convex fitting code.
    return bank_slice(bank,rows),mf[rows],u[rows],y[rows]
def candidate_predictions(obs,heads,banks,route,base):
    parts=[];provenance=[];_,price,_=data();src=head_sources()
    for cutoff in cfg()['decision_dates']:
        rows=route[route.head_cutoff.eq(cutoff)].row_index.to_numpy(int);annual=annual_for(cutoff)
        if cutoff.endswith('12-31'):
            g=base[base.history.eq('rolling5_annual20')&base.method.isin(LEARNED)&base.row_index.isin(rows)].copy();g['history']=NEW[0];parts.append(g)
        else:
            for seed in cfg()['seeds']:
                hs=[next(h for h in heads if h['cutoff']==cutoff and h['seed']==seed and h['method']==m) for m in LEARNED];bank=arrays(next(b for b in banks if b['cutoff']==annual and b['seed']==seed));f=bank_slice(bank,rows);xx=apply_pipeline(f,market(price,obs,rows),gates(price,obs,rows),arrays(hs[0]))
                for h,x in zip(hs,xx):
                    p=probability(design(x)@np.asarray(h['coefficients']));g=prior.rows_for_predictions(obs,rows,cutoff,h['method'],seed,p,p);g.insert(0,'history',NEW[0]);parts.append(g)
        for seed in cfg()['seeds']:provenance.append(dict(head_cutoff=cutoff,encoder_cutoff=annual,seed=seed,test_n=len(rows),head_source='annual_reuse' if cutoff.endswith('12-31') else 'quarterly_refit'))
    for method,history in [('native_mse','rolling5_annual20'),('training_frequency','rolling5_quarterly20')]:
        g=base[base.history.eq(history)&base.method.eq(method)].copy();g['history']=NEW[0];parts.append(g)
    r=pd.concat(parts,ignore_index=True);assert len(r)==4352 and not r.duplicated(['method','seed','date']).any();return r,pd.DataFrame(provenance)
def runtime_budgets():
    b=pd.read_csv(V31/'results/budgets.csv');a=b[b.history.eq('rolling5_annual20')].iloc[0].to_dict();a.update(history=NEW[0],refit_dates=23,head_fits_if_run_online=276,actual_new_neural_fits=0,actual_new_head_fits=204,selected_incumbent_dates=6);return pd.concat([b,pd.DataFrame([a])],ignore_index=True)

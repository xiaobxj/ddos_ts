from experiment3 import *
from slow3 import shrink_decision,cal_record,apply_record,metric,v2cal
from scipy.special import expit
import argparse

def preflight():
    dates=pd.bdate_range('2020-01-01',periods=1100).strftime('%Y-%m-%d').to_numpy();n=len(dates)-5
    p=np.tile([.2,.8],int(np.ceil(n/2)))[:n];y=(p<.5).astype(float)
    b=pd.DataFrame(dict(h=5,anchor=np.arange(n),exit=np.arange(n)+5,date=dates[:n],joint_completed=dates[5:],
        matured=True,probability=p,actual_up=y,actual_return=y-.5,state=0))
    c='2023-06-30';r=shrink_decision(b,c);assert r['candidate_alpha']==0.
    assert 0<=r['alpha']<=1
    altered=b.copy();mask=altered.joint_completed.gt(c);altered.loc[mask,['actual_up','probability']]=[0.,.999]
    assert shrink_decision(altered,c)==r
    for a in [0,.3,1]:
        spec=dict(scheme='shrink',accepted=True,q=.4,alpha=a);x=np.linspace(0,1,21);out=apply_record(x,spec)
        assert (np.diff(out)>=0).all() and (out>=np.minimum(x,.4)-1e-15).all() and (out<=np.maximum(x,.4)+1e-15).all()
    assert not shrink_decision(b,'2022-12-31')['accepted']
    save(OUT/'preflight.json',dict(status='PASS',completed_utc=now(),checks=['anti-ranking training yields zero candidate weight','future-label suffix invariant','no ranking inversion','probabilities inside raw/prior interval','Q1 reset']))
    save(OUT/'evaluation_freeze.json',dict(created_utc=now(),before_scoring=True,files={p:sha(ROOT/p) for p in ['slow3.py','verify3.py']}))

def models():
    check_freeze();r,_,_=modules(True);checks=[];gradients=[];f=data()
    for task in tasks():
        folder=fit_folder(task['cutoff']);a=interface(f,task);ids=a['ids'];obs=a['obs'];oos=np.flatnonzero(obs.date.gt(task['cutoff']))
        sample=np.r_[ids[np.linspace(0,len(ids)-1,8,dtype=int)],oos[np.linspace(0,len(oos)-1,8,dtype=int)]]
        values={k:r.torch.from_numpy(v[sample]).cuda() for k,v in a['values'].items()}
        pf=f[f.date.le(task['cutoff'])].reset_index(drop=True);pa=interface(pf,task)
        assert a['scales']==pa['scales']
        for k in a['values']:np.testing.assert_array_equal(a['values'][k][ids],pa['values'][k][pa['ids']])
        for seed in SEEDS:
            for epoch in [10,20]:
                m,s=load_model(folder,seed,epoch);head=read(folder/f'heads_{seed}_e{epoch}.json')
                assert sha(folder/f'model_{seed}_e{epoch}.pt')==head['checkpoint_sha256']
                assert sha(folder/f'cache_{seed}_e{epoch}.npz')==head['cache_sha256']
                with np.load(folder/f'cache_{seed}_e{epoch}.npz') as z:d={k:z[k] for k in z.files}
                features,native=r.neural.extract_features(m,values);gap=float(abs(features-d['all_features'][sample]).max());assert gap<1e-4
                xs=r.apply_pipeline(features,a['market'][sample],a['gate'][sample],d);allxs=r.apply_pipeline(d['all_features'],a['market'],a['gate'],d)
                maxp=0.
                for method,x,xall in zip(METHODS,xs,allxs):
                    theta=np.array(next(x['coefficients'] for x in head['heads'] if x['method']==method and x['scheme']=='U'))
                    p=expit(np.c_[x,np.ones(len(x))]@theta);pgap=float(abs(p-d['p__'+method+'.U'][sample]).max());maxp=max(maxp,pgap);assert pgap<5e-5
                    z=np.c_[xall[ids],np.ones(len(ids))];error=expit(z@theta)-a['y']
                    if method=='order_fixed':grad=abs(float(error@xall[ids,-1]/len(ids)+.01*theta[-2]))
                    else:grad=float(abs(z.T@error/len(ids)+.01*np.r_[theta[:-1],0.]).max())
                    assert grad<3e-9;gradients.append(grad)
                checks.append(dict(task=task_id(task),seed=seed,epoch=epoch,feature_delta=gap,probability_delta=maxp))
                del m;r.torch.cuda.empty_cache()
        print('VERIFIED '+task_id(task),flush=True)
    csv(OUT/'model_checks.csv',pd.DataFrame(checks));save(OUT/'models_verified.json',dict(status='PASS',completed_utc=now(),checkpoints=len(checks),heads=len(gradients),max_gradient=max(gradients),max_p_delta=max(c['probability_delta'] for c in checks)))

def outputs():
    check_freeze();p=load_csv(OUT/'slow_predictions.csv');raw=load_csv(OUT/'slow_raw_banks.csv');records=read(OUT/'calibration_records.json');count=0
    for row in records:
        bank=raw[raw.cadence.eq(row['cadence'])&raw.method.eq(row['method'])].sort_values('date').reset_index(drop=True);c=row['cutoff']
        spec=cal_record(bank,c,row['scheme']);prefix=bank[bank.date.le(c)].copy();notdone=prefix.joint_completed.isna()|prefix.joint_completed.gt(c)
        prefix.loc[notdone,'matured']=False;prefix.loc[notdone,'actual_up']=np.nan
        ps=cal_record(prefix,c,row['scheme']);assert spec==ps
        if row['scheme']=='shrink':
            assert 0<=spec['alpha']<=1 and 0<=spec['candidate_alpha']<=1
            tr,va,ready=v2cal.split_bank(bank,c)
            if ready and c[5:7]!='12':
                q=tr.actual_up.mean();d=tr.probability.to_numpy()-q;a=spec['candidate_alpha'];gradient=float(2*np.mean(d*(q+a*d-tr.actual_up.to_numpy())))
                assert abs(gradient)<1e-10 or (a==0 and gradient>=0) or (a==1 and gradient<=0)
            if spec['accepted']:assert spec['validation_brier_delta']<-1e-12 and spec['validation_correct_delta']>=0
        count+=1
    oldp=load_csv(V2/'results/predictions_h5.csv')
    for newname,oldname in [('annual.vol.raw','mse_5y_e20.vol.U'),('annual.vol.platt','cal.quarter_platt')]:
        a=p[p.name.eq(newname)].sort_values('date');b=oldp[oldp.method.eq(oldname)].sort_values('date')
        assert a.date.tolist()==b.date.tolist();np.testing.assert_allclose(a.probability,b.probability,rtol=0,atol=2e-15)
    for name,g in p[p.matured].groupby('name'):assert len(g)==893 and g.date.is_unique
    assert not p[p.date.eq(cfg()['data_end'])].matured.any()
    for path,s in read(OUT/'preserved_files.json').items():assert sha(PROJECT/path)==s,path
    save(OUT/'outputs_verified.json',dict(status='PASS',completed_utc=now(),calibration_prefix_checks=count,
        old_annual_and_platt_replay='exact within 2e-15',common_n=893,old_files_preserved=len(read(OUT/'preserved_files.json'))))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['preflight','models','outputs']);args=ap.parse_args()
    globals()[args.phase]()

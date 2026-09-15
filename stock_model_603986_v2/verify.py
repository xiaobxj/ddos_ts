"""Independent numerical, temporal and artifact verification of H1/H5 fits."""
from core import *
from scipy.special import expit,logit
from scipy.optimize import brentq
from calibration import decision,split_bank,calibrate,POLICIES
import argparse

def verify_models():
    check_freeze();r,w,_=modules(True);f=data();checks=[];headchecks=[];prefixchecks=[]
    for h in [1,5]:
        for task in tasks(h):
            folder=OUT/'fits'/task_id(task);done=read(folder/'completed.json')
            for p,s in done['files'].items():assert sha(folder/p)==s,p
            a=interface(f,task);ids=a['ids'];obs=a['obs'];cutoff=task['cutoff']
            prefix=f[f.date.le(cutoff)].reset_index(drop=True);pa=interface(prefix,task)
            pd.testing.assert_frame_equal(obs.iloc[ids].reset_index(drop=True),pa['obs'].iloc[pa['ids']].reset_index(drop=True),check_exact=True)
            for k in a['labels']:np.testing.assert_array_equal(a['labels'][k],pa['labels'][k])
            assert a['scales']==pa['scales']
            np.testing.assert_array_equal(a['market'][ids],pa['market'][pa['ids']])
            np.testing.assert_array_equal(a['gate'][ids],pa['gate'][pa['ids']])
            for k in a['values']:np.testing.assert_array_equal(a['values'][k][ids],pa['values'][k][pa['ids']])
            # Explicitly mutate the future suffix; pre-cutoff inputs/labels cannot change.
            mutated=f.copy();later=mutated.date.gt(cutoff)
            mutated.loc[later,['open','high','low','close']]*=3
            mutated.loc[later,'volume']*=2
            mo,mt=observations(mutated,h);mi,_=membership(mo,cutoff,task['years'])
            np.testing.assert_array_equal(mi,ids)
            for k in a['targets']:np.testing.assert_array_equal(a['targets'][k][ids],mt[k][ids])
            prefixchecks.append(dict(task=task_id(task),train_n=len(ids),status='PASS',last_mature_label=obs.joint_completed.iloc[ids].max()))
            oos=np.flatnonzero(obs.date.gt(cutoff));selected=np.unique(np.r_[ids[np.linspace(0,len(ids)-1,8,dtype=int)],oos[np.linspace(0,len(oos)-1,8,dtype=int)]])
            values={k:r.torch.from_numpy(v[selected]).cuda() for k,v in a['values'].items()}
            dates=obs.date.iloc[ids];age=(pd.Timestamp(cutoff)-pd.to_datetime(dates)).dt.days.to_numpy(float)
            ww=np.exp2(-age/730.5);ww/=ww.sum()
            for seed in SEEDS:
                previous_model=None
                for epoch in [10,20]:
                    model,state=model_load(folder/f'model_{seed}_e{epoch}.pt')
                    head=read(folder/f'heads_{seed}_e{epoch}.json')
                    assert sha(folder/f'model_{seed}_e{epoch}.pt')==head['checkpoint_sha256']
                    assert sha(folder/f'cache_{seed}_e{epoch}.npz')==head['cache_sha256']
                    with np.load(folder/f'cache_{seed}_e{epoch}.npz') as z:d={k:z[k] for k in z.files}
                    feature,native=r.neural.extract_features(model,values)
                    gap=float(abs(feature-d['all_features'][selected]).max());ngap=float(abs(native-d['all_native'][selected]).max())
                    # Batch dispatch differences are bounded at probability precision, not exact tensor equality.
                    assert gap<1e-4 and ngap<1e-4,(task,seed,epoch,gap,ngap)
                    xs=r.apply_pipeline(feature,a['market'][selected],a['gate'][selected],d)
                    fullxs=r.apply_pipeline(d['all_features'],a['market'],a['gate'],d)
                    maxp=0.;by={(x['method'],x['scheme']):np.asarray(x['coefficients']) for x in head['heads']}
                    for method,x,fullx in zip(METHODS,xs,fullxs):
                        u=by[method,'U'];weighted=by[method,'W'];i=by[method,'I'];s=by[method,'S']
                        np.testing.assert_array_equal(i[:-1],u[:-1]);assert i[-1]==weighted[-1]
                        np.testing.assert_array_equal(s[:-1],weighted[:-1]);assert s[-1]==u[-1]
                        for scheme,theta in [('U',u),('W',weighted),('I',i),('S',s)]:
                            pred=expit(np.c_[x,np.ones(len(x))]@theta)
                            pgap=float(abs(pred-d['p__'+method+'.'+scheme][selected]).max());maxp=max(maxp,pgap)
                            assert pgap<5e-5,(task,seed,epoch,method,scheme,pgap)
                            if scheme not in ['U','W']:continue
                            xtr=fullx[ids];z=np.c_[xtr,np.ones(len(ids))];weights=ww if scheme=='W' else np.full(len(ids),1/len(ids))
                            pp=expit(z@theta);err=pp-a['y']
                            if method=='order_fixed':
                                grad=float((weights*err)@xtr[:,-1]+.01*theta[-2]);curvature=float((weights*pp*(1-pp))@(xtr[:,-1]**2)+.01)
                                assert abs(grad)<3e-9 and curvature>0
                                parent=by['vol',scheme];np.testing.assert_array_equal(theta[:-2],parent[:-1]);assert theta[-1]==parent[-1]
                            else:
                                gradient=z.T@(weights*err)+.01*np.r_[theta[:-1],0.]
                                grad=float(abs(gradient).max());curvature=float(np.linalg.eigvalsh(z.T@((weights*pp*(1-pp))[:,None]*z)+.01*np.diag(np.r_[np.ones(len(theta)-1),0])).min())
                                assert grad<3e-9 and curvature>0,(task,seed,epoch,method,scheme,grad)
                            headchecks.append(dict(task=task_id(task),seed=seed,epoch=epoch,method=method,scheme=scheme,gradient_inf=abs(grad),curvature_min=curvature))
                    for prefix in ['vol','order']:
                        assert float(d[prefix+'__training_orthogonality_inf'])<1e-8
                        assert float(d[prefix+'__residual_raw_sd'])>1e-6
                        assert int(d[prefix+'__augmented_design_rank'])==int(d[prefix+'__base_design_rank'])+1
                    if previous_model:assert state['model_sha256']!=previous_model,'No parameter update between epochs'
                    previous_model=state['model_sha256']
                    # Prediction prefix replay for the latest selected OOS date, using only that day's history.
                    anchor=int(obs.anchor.iloc[selected[-1]]);short=f.iloc[:anchor+1].reset_index(drop=True)
                    so,_=observations(short,h);sl=so.index[so.date.eq(obs.date.iloc[selected[-1]])].to_numpy()
                    sv=old.packed(short,so.anchor.iloc[sl].to_numpy(int))
                    sf,sn=r.neural.extract_features(model,{k:r.torch.from_numpy(v).cuda() for k,v in sv.items()})
                    sm=r.market(short,so,sl);sg=r.gates(short,so,sl);sx=r.apply_pipeline(sf,sm,sg,d)
                    t=by['vol','U'];pp=float(expit(np.c_[sx[1],np.ones(1)]@t)[0]);pg=abs(pp-float(d['p__vol.U'][selected[-1]]))
                    assert pg<5e-5,(task,seed,epoch,pg)
                    checks.append(dict(task=task_id(task),seed=seed,epoch=epoch,feature_max_delta=gap,native_max_delta=ngap,
                        probability_max_delta=maxp,truncated_prediction_probability_delta=pg,status='PASS'))
                    del model;r.torch.cuda.empty_cache()
            print('VERIFIED '+task_id(task),flush=True)
    csv(OUT/'checkpoint_checks.csv',pd.DataFrame(checks));csv(OUT/'head_gradient_checks.csv',pd.DataFrame(headchecks))
    save(OUT/'causal_training_prefix_checks.json',prefixchecks)
    save(OUT/'models_verified.json',dict(status='PASS',completed_utc=now(),checkpoints=len(checks),heads=len(headchecks),
        max_probability_replay_delta=max(x['probability_max_delta'] for x in checks),
        max_head_gradient=max(x['gradient_inf'] for x in headchecks)))

def verify_outputs():
    check_freeze();assert read(OUT/'scoring_completed.json')['status']=='PASS'
    checks=[]
    for h in [1,5]:
        predictions=load_csv(OUT/f'predictions_h{h}.csv');obs=load_csv(OUT/f'context_h{h}.csv')
        assert predictions.probability.between(0,1).all() and predictions.probability.notna().all()
        assert not predictions[predictions.date.eq(cfg()['data_end'])].matured.any()
        scored=predictions[predictions.matured];expected=obs[obs.matured].date.tolist()
        for name,g in scored.groupby('method'):assert sorted(g.date)==expected and g.date.is_unique,(h,name)
        bank=predictions[predictions.method.eq(cfg()['primary'])].sort_values('date').reset_index(drop=True)
        stored=read(OUT/f'calibration_decisions_h{h}.json')
        for policy,cutoff in sorted(set((row['policy'],row['cutoff']) for row in stored)):
            params,rows=decision(bank,cutoff,policy)
            prefix=bank[bank.date.le(cutoff)].copy()
            # Remove all not-yet-mature labels, including an anchor on/before cutoff.
            immature=prefix.joint_completed.isna()|prefix.joint_completed.gt(cutoff)
            prefix.loc[immature,['actual_up','actual_return']]=np.nan;prefix.loc[immature,'matured']=False
            pp,rr=decision(prefix,cutoff,policy);assert params==pp and rows==rr,(h,policy,cutoff)
            tr,va,ready=split_bank(bank,cutoff)
            if ready:assert tr.joint_completed.max()<va.date.min() and va.joint_completed.max()<=cutoff
            for row in rows:
                if row['accepted']:
                    assert row['train_disjoint']>=10
                    if policy!='quarter_ungated':assert row['validation_disjoint']>=5 and row['validation_brier_delta']<-1e-12 and row['validation_correct_delta']>=0
                t=tr if row['state']==-1 else tr[tr.state.eq(row['state'])]
                if row['ready'] and row['train_disjoint']>=10 and row['reason']!='annual_reset' and policy!='quarter_platt':
                    z=logit(t.probability.to_numpy());delta=row['delta'];gradient=float((expit(z+delta)-t.actual_up.to_numpy()).sum()+20*delta)
                    assert abs(gradient)<1e-8 or (delta==-.5 and gradient>=0) or (delta==.5 and gradient<=0)
            checks.append(dict(h=h,policy=policy,cutoff=cutoff,status='PASS'))
        # End-to-end calibration replay; no read of future outcome beyond each fixed cutoff.
        cal,_,_=calibrate(bank)
        for policy in POLICIES:
            a=cal[cal.method.eq('cal.'+policy)].sort_values('date')
            b=predictions[predictions.method.eq('cal.'+policy)].sort_values('date')
            np.testing.assert_allclose(a.probability,b.probability,rtol=0,atol=2e-15)
    oldfiles=read(OUT/'preserved_files.json')
    for p,s in oldfiles.items():assert sha(PROJECT/p)==s,('Prior evidence changed',p)
    save(OUT/'outputs_verified.json',dict(status='PASS',completed_utc=now(),calibration_prefix_decisions=len(checks),
        preserved_files=len(oldfiles),index_predictions_added=0,index_labels_added=0,stock_v1_modified=False,checks=checks))

def main():
    for name,digest in read(OUT/'evaluation_implementation_freeze.json')['files'].items():assert sha(ROOT/name)==digest,name
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['models','outputs']);a=ap.parse_args()
    verify_models() if a.phase=='models' else verify_outputs()

if __name__=='__main__':main()

from common35 import *
from contract35 import ablation_checks,synthetic_cases,base_memberships_check,boundary_cases,independent_route
from inference35 import verify_inference
import evaluate35 as evaluation
import diagnose35 as diagnosis
import verify15 as independent
from verify28 import independent_inputs
from scipy.optimize import minimize
from scipy.special import expit
import math

def verify_effects():
    seed=csv('seed_effects');weekly=csv('weekly_effects');coef=csv('coefficient_effects');models=csv('model_predictions');ensemble=csv('ensemble_predictions')
    assert len(seed)==3264 and len(weekly)==1088 and len(coef)==6375;mi=models[models.method.isin(LEARNED)].set_index(['history','method','seed','date']);ei=ensemble.set_index(['history','method','date']);maxgap=0.
    for r in seed.itertuples():
        ps=[float(mi.loc[(ARMS[a],r.method,r.seed,r.date),'probability']) for a in ARMS]
        for a,p in zip(ARMS,ps):assert p==getattr(r,'p_'+a)
        p0,pa,pr,pb=ps;add=(pb-p0+pa-pr)/2;remove=(pb-p0+pr-pa)/2
        vals=[pa-p0,pb-pr,pr-p0,pb-pa,math.fsum([pb,-pa,-pr,p0]),add,remove,pb-p0]
        for n,value in zip(EFFECTS,vals):maxgap=max(maxgap,abs(value-getattr(r,n)));assert abs(value-getattr(r,n))<1e-14
    for r in weekly.itertuples():
        g=seed[seed.method.eq(r.method)&seed.date.eq(r.date)];assert len(g)==3
        for arm in ARMS:
            q=math.fsum(g['p_'+arm])/3;assert abs(q-getattr(r,'p_'+arm))<1e-14 and abs(q-ei.loc[(ARMS[arm],r.method,r.date),'probability'])<1e-14
        for name in EFFECTS:
            q=math.fsum(g[name])/3;assert abs(q-getattr(r,name))<1e-14 and abs((2*r.actual_up-1)*q-getattr(r,'signed_'+name))<1e-14
        assert abs(r.add_effect+r.remove_effect-(r.p_both-r.p_annual))<1e-14
        a=int(r.p_annual>.5)==r.actual_up;b=int(r.p_both>.5)==r.actual_up;case='regression' if a and not b else 'recovery' if b and not a else 'stable_correct' if a else 'stable_wrong';assert case==r.case
        sa=(2*r.actual_up-1)*r.add_effect;sr=(2*r.actual_up-1)*r.remove_effect;dominant='none' if min(sa,sr)>=-1e-12 else 'tie' if abs(sa-sr)<=1e-12 else 'add' if sa<sr else 'remove';assert dominant==r.dominant_adverse
        if r.head_cutoff==r.encoder_cutoff:assert all(getattr(r,n)==0 for n in EFFECTS)
    for r in coef.itertuples():
        a,b,c,d=r.theta_annual,r.theta_add,r.theta_remove,r.theta_both
        expected=[b-a,d-c,c-a,d-b,math.fsum([d,-b,-c,a]),(d-a+b-c)/2,(d-a+c-b)/2,d-a]
        for n,v in zip(EFFECTS,expected):assert abs(v-getattr(r,n))<1e-14
    for r in csv('coefficient_effect_summary').itertuples():
        g=coef[coef.cutoff.eq(r.cutoff)&coef.seed.eq(r.seed)&coef.method.eq(r.method)].sort_values('coordinate');assert len(g)==r.dimensions and g.is_intercept.sum()==1 and g.is_intercept.iloc[-1]
        for n in EFFECTS:
            assert abs(math.sqrt(math.fsum(float(v*v) for v in g[n]))-getattr(r,n+'_norm'))<1e-14
            assert abs(g[n].iloc[-1]-getattr(r,n+'_intercept'))<1e-14
    for r in csv('effect_summary').itertuples():
        if r.period.startswith('year_'):g=weekly[weekly.year.eq(int(r.period[-4:]))]
        elif r.period=='pooled_2021_2026':g=weekly
        else:w=next(w for w in cfg()['windows'] if w['name']==r.period);g=weekly[weekly.date.between(w['start'],w['end'])]
        g=g[g.method.eq(r.method)];g=g if r.case=='all' else g[g['case'].eq(r.case)];assert len(g)==r.n
        for n in EFFECTS:
            assert abs(math.fsum(g[n])/len(g)-getattr(r,'mean_'+n))<1e-14
            assert abs(math.fsum(g['signed_'+n])/len(g)-getattr(r,'mean_signed_'+n))<1e-14
        assert r.annual_correct==int(g.p_annual.gt(.5).eq(g.actual_up).sum()) and r.both_correct==int(g.p_both.gt(.5).eq(g.actual_up).sum())
        for n in ['add','remove','tie','none']:assert getattr(r,'dominant_'+n)==int(g.dominant_adverse.eq(n).sum())
    # Rebuild serialized joins/classifications after the independent arithmetic above.
    s=diagnosis.unlabelled_effects(models);w=diagnosis.weekly_from(s,ensemble,csv('routing'));c,cs=diagnosis.coefficient_effects();f=diagnosis.all_flips(w)
    for n,g in [('seed_effects',s),('weekly_effects',w),('effect_summary',diagnosis.summaries(w)),('coefficient_effects',c),('coefficient_effect_summary',cs),('direction_flips',f),('all_2026_cases',w[w.year.eq(2026)]),('flips_2026',f[f.year.eq(2026)])]:pd.testing.assert_frame_equal(g.reset_index(drop=True),csv(n).reset_index(drop=True),check_dtype=False,atol=1e-14,rtol=0)
    return maxgap

def main():
    legacy.initialize();started=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists();last=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['fitting','scoring','evaluation','diagnosis']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for k in ['protocol_sha256','source_sha256','input_sha256']:assert r[k]==prep[k]
    base_memberships_check();checks=ablation_checks();assert len(boundary_cases()+synthetic_cases())==15
    obs,price,targets=data();ids=np.arange(len(obs));mf=market(price,obs,ids);u=gates(price,obs,ids);y=(targets['returns']>0).astype(float)
    banks=read(OUT/'feature_banks.json');annuals=read(OUT/'source_heads.json');heads=read(OUT/'heads.json')+[dict(arm='both',history=ARMS['both'],**h) for h in read(OUT/'both_heads.json')]
    trrefs=read(OUT/'training_designs.json')+[dict(arm='both',**h) for h in read(OUT/'both_training_designs.json')];models=csv('model_predictions');ensemble=csv('ensemble_predictions');base,be=baselines();route=csv('routing');solutions=[];interfaces=[];maxfeature=maxpred=maxanchor=0.
    for trref in trrefs:
        cutoff=trref['cutoff'];annual=trref['encoder_cutoff'];seed=trref['seed'];arm=trref['arm'];rows=member_sets(obs,cutoff)[arm]
        ah=[next(h for h in annuals if h['cutoff']==annual and h['seed']==seed and h['method']==m) for m in LEARNED];hs=[next(h for h in heads if h['arm']==arm and h['cutoff']==cutoff and h['seed']==seed and h['method']==m) for m in LEARNED]
        bank=arrays(next(b for b in banks if b['cutoff']==annual and b['seed']==seed));d=arrays(ah[0]);saved=arrays(trref);f=bank_slice(bank,rows);xx=independent_inputs(f,mf[rows],u[rows],d);labels=y[rows]
        np.testing.assert_array_equal(saved['row_index'],rows);np.testing.assert_array_equal(saved['direction'],labels)
        for x,key in zip(xx,['x18','x19','x23','x23']):gap=float(abs(x-saved[key]).max());maxfeature=max(maxfeature,gap);assert gap<1e-9
        badbank={k:v.copy() for k,v in bank.items()};outside=~np.isin(bank['row_index'],rows);badbank['features'][outside]=123456.;out=~np.isin(ids,rows);bad_m=mf.copy();bad_m[out]=-98765.;bad_u=u.copy();bad_u[out]=42.;bad_y=y.copy();bad_y[out]=1-bad_y[out]
        normal_x,normal_y=training_interface(bank,rows,mf,u,y,d);poison_x,poison_y=training_interface(badbank,rows,bad_m,bad_u,bad_y,d)
        for a,b in zip(normal_x,poison_x):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(normal_y,poison_y);interfaces.append(dict(arm=arm,cutoff=cutoff,encoder_cutoff=annual,seed=seed,n=len(rows),excluded_bank_rows=int(outside.sum()),excluded_labels=int(out.sum()),all_inputs_unchanged=True))
        ts=[np.asarray(h['coefficients']) for h in hs];ta=[np.asarray(h['coefficients']) for h in ah];np.testing.assert_array_equal(np.r_[ts[3][:-2],ts[3][-1]],ts[1])
        te=route[route.head_cutoff.eq(cutoff)].row_index.to_numpy(int);tx=independent_inputs(bank_slice(bank,te),mf[te],u[te],d)
        for j,(h,x,t,testx,annual_t) in enumerate(zip(hs,xx,ts,tx,ta)):
            assert h['cache_file']==ah[j]['cache_file'] and h['cache_sha256']==ah[j]['cache_sha256'] and h['transform_cutoff']==annual and h['train_n']==len(rows)
            if j<3:
                initial=np.zeros(x.shape[1]+1);rate=labels.mean();initial[-1]=np.log(rate/(1-rate));alt=minimize(lambda t:independent.independent_objective(t,x,labels,.01),initial,jac=True,method='L-BFGS-B',options=dict(ftol=1e-15,gtol=1e-11,maxiter=2000,maxls=40));val,gr=independent.independent_objective(alt.x,x,labels,.01)
                gap=float(abs(expit(x@alt.x[:-1]+alt.x[-1])-expit(x@t[:-1]+t[-1])).max());objgap=abs(val-h['objective']);gn=float(abs(gr).max());assert objgap<=1e-10 and gap<=1e-5 and gn<=5e-7
                pv,pg=independent.independent_objective(t,x,labels,.01);assert abs(pv-h['objective'])<1e-12 and abs(pg).max()<2e-9
                solutions.append(dict(arm=arm,job=h['job'],solver='L-BFGS-B',solver_success=bool(alt.success),iterations=int(alt.nit),probability_gap=gap,objective_gap=objgap,gradient_inf=gn))
            else:
                offset=xx[1]@ts[1][:-1]+ts[1][-1];alt=prior.previous.independent_root(offset,x[:,-1],labels,cfg()['offset_probe']);gap=abs(alt['gamma']-t[-2]);assert gap<=1e-7 and abs(alt['objective']-h['reduced_objective'])<=1e-10
                solutions.append(dict(arm=arm,job=h['job'],solver='brentq',solver_success=alt['converged'],iterations=alt['iterations'],coefficient_gap=gap,objective_gap=abs(alt['objective']-h['reduced_objective']),gradient_inf=abs(alt['gradient'])))
            z=np.array([math.fsum(float(v*w) for v,w in zip(np.r_[row,1.],t)) for row in testx]);p=expit(z);g=models[models.history.eq(ARMS[arm])&models.cutoff.eq(cutoff)&models.method.eq(h['method'])&models.seed.eq(seed)].sort_values('row_index');np.testing.assert_array_equal(g.row_index,te);gap=float(abs(p-g.probability.to_numpy()).max());maxpred=max(maxpred,gap);assert gap<1e-10;np.testing.assert_array_equal(g.direction_up,(p>.5).astype(int))
            oldp=expit(testx@annual_t[:-1]+annual_t[-1]);ag=base[base.history.eq(ARMS['annual'])&base.method.eq(h['method'])&base.seed.eq(seed)&base.row_index.isin(te)].sort_values('row_index');gap=float(abs(oldp-ag.probability.to_numpy()).max());maxanchor=max(maxanchor,gap);assert gap<1e-10
        if len(solutions)%24==0 or len(solutions)==612:print(f'Verified {len(solutions)}/612 new/reused coefficient fits and causal interfaces.',flush=True)
    assert len(solutions)==612 and len(interfaces)==153;assert {r['arm'] for r in solutions}=={'add','remove','both'}
    pd.testing.assert_frame_equal(models[~models.history.isin(NEW)].reset_index(drop=True),base,check_dtype=False,check_exact=True);pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),be,check_dtype=False,atol=1e-14,rtol=0)
    key=['method','seed','date'];q1=route[route.head_cutoff.str.endswith('12-31')].row_index
    for arm in ['add','remove']:
        g=models[models.history.eq(ARMS[arm])]
        a=g[g.method.eq('native_mse')].drop(columns='history').sort_values(key).reset_index(drop=True);b=base[base.history.eq(ARMS['annual'])&base.method.eq('native_mse')].drop(columns='history').sort_values(key).reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_exact=True)
        a=g[g.row_index.isin(q1)].drop(columns='history').sort_values(key).reset_index(drop=True);b=base[base.history.eq(ARMS['annual'])&base.row_index.isin(q1)].drop(columns='history').sort_values(key).reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_exact=True)
        for r in g.itertuples():
            annual,head=independent_route(r.date);assert r.cutoff==(annual if r.method=='native_mse' else head)
            if r.method=='training_frequency':assert abs(r.probability-y[member_sets(obs,head)[arm]].mean())<1e-14
    np.testing.assert_allclose(models.actual,targets['returns'][models.row_index],rtol=0,atol=1e-14);np.testing.assert_array_equal(models.actual_up,(models.actual>0).astype(int));indexed=ensemble.set_index(['history','method','date'])
    for key,g in models.groupby(['history','method','date']):
        r=indexed.loc[key];assert len(g)==(1 if key[1]=='training_frequency' else 3);score=math.fsum(g.score)/len(g);assert abs(score-r.score)<1e-14
        if key[1]=='native_mse':assert pd.isna(r.probability) and int(score>0)==r.direction_up
        else:p=math.fsum(g.probability)/len(g);assert abs(p-r.probability)<1e-14 and int(p>.5)==r.direction_up
    actual,sources=candidate_predictions(obs,read(OUT/'heads.json'),banks,route,base);pd.testing.assert_frame_equal(actual,models[models.history.isin(NEW)].reset_index(drop=True),check_dtype=False,check_exact=True);pd.testing.assert_frame_equal(sources,csv('prediction_sources'),check_dtype=False)
    tables,pairs=evaluation.compute(models,ensemble)
    for n,t in tables.items():pd.testing.assert_frame_equal(t,csv(n),check_dtype=False,atol=1e-12,rtol=0)
    independent.close_tree(pairs,read(OUT/'primary_comparisons.json'));assert len(pairs)==60;verify_inference(ensemble,pairs)
    independent.verify_metric_table(ensemble,csv('yearly_metrics'),['history','method','year']);independent.verify_metric_table(models[models.seed.ne(-1)],csv('seed_yearly_metrics'),['history','method','seed','year'])
    for w in evaluation.windows():
        independent.verify_metric_table(evaluation.select(ensemble,w),csv('ensemble_metrics')[csv('ensemble_metrics').window.eq(w['name'])],['history','method']);independent.verify_metric_table(evaluation.select(models[models.seed.ne(-1)],w),csv('seed_metrics')[csv('seed_metrics').window.eq(w['name'])],['history','method','seed'])
    effectgap=verify_effects();assert len(models)==56576 and len(ensemble)==21216;assert old_evidence()==prep['old_evidence'];check_frozen();files=[]
    for n,v in [('independent_solutions',solutions),('training_interface_checks',interfaces)]:p=OUT/f'{n}.csv';pd.DataFrame(v).to_csv(p,index=False);files.append(p)
    save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-started,protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=5063,new_neural_fits=0,new_feature_inference=0,new_transform_fits=0,independent_endpoint_solutions=612,new_head_fits=408,causal_training_interfaces=153,ablation_membership_checks=len(checks),maximum_probability_gap=maxpred,maximum_annual_anchor_probability_gap=maxanchor,maximum_feature_algebra_gap=maxfeature,maximum_effect_identity_gap=effectgap,all_q1_predictions_exact=True,native_exactly_annual=True,frequency_uses_own_members=True,all60comparisons_recomputed=True,independent_bootstrap_and_holm=True,independent_metrics=True,coefficient_and_probability_accounting=True,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));print('Verification PASS: four member arms, all fits, causal inputs, metrics and operation effects.',flush=True)
if __name__=='__main__':main()

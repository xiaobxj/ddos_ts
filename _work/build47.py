from pathlib import Path
import json
P=Path('D:/ddos_v3');R=P/'research_v47';assert not R.exists();R.mkdir()
old=P/'research_v46'
def put(n,s):(R/n).write_text(s,encoding='utf-8')
prev=lambda n:(old/(n+'46.py')).read_text(encoding='utf-8')
s=prev('common');a=s.index('def read(');b=s.index('def component(');common=s[a:b]
common=common.replace('5986','6056')
a=common.index('def input_hashes():');b=common.index('def old_evidence():')
common=common[:a]+'''def source_inventory():
    heads=[h for h in read(V28/'results/heads.json') if h['history']=='rolling5' and h['cutoff'].endswith('12-31')]
    models=[m for m in read(V28/'results/models.json') if m['history']=='rolling5' and m['cutoff'].endswith('12-31')]
    tests=[t for t in read(V28/'results/testing_features.json') if t['history']=='rolling5' and t['cutoff'].endswith('12-31')]
    assert len(heads)==72 and len(models)==18 and len(tests)==18
    return heads,models,tests
def input_hashes():
    heads,models,tests=source_inventory()
    paths=[a for a,b in COPIES]+[PREV/'protocol.json',V28/'protocol.json']
    paths += [PREV/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','verification.json','结果解读与下一步.md']]
    paths += [V28/'results'/n for n in ['models.json','heads.json','testing_features.json','verification.json','training_manifest.json','scoring_manifest.json']]
    paths += [PROJECT/h['cache_file'] for h in heads]+[PROJECT/t['cache_file'] for t in tests]+[PROJECT/m['project_file'] for m in models]
    paths += [PROJECT/'research_v5/results/observation_table.csv',PROJECT/'research/data/1_000300.csv',PROJECT/'research_v5/cache/targets.npz']
    return {str(p.relative_to(PROJECT)):sha(p) for p in paths}
'''+common[b:]
base='''"""Annual time-weighted convex heads in frozen neural and interaction coordinates."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from solver47 import probability,design,weights_for,objective,offset_objective,fit_newton,fit_offset
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache';PREV=PROJECT/'research_v46';V28=PROJECT/'research_v28'
METHODS=['learned_market','learned_vol_interaction','learned_order_extension','learned_order_offset'];PRIMARY=METHODS[1:]
STATES=['negative_low','negative_high','nonnegative_low','nonnegative_high']
ANNUAL='rolling5_annual20';QUARTER='weekly_state_validated';NEW=['annual_head_timeweight2y'];REPORT_HIST=[ANNUAL,QUARTER]+NEW
COPIES=[(PREV/'results'/f'{a}.csv',f'{b}.csv') for a,b in [('model_predictions','baseline_model_predictions'),('ensemble_predictions','baseline_ensemble_predictions'),('ensemble_metrics','baseline_metrics'),('weekly_context','weekly_context')]]+[(PROJECT/'research_v5/results/observation_table.csv','observation_table.csv')]
'''
tail=prev('common')[prev('common').index('def periods():'):]
put('common47.py',base+common+'''
def arrays(ref):
    path=PROJECT/ref['cache_file'];assert sha(path)==ref['cache_sha256']
    with np.load(path) as d:return {k:d[k].copy() for k in d.files}
def training_rows(obs,cutoff):
    lower=(pd.Timestamp(cutoff)-pd.DateOffset(years=5)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    return np.flatnonzero(obs.date.ge(lower)&obs.joint_completed.le(cutoff))
def training_interface(obs,d,cutoff):
    ids=training_rows(obs,cutoff);positions=pd.Index(d['row_index']).get_indexer(ids);assert (positions>=0).all()
    y=obs.exec_return.iloc[ids].gt(0).to_numpy(float);raw,w,age=weights_for(obs.date.iloc[ids],cutoff)
    return ids,[d[k][positions].copy() for k in ['x18','x19','x23']],y,raw,w,age
def prediction_rows(obs,ids,cutoff,method,seed,p):
    r=obs.iloc[ids][['date','joint_completed','exec_return']].copy().rename(columns={'exec_return':'actual'})
    r.insert(0,'row_index',ids);r.insert(0,'history',NEW[0]);r['year']=r.date.str[:4].astype(int);r['cutoff']=cutoff;r['method']=method;r['seed']=seed;r['score']=p;r['probability']=p;r['direction_up']=(p>.5).astype(int);r['actual_up']=(r.actual>0).astype(int)
    return r
'''+tail)
put('prepare47.py','''from common47 import *
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
''')
put('fit47.py','''from common47 import *
def main():
    check_frozen();assert not (OUT/'fitting_manifest.json').exists();run=manifest('fitting');obs=csv('observation_table');source=read(OUT/'source_heads.json');heads=[];traces=[];metrics=[];files=[]
    for cutoff in cfg()['decision_dates']:
        for seed in cfg()['seeds']:
            src=[next(h for h in source if h['cutoff']==cutoff and h['seed']==seed and h['method']==m) for m in METHODS];d=arrays(src[0]);ids,xx,y,raw,w,age=training_interface(obs,d,cutoff)
            coefs=[];ms=[];ts=[]
            for x in xx:
                theta,trace=fit_newton(x,y,w,cfg()['probe']);val,grad,hess=objective(theta,design(x),y,w,.01);coefs.append(theta);ts.append(trace);ms.append(dict(objective=val,gradient_inf=float(abs(grad).max()),hessian_min=float(np.linalg.eigvalsh(hess).min())))
            offset=design(xx[1])@coefs[1];gamma,trace=fit_offset(offset,xx[2][:,-1],y,w,cfg()['offset_probe']);theta=np.r_[coefs[1][:-1],gamma,coefs[1][-1]];val,grad,hess=offset_objective(gamma,offset,xx[2][:,-1],y,w,.01);coefs.append(theta);ts.append(trace);ms.append(dict(objective=val+.005*float(coefs[1][:-1]@coefs[1][:-1]),reduced_objective=val,gradient_inf=abs(grad),hessian_min=hess))
            assert ms[2]['objective']<=ms[3]['objective']+1e-12<=ms[1]['objective']+2e-12 and ms[1]['objective']<=ms[0]['objective']+1e-12
            path=CACHE/f'training_{cutoff}_{seed}.npz';np.savez_compressed(path,row_index=ids,x18=xx[0],x19=xx[1],x23=xx[2],direction=y,weight=w,raw_weight=raw,age_days=age);files.append(path)
            for method,theta,trace,stats,old in zip(METHODS,coefs,ts,ms,src):
                job=f'{method}_{cutoff}_{seed}';h=dict(history=NEW[0],method=method,cutoff=cutoff,seed=seed,job=job,train_n=len(ids),coefficients=theta.tolist(),iterations=trace[-1]['iteration'],frozen_pipeline_file=old['cache_file'],frozen_pipeline_sha256=old['cache_sha256'],model_project_file=old['model_project_file'],cache_file=str(path.relative_to(PROJECT)),cache_sha256=sha(path),**stats);heads.append(h);traces.extend(dict(job=job,**r) for r in trace);metrics.append(dict(job=job,method=method,cutoff=cutoff,seed=seed,**stats))
            print(f'Fitted {len(heads)}/72 weighted heads: {cutoff},seed{seed}; no weekly scoring.',flush=True)
    assert len(heads)==72
    p=OUT/'heads.json';save(p,heads);files.append(p)
    for name,data in [('solver_trace',traces),('training_metrics',metrics)]:p=OUT/f'{name}.csv';pd.DataFrame(data).to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_neural_fits=0,new_feature_inference=0,new_head_fits=72,joint_logistic_fits=54,conditional_scalar_fits=18,all_converged=True,weekly_scoring_during_fit=False)
    print('All72weighted heads frozen before new scoring.',flush=True)
if __name__=='__main__':main()
''')
put('score47.py','''from common47 import *
def main():
    check_frozen();check_phase('fitting');assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');obs=csv('observation_table');base=csv('baseline_model_predictions');be=csv('baseline_ensemble_predictions');heads=read(OUT/'heads.json');parts=[];provenance=[];freq=[]
    summary=csv('weight_summary').set_index('cutoff')
    for ref in read(OUT/'source_testing_features.json'):
        d=arrays(ref);cutoff=ref['cutoff'];seed=ref['seed'];ids=d['row_index'];assert all(f'{int(date[:4])-1}-12-31'==cutoff for date in obs.date.iloc[ids])
        for method,key in zip(METHODS,['x18','x19','x23','x23']):
            h=next(h for h in heads if h['cutoff']==cutoff and h['seed']==seed and h['method']==method);p=probability(design(d[key])@np.asarray(h['coefficients']));parts.append(prediction_rows(obs,ids,cutoff,method,seed,p));provenance.append(dict(cutoff=cutoff,seed=seed,method=method,n=len(ids),training_source=h['frozen_pipeline_file'],testing_source=ref['cache_file'],head_job=h['job']))
        if seed==cfg()['seeds'][0]:
            p=np.full(len(ids),summary.loc[cutoff,'weighted_up_frequency']);z=prediction_rows(obs,ids,cutoff,'timeweighted_training_frequency',-1,p);freq.append(z)
    g=base[base.history.eq(ANNUAL)&~base.method.isin(METHODS)].copy();g['history']=NEW[0];parts.append(g)
    new=pd.concat(parts,ignore_index=True)[base.columns];assert len(new)==4352;models=pd.concat([base,new],ignore_index=True);assert len(models)==121856
    en=[]
    for (history,method,date),g in new.groupby(['history','method','date'],sort=False):
        assert len(g)==(1 if method=='training_frequency' else 3);r=g.iloc[0].to_dict();r.pop('seed');r['score']=float(g.score.mean());r['probability']=float(g.probability.mean()) if method!='native_mse' else np.nan;r['direction_up']=int(r['score']>0 if method=='native_mse' else r['probability']>.5);en.append(r)
    ensemble=pd.concat([be,pd.DataFrame(en)[be.columns]],ignore_index=True);assert len(ensemble)==45696 and ensemble.history.nunique()==28
    files=[]
    for name,t in [('model_predictions',models),('ensemble_predictions',ensemble),('prediction_sources',pd.DataFrame(provenance)),('weighted_frequency_predictions',pd.concat(freq,ignore_index=True))]:p=OUT/f'{name}.csv';t.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_learned_seed_rows=3264,new_model_rows=4352,old_histories_preserved=27,new_head_changes_during_scoring=0);print('One annual weighted-head policy scored on272weeks;27oldhistories preserved.',flush=True)
if __name__=='__main__':main()
''')
put('evaluate47.py','''from common47 import *
def case(a,b,y):return 'recovery' if a==y and b!=y else 'regression' if a!=y and b==y else 'stable_correct' if a==y else 'stable_wrong'
def compute(models,ensemble):
    met=[];seeds=[];states=[];freqmet=[];ctx=csv('weekly_context')[['row_index','date','state']];freq=csv('weighted_frequency_predictions')
    for w in periods():
        e=ensemble[ensemble.date.between(w['start'],w['end'])];s=models[models.date.between(w['start'],w['end'])&models.seed.ne(-1)]
        for (h,m),g in e.groupby(['history','method']):met.append(dict(period=w['name'],history=h,method=m,**metric(g)))
        for (h,m,seed),g in s.groupby(['history','method','seed']):seeds.append(dict(period=w['name'],history=h,method=m,seed=int(seed),**metric(g)))
        q=e[e.history.isin(REPORT_HIST)&e.method.isin(METHODS)].merge(ctx,on=['row_index','date'],validate='many_to_one')
        for h in REPORT_HIST:
            for m in METHODS:
                for state in STATES:states.append(dict(period=w['name'],history=h,method=m,state=state,**metric(q[q.history.eq(h)&q.method.eq(m)&q.state.eq(state)])))
        freqmet.append(dict(period=w['name'],**metric(freq[freq.date.between(w['start'],w['end'])])))
    pairs=[]
    for w in cfg()['windows']:
        e=ensemble[ensemble.date.between(w['start'],w['end'])];ids=bootstrap_indices(w['n'])
        for spec in cfg()['primary_comparisons_per_window']:
            a=e[e.history.eq(spec['history'])&e.method.eq(spec['candidate'])].sort_values('date');b=e[e.history.eq(spec['reference_history'])&e.method.eq(spec['reference'])].sort_values('date');assert a.date.tolist()==b.date.tolist()
            def loss(x):return x.direction_up.ne(x.actual_up).to_numpy(float) if spec['metric']=='direction_error' else (x.probability.to_numpy()-x.actual_up.to_numpy())**2
            pairs.append(dict(window=w['name'],**spec,**difference(loss(a)-loss(b),ids)))
    for r,p in zip(pairs,holm([r['p'] for r in pairs])):r['holm_adjusted_p']=p
    ei=ensemble.set_index(['history','method','row_index']);q=ensemble[ensemble.history.isin(NEW)&ensemble.method.isin(METHODS)].merge(ctx,on=['row_index','date'],validate='many_to_one');weekly=[]
    for r in q.itertuples():
        out=dict(history=r.history,method=r.method,cutoff=r.cutoff,date=r.date,row_index=r.row_index,state=r.state,year=r.year,probability=r.probability,actual_up=r.actual_up)
        for label,h in [('annual',ANNUAL),('quarter4',QUARTER)]:
            b=ei.loc[(h,r.method,r.row_index)];out.update({label+'_probability':b.probability,'changed_vs_'+label:r.probability!=b.probability,'case_vs_'+label:case(r.direction_up,b.direction_up,r.actual_up),'brier_vs_'+label:(r.probability-r.actual_up)**2-(b.probability-b.actual_up)**2})
        weekly.append(out)
    weekly=pd.DataFrame(weekly);comparisons=[]
    for w in periods():
        for method,g in weekly[weekly.date.between(w['start'],w['end'])].groupby('method'):
            for label,h in [('annual',ANNUAL),('quarter4',QUARTER)]:comparisons.append(dict(period=w['name'],history=NEW[0],method=method,reference_history=h,n=len(g),changed_probability_weeks=int(g['changed_vs_'+label].sum()),recoveries=int(g['case_vs_'+label].eq('recovery').sum()),regressions=int(g['case_vs_'+label].eq('regression').sum()),brier_difference=float(g['brier_vs_'+label].mean())))
    return dict(ensemble_metrics=pd.DataFrame(met),seed_metrics=pd.DataFrame(seeds),state_metrics=pd.DataFrame(states),weighted_frequency_metrics=pd.DataFrame(freqmet),weekly_policy_effects=weekly,reference_comparisons=pd.DataFrame(comparisons),all_2026_cases=weekly[weekly.year.eq(2026)]),pairs
def main():
    check_frozen();check_phase('scoring');assert not (OUT/'evaluation_manifest.json').exists();run=manifest('evaluation');tables,pairs=compute(csv('model_predictions'),csv('ensemble_predictions'));files=[]
    for name,t in tables.items():p=OUT/f'{name}.csv';t.to_csv(p,index=False);files.append(p)
    p=OUT/'primary_comparisons.json';save(p,pairs);files.append(p);assert len(pairs)==24;check_frozen();finish(run,files,exploratory_comparisons=24,new_blind_holdout=False)
    m=tables['ensemble_metrics'];print(m[m.period.isin(['extension_2021_2023','recent_2024_2026','year_2026'])&m.history.isin(REPORT_HIST)&m.method.isin(PRIMARY)][['period','history','method','correct_directions','n','brier']].to_string(index=False),flush=True)
if __name__=='__main__':main()
''')
put('delivery47.py',prev('delivery').replace('common46','common47').replace("['fitting','validation','scoring','evaluation','report']","['fitting','scoring','evaluation','report']").replace('5986','6056'))
put('run47.py',prev('run').replace('{phase}46.py','{phase}47.py').replace("'fit','validate','score'","'fit','score'"))
pold=json.loads((old/'protocol.json').read_text(encoding='utf-8'));p28=json.loads((P/'research_v28/protocol.json').read_text(encoding='utf-8'))
p={k:pold[k] for k in ['label_end','windows','seeds','report_python','states','state_definition','preserved_results']}
p.update(version=47,experiment='Annual fixed5-year frozen-design logistic heads with fixed2-year exponential sample weighting',decision_dates=[f['cutoff'] for f in p28['folds'] if f['cutoff'].endswith('12-31')],probe=p28['probe'],offset_probe=p28['offset_probe'],neural=p28['neural'])
p['probe']['objective']='Weighted MEAN binary log loss (positive weights normalized to sum1) +.01/2*sum(squared slopes) in frozen original annual standardized coordinates. Free unpenalized intercept, initialized logit(weighted training up frequency); original full-batch float64 Newton settings. No re-estimation of clipping/scales/rays/projections or neural features.'
p['scope']='One new annual policy only; R19/R23/R25 primary, R18 diagnostic. Keep5-calendar-year mature daily anchors, original nextopen-to-nextopen weekly target, annual natural20 neural models, all27oldhistories and fourstate diagnostic labels. Change only supervised head loss sample weights. No neural fits/inference, no extra calibration, gate, window/cadence/half-life/seed selection.'
p['weights']='At priorDec31 cutoff, age=(cutoff-original signal date) in calendar days. Raw weight=2**(-age/730.5),2years=730.5days fixed; no floors or clipping; normalized weight=raw/sum(raw). Weight depends on date only. Membership requires joint label maturity<=cutoff and signaldate>=cutoff minus5calendar years plus1day. Allretained rows have positive mass. No latest outcomes or volatility-dependent weighting. Kish1/sum(w^2)descriptive concentration, not independent sample size.'
p['features']='Reuse R28 annual rolling5 training/testing arrays x18,x19,x23 exactly. Neural features, market descriptor clipping/means/sd, original unweighted supervisedR18 ray and vol/order least-squares projections remain frozen. WeightedR18 fit is diagnostic and does not redefine interaction rays. Training-state residual orthogonality remains the original unweighted property; no weighted orthogonality claim. This isolates weighted convex coefficients under fixed supervised feature engineering, not a fully weighted pipeline or OOF experiment.'
p['fitting']='18annualcutoff/seedjobs. Fit weightedR18,R19,R23 jointly in respective frozen29/30/31-feature coordinates; forR25 freeze newly weightedR19 allcoefficients/intercept and fit onlygamma on frozenorder residual. Originallambda.01, free intercept exceptR25 none, no new coefficient caps. Normalized weighted objective keeps regularization scale comparable; 54joint+18scalar=72candidatefits. Trainall72beforeANYnewweeklyscoring.'
p['controls']='Copy annual native_mse and unweighted training_frequency exactly into candidatehistory. Separately report timeweighted training frequency and its weekly forecasts as descriptive diagnostic only, to expose class-prior shifts; no additional candidate fits or p-value tests for it.'
p['scoring']='Annual cutoff=priorDec31 strictly earlier than originalweeklysignal. Three original seed mean probabilities using original per-groupSeries.mean;strict>.5up. Frozen same272weeks2021-Aug2026 and no future labels in fitting. No posthoc probability calibration. One newhistory,27oldhistories preserved.'
p['primary_comparisons_per_window']=[dict(history='annual_head_timeweight2y',candidate=m,reference_history=h,reference=m,metric=metric) for h in ['rolling5_annual20','weekly_state_validated'] for m in ['learned_vol_interaction','learned_order_extension','learned_order_offset'] for metric in ['direction_error','brier']]
p['inference']='24predeclared exploratorycomparisons:1candidate x2references x3primarymethods x2losses x2periods. Circular8week10000bootstrap,seed20260910,centered2sidedp andHolmacross24. Allhistoryviewed; no blindholdout or adjustmentforadaptivepriorrounds.'
p['verification']='Freeze11sources,protocol,weightmembership,inputhashes and6056oldfiles; inheritedverified18annualmodel/cacheprovenance. Independentlyreconstruct membership,causality,calendarweights,ESS,andfrozenfeaturealgebra. Equalweightobjective/gradient/Hessian equivalence tooriginal; finite differences andweightscaling checks. Archiveduniformheads must still satisfyoriginaloptimalityandpredictions. Freeze72candidatefits;independently solve54viaL-BFGS-B and18viaBrent. Poisonfuture/outside-trainingdata andrequire unchangedtraininginterface; verify allpredictions,exactcontrols,27oldhistories,metrics,diagnosticsand24comparisons;reportvisualreviewanddelivery.'
p['limitations']='Daily labels overlap; weight concentration is not independent N. Learned frozen features/rays already use same historical training labels as original, no OOFclaim. Dateweighting doesnot identify latentstates. R25conditionalparent refit uses sameweightedtrainingrows. Allhistoricaloutcomes previouslyviewed; no strategyPnL, baselinepromotionordeployment claim.'
p['budgets']=dict(source_files=11,old_files=6056,annual_cutoffs=6,seeds=3,jobs=18,new_head_fits=72,joint_fits=54,scalar_fits=18,new_neural_fits=0,new_feature_inference=0,old_histories=27,total_histories=28,new_learned_seed_predictions=3264,new_model_rows=4352,total_model_rows=121856,total_ensemble_rows=45696,comparisons=24)
put('protocol.json',json.dumps(p,ensure_ascii=False,indent=2))
put('README.md','''第47轮固定比较一个候选：5年窗口、年度更新、2年半衰期的时间加权分类头。日期权重2^(-age_days/730.5)，归一化总和为1；原lambda=0.01不变。神经网络、原特征缩放、原监督信号射线及交互项投影全部冻结，只重拟合分类系数。R25冻结本轮加权R19后拟合一个订单残差系数。

共18个年度／种子任务、72次候选拟合，无神经训练或推理。拟合全部完成再评分。两个固定对照为原年度与原四状态季度，24项预定探索性比较；加权标签频率另作诊断，不混为学习方法。历史已查看，没有盲测。保留27条旧历史和6,056个旧文件。

运行 `research_v4/.venv_gpu/Scripts/python.exe -B -u research_v47/run47.py prepare contract fit score evaluate verify report`。图表和解读核对后单独运行delivery47.py。全部11个源文件需在准备前就绪。
''')
print('R47 seven sources and protocol staged. Solver,contract,verify,report remain before freeze.')

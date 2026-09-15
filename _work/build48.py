from pathlib import Path
import json
P=Path('D:/ddos_v3');R=P/'research_v48';assert not R.exists();R.mkdir();OLD=P/'research_v47'
def put(n,s):(R/n).write_text(s,encoding='utf-8')
def old(n):return (OLD/(n+'47.py')).read_text(encoding='utf-8')
s=old('common');s=s.replace('from solver47 import probability,design,weights_for,objective,offset_objective,fit_newton,fit_offset','')
s=s.replace("PREV=PROJECT/'research_v46'","PREV=PROJECT/'research_v47'")
s=s.replace("ANNUAL='rolling5_annual20';QUARTER='weekly_state_validated';NEW=['annual_head_timeweight2y'];REPORT_HIST=[ANNUAL,QUARTER]+NEW", "ANNUAL='rolling5_annual20';QUARTER='weekly_state_validated';WEIGHTED='annual_head_timeweight2y';INTERCEPT='annual_head_weighted_intercept';SLOPES='annual_head_weighted_slopes';NEW=[INTERCEPT,SLOPES];REPORT_HIST=[ANNUAL,WEIGHTED,INTERCEPT,SLOPES,QUARTER];REFS=[ANNUAL,WEIGHTED,QUARTER]")
s=s.replace('6056','6143')
a=s.index('def source_inventory():');b=s.index('def old_evidence():')
s=s[:a]+'''def source_inventory():
    uniform=read(PREV/'results/source_heads.json');weighted=read(PREV/'results/heads.json');models=read(PREV/'results/source_models.json');tests=read(PREV/'results/source_testing_features.json')
    assert len(uniform)==len(weighted)==72 and len(models)==len(tests)==18
    return uniform,weighted,models,tests
def input_hashes():
    uniform,weighted,models,tests=source_inventory();paths=[a for a,b in COPIES]+[PREV/'protocol.json',V28/'protocol.json']
    paths += [PREV/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','verification.json','结果解读与下一步.md','source_heads.json','heads.json','source_models.json','source_testing_features.json','training_weights.csv','weight_summary.csv']]
    paths += [PROJECT/h['cache_file'] for h in uniform+weighted]+[PROJECT/t['cache_file'] for t in tests]+[PROJECT/m['project_file'] for m in models]
    return {str(p.relative_to(PROJECT)):sha(p) for p in paths}
'''+s[b:]
a=s.index('def training_interface(');b=s.index('def prediction_rows(',a);s=s[:a]+s[b:]
s=s.replace('def prediction_rows(obs,ids,cutoff,method,seed,p):','def prediction_rows(obs,ids,history,cutoff,method,seed,p):').replace("r.insert(0,'history',NEW[0])","r.insert(0,'history',history)")
idx=s.index('def periods():')
s=s[:idx]+'''def probability(z):return np.exp(-np.logaddexp(0.,-np.asarray(z,float)))
def design(x):return np.column_stack([np.asarray(x,float),np.ones(len(x))])
def assemble_records(uniform,weighted):
    ui={(h['cutoff'],h['seed'],h['method']):h for h in uniform};wi={(h['cutoff'],h['seed'],h['method']):h for h in weighted};assert set(ui)==set(wi);rows=[]
    for cutoff in cfg()['decision_dates']:
        for seed in cfg()['seeds']:
            for method in METHODS:
                key=(cutoff,seed,method);u=ui[key];w=wi[key];a=np.asarray(u['coefficients']);b=np.asarray(w['coefficients']);assert a.shape==b.shape
                for history,theta in [(INTERCEPT,np.r_[a[:-1],b[-1]]),(SLOPES,np.r_[b[:-1],a[-1]])]:
                    rows.append(dict(history=history,cutoff=cutoff,seed=seed,method=method,job=f'{history}_{method}_{cutoff}_{seed}',coefficients=theta.tolist(),uniform_job=u['job'],weighted_job=w['job'],train_n=u['train_n'],frozen_pipeline_file=u['cache_file'],frozen_pipeline_sha256=u['cache_sha256'],model_project_file=u['model_project_file'],slope_source='uniform' if history==INTERCEPT else 'weighted',intercept_source='weighted' if history==INTERCEPT else 'uniform'))
    return rows
def loss_values(row,metric):return float(row.direction_up!=row.actual_up) if metric=='direction_error' else float((row.probability-row.actual_up)**2)
def factorial_values(u,i,s,w):
    total=w-u;interaction=w-i-s+u;phi_i=.5*((i-u)+(w-s));phi_s=.5*((s-u)+(w-i))
    return dict(total=total,intercept_at_uniform_slopes=i-u,slopes_at_uniform_intercept=s-u,intercept_at_weighted_slopes=w-s,slopes_at_weighted_intercept=w-i,interaction=interaction,symmetric_intercept=phi_i,symmetric_slopes=phi_s)
'''+s[idx:]
put('common48.py',s.replace('Annual time-weighted convex heads in frozen neural and interaction coordinates.','Fixed intercept/slope swaps between verified annual equal-weight and time-weighted heads.'))
put('prepare48.py','''from common48 import *
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'preparation_manifest.json').exists();assert len(list(ROOT.glob('*.py')))==10
    r=manifest('preparation');r['old_evidence']=old_evidence();assert read(PREV/'results/verification.json')['status']=='PASS';save(OUT/'initial_freeze.json',r);files=[OUT/'initial_freeze.json']
    for src,name in COPIES:p=OUT/name;p.write_bytes(src.read_bytes());files.append(p)
    for src,dst in [('source_heads.json','uniform_heads.json'),('heads.json','weighted_heads.json'),('source_models.json','source_models.json'),('source_testing_features.json','source_testing_features.json'),('training_weights.csv','training_weights.csv'),('weight_summary.csv','weight_summary.csv')]:
        p=OUT/dst;p.write_bytes((PREV/'results'/src).read_bytes());files.append(p)
    finish(r,files,old_files=6143,new_neural_fits=0,new_head_fits=0,assembled_head_records=144);print('R48:10sources,protocol,inputs and6143oldfiles frozen.',flush=True)
if __name__=='__main__':main()
''')
put('assemble48.py','''from common48 import *
def main():
    check_frozen();assert not (OUT/'assembly_manifest.json').exists();run=manifest('assembly');heads=assemble_records(read(OUT/'uniform_heads.json'),read(OUT/'weighted_heads.json'));assert len(heads)==144;files=[]
    p=OUT/'heads.json';save(p,heads);files.append(p);rows=[]
    for h in heads:
        for j,value in enumerate(h['coefficients']):rows.append(dict(history=h['history'],cutoff=h['cutoff'],seed=h['seed'],method=h['method'],coordinate=j,role='intercept' if j==len(h['coefficients'])-1 else 'slope',source=h['intercept_source'] if j==len(h['coefficients'])-1 else h['slope_source'],value=value))
    p=OUT/'coefficient_sources.csv';pd.DataFrame(rows).to_csv(p,index=False);files.append(p);check_frozen();finish(run,files,assembled_heads=144,new_head_fits=0,new_neural_fits=0,labels_used=False,weekly_scoring_during_assembly=False);print('144fixed hybrid heads frozen;no fitting or weekly scoring.',flush=True)
if __name__=='__main__':main()
''')
put('score48.py','''from common48 import *
def main():
    check_frozen();check_phase('assembly');assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');obs=csv('observation_table');base=csv('baseline_model_predictions');be=csv('baseline_ensemble_predictions');heads=read(OUT/'heads.json');uniform=read(OUT/'uniform_heads.json');weighted=read(OUT/'weighted_heads.json');parts=[];provenance=[];terms=[]
    ui={(h['cutoff'],h['seed'],h['method']):h for h in uniform};wi={(h['cutoff'],h['seed'],h['method']):h for h in weighted};hi={(h['history'],h['cutoff'],h['seed'],h['method']):h for h in heads}
    for ref in read(OUT/'source_testing_features.json'):
        d=arrays(ref);cutoff=ref['cutoff'];seed=ref['seed'];ids=d['row_index'];assert all(f'{int(date[:4])-1}-12-31'==cutoff for date in obs.date.iloc[ids])
        for method,key in zip(METHODS,['x18','x19','x23','x23']):
            k=(cutoff,seed,method);u=np.asarray(ui[k]['coefficients']);w=np.asarray(wi[k]['coefficients']);x=d[key];z0=design(x)@u;z1=design(x)@w;di=float(w[-1]-u[-1]);ds=x@(w[:-1]-u[:-1]);zz={}
            for history in NEW:
                h=hi[(history,*k)];z=design(x)@np.asarray(h['coefficients']);zz[history]=z;p=probability(z);parts.append(prediction_rows(obs,ids,history,cutoff,method,seed,p));provenance.append(dict(history=history,cutoff=cutoff,seed=seed,method=method,n=len(ids),training_source=h['frozen_pipeline_file'],testing_source=ref['cache_file'],head_job=h['job']))
            for j,idx in enumerate(ids):terms.append(dict(cutoff=cutoff,seed=seed,method=method,row_index=int(idx),date=obs.date.iloc[idx],uniform_logit=float(z0[j]),weighted_logit=float(z1[j]),intercept_hybrid_logit=float(zz[INTERCEPT][j]),slopes_hybrid_logit=float(zz[SLOPES][j]),intercept_change=di,slopes_change=float(ds[j]),identity_residual=float(z1[j]-z0[j]-di-ds[j])))
    for history in NEW:
        g=base[base.history.eq(ANNUAL)&~base.method.isin(METHODS)].copy();g['history']=history;parts.append(g)
    new=pd.concat(parts,ignore_index=True)[base.columns];assert len(new)==8704;models=pd.concat([base,new],ignore_index=True);assert len(models)==130560
    en=[]
    for (history,method,date),g in new.groupby(['history','method','date'],sort=False):
        assert len(g)==(1 if method=='training_frequency' else 3);r=g.iloc[0].to_dict();r.pop('seed');r['score']=float(g.score.mean());r['probability']=float(g.probability.mean()) if method!='native_mse' else np.nan;r['direction_up']=int(r['score']>0 if method=='native_mse' else r['probability']>.5);en.append(r)
    ensemble=pd.concat([be,pd.DataFrame(en)[be.columns]],ignore_index=True);assert len(ensemble)==48960 and ensemble.history.nunique()==30;files=[]
    for name,t in [('model_predictions',models),('ensemble_predictions',ensemble),('prediction_sources',pd.DataFrame(provenance)),('seed_logit_decomposition',pd.DataFrame(terms))]:p=OUT/f'{name}.csv';t.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_learned_seed_rows=6528,new_model_rows=8704,old_histories_preserved=28,new_head_changes_during_scoring=0,logit_cells=3264);print('Two fixed hybrids scored on272weeks;28oldhistories preserved.',flush=True)
if __name__=='__main__':main()
''')
s=old('evaluate').replace('common47','common48')
s=s.replace(";freqmet=[]",'').replace(";freq=csv('weighted_frequency_predictions')",'')
s=s.replace("        freqmet.append(dict(period=w['name'],**metric(freq[freq.date.between(w['start'],w['end'])])))\n",'')
s=s.replace("[('annual',ANNUAL),('quarter4',QUARTER)]","[('annual',ANNUAL),('weighted',WEIGHTED),('quarter4',QUARTER)]")
s=s.replace("for method,g in weekly[weekly.date.between(w['start'],w['end'])].groupby('method'):","for (history,method),g in weekly[weekly.date.between(w['start'],w['end'])].groupby(['history','method']):")
s=s.replace("history=NEW[0],method=method,reference_history=h","history=history,method=method,reference_history=h")
a=s.index('    return dict(ensemble_metrics=');b=s.index('\ndef main():',a)
s=s[:a]+'''    factorial=[];summary=[]
    for method in METHODS:
        for idx in ensemble[ensemble.history.eq(ANNUAL)&ensemble.method.eq(method)].row_index:
            rows=[ei.loc[(h,method,idx)] for h in [ANNUAL,INTERCEPT,SLOPES,WEIGHTED]]
            for loss in ['direction_error','brier']:
                u,i,s,w=[loss_values(r,loss) for r in rows];factorial.append(dict(method=method,row_index=int(idx),date=rows[0].date,metric=loss,uniform_loss=u,intercept_loss=i,slopes_loss=s,weighted_loss=w,**factorial_values(u,i,s,w)))
    factorial=pd.DataFrame(factorial)
    columns=['total','intercept_at_uniform_slopes','slopes_at_uniform_intercept','intercept_at_weighted_slopes','slopes_at_weighted_intercept','interaction','symmetric_intercept','symmetric_slopes']
    for period in periods():
        for (method,loss),g in factorial[factorial.date.between(period['start'],period['end'])].groupby(['method','metric']):summary.append(dict(period=period['name'],method=method,metric=loss,n=len(g),**{k:float(g[k].mean()) for k in columns}))
    return dict(ensemble_metrics=pd.DataFrame(met),seed_metrics=pd.DataFrame(seeds),state_metrics=pd.DataFrame(states),weekly_policy_effects=weekly,reference_comparisons=pd.DataFrame(comparisons),weekly_factorial_effects=factorial,factorial_effects=pd.DataFrame(summary),all_2026_cases=weekly[weekly.year.eq(2026)]),pairs
'''+s[b:]
s=s.replace('len(pairs)==24','len(pairs)==72').replace('exploratory_comparisons=24','exploratory_comparisons=72');put('evaluate48.py',s)
put('delivery48.py',old('delivery').replace('common47','common48').replace("'fitting'","'assembly'").replace('6056','6143'))
put('run48.py',old('run').replace('{phase}47.py','{phase}48.py').replace("'fit'","'assemble'"))
p47=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'));p={k:p47[k] for k in ['label_end','windows','seeds','report_python','states','state_definition','preserved_results','decision_dates']}
p.update(version=48,experiment='Fixed two-by-two intercept and slope swaps of equal-weight and timeweighted annual heads',policies=['annual_head_weighted_intercept','annual_head_weighted_slopes'],references=['rolling5_annual20','annual_head_timeweight2y','weekly_state_validated'])
p['scope']='Two mechanical hybrids only:Uslopes/Wintercept andWslopes/Uintercept; U=originalR28annualrolling5natural20 equal-weight heads;W=R47fixed2yearhalf-life weighted heads. Same6annualdates/3seeds/R18diagnostic/R19R23R25primary. No coefficient refitting, calibration, neural training/inference, half-life/cadence/window/threshold/seed search or selection after results.'
p['coordinate_contract']='R47 used exact originalR28 x18/x19/x23 designs, unweighted neural-feature standardization and supervised rays/projections; verify this byte-for-byte for every source job. Coefficients use same coordinate order and final element is free intercept. R25 last slope is ordergamma; allslopes (includinggamma) copied together; originalR25 andweightedR25 each inherit their own R19 parent but hybrids are not new constrained optima.'
p['assembly']='For everycutoff/seed/method u,w: I=concat(u[:-1],w[-1]),S=concat(w[:-1],u[-1]). Exact element copies, no arithmetic blend or optimization. Build all144records before any new weekly scoring. Source metadata is included; no labels or outcomes enter construction.'
p['training_contract']='Inheritedsource training members unchanged:signaldate>=priorDec31minus5calendar years plus1day,alltargetsjoint_completed<=cutoff. R47rawdateweights2**(-age_days/730.5),sum-normalized;original lambda.01. Sourcecoefficients already verified. This round does not refit or redefine weights,scales,features or projections.'
p['scoring']='AnnualpreviousDec31strictlyearlierthansignal;272originalweeklysignals2021-Aug2026 withoriginalmaturetarget. Frozen testing arrays; score each hybrid in exactsame design. Original3seedSeries.meanprobabilities andstrict>.5up. Copy native_mse andunweightedtraining_frequency exactly fromannual. Preserveall28oldhistories;2newhistories.'
p['diagnostics']='Reportseedlogitidentity zW-zU=(bW-bU)+x@(betaW-betaU),andallfourcorneroutputs. Logit changesadd;sigmoidprobabilities,ensemble averaging,direction error andBrier do not generallyadd. Within sameyear/seed,intercept-only shifts preserve score ranking; pooledyears or three-seed ensembles need not. Report allseed/yearmetrics andoriginalfourstates includingzero cells.'
p['factorial_diagnostics']='For eachweek/method anddirectionerror/Brier,losses U,I,S,W. Conditionalintercept effects I-U andW-S;conditionalslopeeffects S-U andW-I;interaction W-I-S+U. Symmetricalgebraicshares phiI=((I-U)+(W-S))/2,phiS=((S-U)+(W-I))/2;sum exactlyW-U. Direction shares canbe fractionalweek equivalents. Prespecifythese descriptive comparisons across9periods including6years, no addedpvalues. This is fixed-coordinate model-output attribution,not economic causal inference ornewoptima.'
p['primary_comparisons_per_window']=[dict(history=h,candidate=m,reference_history=ref,reference=m,metric=metric) for h in p['policies'] for ref in p['references'] for m in ['learned_vol_interaction','learned_order_extension','learned_order_offset'] for metric in ['direction_error','brier']]
p['inference']='72prespecifiedexploratorycomparisons:2hybrids x3refs x3primarymethods x2losses x2periods. Circular8week10000bootstrapseed20260910,centeredtwo-sidedp,Holmacross72. Allhistoryalreadyviewed;no blindholdout oradjustmentforpreviousadaptive rounds.'
p['verification']='Freeze10sources,protocol,sourcecoefficients/caches and6143oldfiles. Validate6annualmaturemembershipsplits and18sharedfeaturejobs; replayboth72headendpoints. Verifyall144coefficientassemblies andeverycoordinate source; syntheticidentical-endpoint/boundary/logitidentitychecks. Independentreconstructnewpredictionsfromscalarsum/algebra;exactcontrols andoldhistories,seedrankingconstraint,allmetrics/logit/factorialdiagnostics and72statistics. Scientificsourceunchanged;report/visualQAthen delivery.'
p['limitations']='Hybrids are mechanicalcross-combinations,not fitted optima. Intercept/slopedivision depends onchosen coordinates,which remainfrozen here. Dailytraininglabels overlap; supervisedfeatures inheritedsame-samplefitting, noOOFclaim. Attribution confined tothese fixed models andhistoricalforecasts. No baselinepromotion, tradingPnL,deployment orindependentholdout claim.'
p['budgets']=dict(source_files=10,old_files=6143,annual_cutoffs=6,seeds=3,source_jobs=18,source_heads_per_endpoint=72,assembled_heads=144,new_head_fits=0,new_neural_fits=0,new_feature_inference=0,old_histories=28,total_histories=30,new_learned_seed_predictions=6528,new_model_rows=8704,total_model_rows=130560,total_ensemble_rows=48960,seed_logit_cells=3264,weekly_factorial_cells=2176,period_factorial_cells=72,comparisons=72)
put('protocol.json',json.dumps(p,ensure_ascii=False,indent=2));put('README.md','''第48轮固定拆分原等权与第47轮时间加权系数：原斜率＋加权截距、加权斜率＋原截距。保留同一年度、种子和特征坐标，144条组合参数逐元素复用；无新增拟合、神经推理或校准。

对照原年度、完整时间加权、原四状态季度，固定72项探索性比较。另报告四组合的条件影响与对称代数分摊，区分logit可加性和概率／损失非线性；不作经济因果归因。原28条历史与6,143个旧文件保留，全部历史已查看。

运行 `research_v4/.venv_gpu/Scripts/python.exe -B -u research_v48/run48.py prepare contract assemble score evaluate verify report`。图表及解读核对后单独运行delivery48.py。全部10个源文件需在准备前就绪。
''');print('R48 seven sources and protocol staged;contract,verify,report remain before freeze.')

from pathlib import Path
import json
ROOT=Path('D:/ddos_v3');old=ROOT/'research_v45';new=ROOT/'research_v46'
assert not new.exists();new.mkdir()
def put(n,s):(new/n).write_text(s,encoding='utf-8')
def previous(n):return (old/(n+'45.py')).read_text(encoding='utf-8').replace('common45','common46')
def replace(s,a,b):
    assert a in s,a
    return s.replace(a,b)
s=previous('common')
s=replace(s,"PREV=PROJECT/'research_v44'","PREV=PROJECT/'research_v45'")
s=replace(s,"POLICIES={'trend':'quarter_trend2_validated','volatility':'quarter_volatility2_validated'};NEW=list(POLICIES.values());ANNUAL='rolling5_annual20';QUARTER='weekly_state_validated';REFS=[QUARTER,ANNUAL];REPORT_HIST=[ANNUAL,QUARTER]+NEW", "PARENTS={'trend':'quarter_trend2_validated','volatility':'quarter_volatility2_validated'};POLICIES={'trend':'quarter_trend2_state4_validated','volatility':'quarter_volatility2_state4_validated'};NEW=list(POLICIES.values());ANNUAL='rolling5_annual20';QUARTER='weekly_state_validated';REFS=[QUARTER,ANNUAL];REPORT_HIST=[ANNUAL,QUARTER]+list(PARENTS.values())+NEW")
pos=s.index('\ndef read(')
s=s[:pos]+"\nCOPIES += [(PREV/'results'/f'{n}.csv',f'parent_{n}.csv') for n in ['gate_decisions','group_support']]\n"+s[pos:]
s=replace(s,"['preparation_manifest.json','delivery_manifest.json','verification.json','结果解读与下一步.md']", "['preparation_manifest.json','delivery_manifest.json','verification.json','结果解读与下一步.md','correction_heads.json','correction_parameters.csv']")
a=s.index('def old_evidence():');b=s.index('\ndef manifest',a)
s=s[:a]+'''def old_evidence():
    r=dict(read(PREV/'results/preparation_manifest.json')['old_evidence'])
    r.update({str((PREV/n).relative_to(PROJECT)):h for n,h in read(PREV/'results/delivery_manifest.json')['files'].items()})
    p=PREV/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==5986
    for n,h in r.items():assert sha(PROJECT/n)==h,n
    return r
'''+s[b:]
a=s.index('def validate_records(');b=s.index('\ndef periods',a)
s=s[:a]+'''def validate_records(bank,members,schedule,heads):
    hi={(h['cutoff'],h['method'],h['seed'],h['family'],h['component']):h for h in heads};gates=[];rows=[]
    for c in schedule.itertuples():
        ids=members[members.cutoff.eq(c.cutoff)&members.role.eq('validation')].row_index.tolist()
        trainids=members[members.cutoff.eq(c.cutoff)&members.role.eq('training')].row_index.tolist()
        for m in METHODS:
            tr=select(bank,trainids,m,cfg()['seeds'][0])
            for family in FAMILIES:
                for state in STATES:
                    group=component(state,family);nt=int(tr.state.eq(state).sum());ps=[];bases=[]
                    for seed in cfg()['seeds']:
                        g=select(bank,ids,m,seed);g=g[g.state.eq(state)];h=hi[(c.cutoff,m,seed,family,group)]
                        p=corrected(g.annual_logit,g.probability,h['offset']);ps.append(p);bases.append(g.probability.to_numpy())
                        for r,q in zip(g.itertuples(),p):rows.append(dict(cutoff=c.cutoff,method=m,seed=seed,family=family,component=group,row_index=r.row_index,date=r.date,joint_completed=r.joint_completed,state=r.state,annual_probability=r.probability,candidate_probability=float(q),actual_up=int(r.actual_up),offset=h['offset']))
                    p=np.mean(ps,axis=0);p0=np.mean(bases,axis=0)
                    decision=gate(c.mode,nt,p0,p,g.actual_up.to_numpy())
                    if decision['accepted']:assert h['fit_eligible'] and h['training_n']>=nt
                    gates.append(dict(cutoff=c.cutoff,mode=c.mode,method=m,family=family,component=group,state=state,training_n=nt,parent_training_n=h['training_n'],**decision))
    return pd.DataFrame(gates),pd.DataFrame(rows)
'''+s[b:]
put('common46.py',s)
s=previous('prepare').replace('5916','5986').replace('R45','R46');put('prepare46.py',s)
s=previous('contract');a=s.index('        for family,groups in FAMILIES.items():');b=s.index("    check_frozen(False);",a)
s=s[:a]+'''        for family in FAMILIES:
            for state in STATES:
                nt=int(tr.state.eq(state).sum());nv=int(va.state.eq(state).sum());group=component(state,family)
                parent_nt=int(tr.state.map(lambda s:component(s,family)).eq(group).sum())
                support.append(dict(cutoff=c.cutoff,family=family,component=group,state=state,mode=c.mode,training_n=nt,parent_training_n=parent_nt,validation_n=nv,train_supported=mode=='ready' and nt>=10,validation_supported=mode=='ready' and nt>=10 and nv>=5))
    p=OUT/'group_support.csv';pd.DataFrame(support).to_csv(p,index=False);assert len(support)==184
    original=csv('original_four_state_support').set_index(['cutoff','state'])
    for r in support:
        q=original.loc[(r['cutoff'],r['state'])]
        assert r['training_n']==q.training_n and r['validation_n']==q.validation_n
        assert r['train_supported']==q.train_supported and r['validation_supported']==q.validation_supported
'''+s[b:]
s=s.replace('group_support_cells=92','group_support_cells=184').replace('and two fixed unions; support frozen.','and shared parent fits with original child-state support; frozen.')
put('contract46.py',s)
put('fit46.py','''from common46 import *

def main():
    check_frozen();assert not (OUT/'fitting_manifest.json').exists();run=manifest('fitting');files=[]
    for name in ['correction_heads.json','correction_parameters.csv']:
        p=OUT/name;p.write_bytes((PREV/'results'/name).read_bytes());files.append(p)
    heads=read(OUT/'correction_heads.json');assert len(heads)==1104 and sum(h['fit_eligible'] for h in heads)==588
    check_frozen();finish(run,files,reused_scalar_fits=588,head_records=1104,new_scalar_fits=0,new_neural_fits=0,validation_labels_used=False)
    print('R45 coefficients reused byte-for-byte:588 fitted scalar offsets,1104 head records; no new fitting.',flush=True)
if __name__=='__main__':main()
''')
s=previous('validate').replace('368','736');put('validate46.py',s)
s=previous('score').replace("set_index(['cutoff','method','family','component'])","set_index(['cutoff','method','family','state'])").replace('gates.loc[(cutoff,b.method,family,group)]','gates.loc[(cutoff,b.method,family,b.state)]')
s=s.replace('108800','117504').replace('40800','44064').replace('nunique()==25','nunique()==27').replace('old_histories_preserved=23','old_histories_preserved=25').replace('23oldhistories','25oldhistories')
put('score46.py',s)
s=previous('evaluate')
s=s.replace("q=ei.loc[(QUARTER,r.method,r.row_index)];out=", "q=ei.loc[(QUARTER,r.method,r.row_index)];parent=ei.loc[(PARENTS[r.family],r.method,r.row_index)];out=")
s=s.replace("[('annual',b),('quarter4',q)]","[('annual',b),('quarter4',q),('parent2',parent)]")
s=s.replace("[('annual',ANNUAL),('quarter4',QUARTER)]","[('annual',ANNUAL),('quarter4',QUARTER),('parent2',PARENTS[g.family.iloc[0]])]")
a=s.index("    gates=csv('gate_decisions');");b=s.index('\ndef main():',a)
s=s[:a]+'''    gates=csv('gate_decisions');outcomes=[]
    parentg=csv('parent_gate_decisions').set_index(['cutoff','method','family','component'])
    original=csv('original_quarter_gates');original=original[original.family.eq('state')].set_index(['cutoff','method','component'])
    for r in gates.itertuples():
        h=POLICIES[r.family];g=weekly[weekly.history.eq(h)&weekly.method.eq(r.method)&weekly.cutoff.eq(r.cutoff)&weekly.state.eq(r.state)]
        pg=parentg.loc[(r.cutoff,r.method,r.family,r.component)];og=original.loc[(r.cutoff,r.method,r.state)]
        out=dict(cutoff=r.cutoff,history=h,method=r.method,family=r.family,component=r.component,state=r.state,accepted=bool(r.accepted),reason=r.reason,training_n=r.training_n,parent_training_n=r.parent_training_n,validation_n=r.validation_n,parent_accepted=bool(pg.accepted),parent_reason=pg.reason,original4_accepted=bool(og.accepted),original4_reason=og.reason,n=len(g))
        for label in ['annual','quarter4','parent2']:
            out.update({f'changed_vs_{label}':int(g[f'changed_vs_{label}'].sum()),f'recoveries_vs_{label}':int(g[f'case_vs_{label}'].eq('recovery').sum()),f'regressions_vs_{label}':int(g[f'case_vs_{label}'].eq('regression').sum()),f'brier_vs_{label}':float(g[f'brier_vs_{label}'].mean()) if len(g) else None})
        outcomes.append(out)
    support=csv('group_support');parent=csv('parent_group_support');old=csv('original_four_state_support');summ=[]
    groups=[('original4',old)]+[(f'parent_{f}',parent[parent.family.eq(f)]) for f in FAMILIES]+[(f'child_{f}',support[support.family.eq(f)]) for f in FAMILIES]
    for scheme,g in groups:
        z=g[g['mode'].eq('ready')];summ.append(dict(scheme=scheme,total_cells=len(g),ready_cells=len(z),training_supported_cells=int(z.train_supported.sum()),validation_supported_cells=int(z.validation_supported.sum()),training_support_fraction=float(z.train_supported.mean()),validation_support_fraction=float(z.validation_supported.mean())))
    return dict(ensemble_metrics=pd.DataFrame(met),seed_metrics=pd.DataFrame(seeds),state_metrics=pd.DataFrame(states),weekly_policy_effects=weekly,policy_coverage=pd.DataFrame(coverage),reference_comparisons=pd.DataFrame(comparisons),decision_outcomes=pd.DataFrame(outcomes),support_summary=pd.DataFrame(summ),all_2026_cases=weekly[weekly.year.eq(2026)]),pairs
'''+s[b:]
s=s.replace('len(pairs)==48','len(pairs)==72').replace('exploratory_comparisons=48','exploratory_comparisons=72');put('evaluate46.py',s)
s=previous('verify')
s=s.replace('len(gates)==368','len(gates)==736').replace('independent_gate_cells=368','independent_gate_cells=736')
s=s.replace("if independent_group(state[i],r.family)==r.component];bases=[]", "if state[i]==r.state];bases=[]")
s=s.replace("        n=len(ids);bd=", "        trainids=[i for i in rows.get((r.cutoff,'training'),[]) if state[i]==r.state]\n        assert r.training_n==len(trainids) and r.parent_training_n==hi[(r.cutoff,r.method,cfg()['seeds'][0],r.family,r.component)]['training_n']\n        n=len(ids);bd=")
s=s.replace("set_index(['cutoff','method','family','component'])","set_index(['cutoff','method','family','state'])")
s=s.replace('gates.loc[(r.cutoff,r.method,r.family,group)]','gates.loc[(r.cutoff,r.method,r.family,b.state)]')
s=s.replace('108800','117504').replace('40800','44064')
a=s.index('def check_diagnostics():');b=s.index('\ndef check_statistics():',a)
s=s[:a]+'''def check_diagnostics():
    weekly=csv('weekly_policy_effects');e=csv('ensemble_predictions').set_index(['history','method','row_index']);assert len(weekly)==2176
    for r in weekly.itertuples():
        a=e.loc[(r.history,r.method,r.row_index)];assert r.probability==a.probability and r.actual_up==a.actual_up
        for label,history in [('annual',ANNUAL),('quarter4',QUARTER),('parent2',PARENTS[r.family])]:
            b=e.loc[(history,r.method,r.row_index)];good=a.direction_up==a.actual_up;old=b.direction_up==a.actual_up
            expected='recovery' if good and not old else 'regression' if old and not good else 'stable_correct' if good else 'stable_wrong'
            assert getattr(r,label+'_probability')==b.probability and getattr(r,'case_vs_'+label)==expected and getattr(r,'changed_vs_'+label)==(a.probability!=b.probability)
            eq(getattr(r,'brier_vs_'+label),(a.probability-a.actual_up)**2-(b.probability-b.actual_up)**2)
    for name in ['policy_coverage','reference_comparisons']:
        for r in csv(name).itertuples():
            w=next(w for w in periods() if w['name']==r.period);g=weekly[weekly.history.eq(r.history)&weekly.method.eq(r.method)&weekly.date.between(w['start'],w['end'])];assert r.n==len(g)
            if name=='policy_coverage':assert [r.fit_eligible_weeks,r.validation_accepted_weeks,r.changed_probability_weeks]==[sum(g.fit_eligible),sum(g.validation_accepted),sum(g.changed_vs_annual)]
            else:
                label='annual' if r.reference_history==ANNUAL else 'quarter4' if r.reference_history==QUARTER else 'parent2'
                assert [r.changed_probability_weeks,r.recoveries,r.regressions]==[sum(g['changed_vs_'+label]),sum(g['case_vs_'+label]=='recovery'),sum(g['case_vs_'+label]=='regression')];eq(r.brier_difference,math.fsum(g['brier_vs_'+label])/len(g))
    outcomes=csv('decision_outcomes');assert len(outcomes)==736
    gates=csv('gate_decisions').set_index(['cutoff','method','family','state']);parent=csv('parent_gate_decisions').set_index(['cutoff','method','family','component']);orig=csv('original_quarter_gates');orig=orig[orig.family.eq('state')].set_index(['cutoff','method','component'])
    for r in outcomes.itertuples():
        d=gates.loc[(r.cutoff,r.method,r.family,r.state)];p=parent.loc[(r.cutoff,r.method,r.family,r.component)];o=orig.loc[(r.cutoff,r.method,r.state)]
        for k in ['accepted','reason','training_n','parent_training_n','validation_n']:assert getattr(r,k)==getattr(d,k)
        assert r.parent_accepted==p.accepted and r.parent_reason==p.reason and r.original4_accepted==o.accepted and r.original4_reason==o.reason
        g=weekly[weekly.history.eq(r.history)&weekly.method.eq(r.method)&weekly.cutoff.eq(r.cutoff)&weekly.state.eq(r.state)];assert len(g)==r.n
        for label in ['annual','quarter4','parent2']:
            assert getattr(r,'changed_vs_'+label)==sum(g['changed_vs_'+label])
            assert getattr(r,'recoveries_vs_'+label)==sum(g['case_vs_'+label]=='recovery') and getattr(r,'regressions_vs_'+label)==sum(g['case_vs_'+label]=='regression')
            eq(getattr(r,'brier_vs_'+label),math.fsum(g['brier_vs_'+label])/len(g) if len(g) else None)
    support=csv('group_support');assert len(support)==184
    members=csv('split_membership');pool=csv('weekly_signal_bank');pool=pool[pool.method.eq(METHODS[0])&pool.seed.eq(cfg()['seeds'][0])].set_index('row_index')
    for r in support.itertuples():
        tr=pool.loc[members[members.cutoff.eq(r.cutoff)&members.role.eq('training')].row_index];va=pool.loc[members[members.cutoff.eq(r.cutoff)&members.role.eq('validation')].row_index]
        nt=sum(tr.state==r.state);nv=sum(va.state==r.state);pn=sum(tr.state.map(lambda s:independent_group(s,r.family))==r.component)
        assert [r.training_n,r.validation_n,r.parent_training_n]==[nt,nv,pn] and r.train_supported==(r.mode=='ready' and nt>=10) and r.validation_supported==(r.mode=='ready' and nt>=10 and nv>=5)
    parent=csv('parent_group_support');old=csv('original_four_state_support')
    for r in csv('support_summary').itertuples():
        g=old if r.scheme=='original4' else (parent if r.scheme.startswith('parent_') else support)
        if r.scheme!='original4':g=g[g.family.eq(r.scheme.split('_')[1])]
        z=g[g['mode'].eq('ready')];assert [r.total_cells,r.ready_cells,r.training_supported_cells,r.validation_supported_cells]==[len(g),len(z),sum(z.train_supported),sum(z.validation_supported)];eq(r.training_support_fraction,sum(z.train_supported)/len(z));eq(r.validation_support_fraction,sum(z.validation_supported)/len(z))
    same_tables(csv('all_2026_cases'),weekly[weekly.year.eq(2026)])
    return dict(independent_weekly_effects=len(weekly),independent_decision_outcomes=len(outcomes),independent_support_cells=len(support))
'''+s[b:]
s=s.replace('==48','==72').replace('[0.]*48','[0.]*72').replace('range(48)','range(72)').replace('(48-rank)','(72-rank)')
a=s.index('    repair_names=');b=s.index("if __name__=='__main__':",a)
s=s[:a]+'''    for n in ['correction_heads.json','correction_parameters.csv']:assert sha(OUT/n)==sha(PREV/'results'/n)
    assert fit['independent_scalar_solutions']==588
    save(OUT/'verification.json',dict(status='PASS',started_utc=start,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=5986,old_histories_preserved=25,new_neural_fits=0,new_scalar_fits=0,reused_scalar_fits=588,independent_comparisons=72,**fit,**pred,**diag,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));print('R46 independent predictions, metrics, diagnostics and72comparisons PASS.',flush=True)
'''+s[b:]
put('verify46.py',s)
put('delivery46.py',previous('delivery').replace('5916','5986'))
put('run46.py',previous('run').replace("{phase}45.py","{phase}46.py"))
# Report is written separately before any freeze or scoring.
p=json.loads((old/'protocol.json').read_text(encoding='utf-8'))
for k in ['numerical_repair']:p.pop(k)
p.update(version=46,experiment='Reuse fixed two-group fits with independent original-four-state validation gates',policies={'trend':'quarter_trend2_state4_validated','volatility':'quarter_volatility2_state4_validated'},parent_policies={'trend':'quarter_trend2_validated','volatility':'quarter_volatility2_validated'},references=['corresponding_R45_parent_policy','weekly_state_validated','rolling5_annual20'])
p['scope']='Two shared-fit, four-state-gated candidates only. Reuse all R45 parent coefficients byte-for-byte, no new neural or scalar optimization for candidate construction. Annual models, market/order interactions, rolling5 natural20 training and all25 old prediction histories frozen. R18 diagnostic; R19,R23,R25 primary. No selecting a partition, threshold, window or seed after scoring.'
p['fitting']='Copy R45 correction_heads.json and correction_parameters.csv byte-for-byte after contract checks and before validation. Original 588 fits under SUM logistic loss +20*offset^2/2, |offset|<=0.5,60 bisections; parent training>=10. Independent verification may re-solve original training-only objective, but candidate coefficients never change.'
p['validation']='For each cutoff/method/family/original state, use that state subset of original heldout13 weeks to validate the corresponding shared parent offsets. Require original child-state training>=10 and child-state validation>=5, three-seed mean Brier improvement>1e-12, correct count nondecreasing. Parent fit must exist. Parent R45 gate is NOT required; child may accept even if parent rejected, or reject even if parent accepted. No new intersection gate, no threshold search, no refit. Candidate validation probabilities are retained even for unsupported states but not applied.'
p['scoring']='Use inherited signal-time original state to choose independent child gate and its parent seed coefficient. Rejected/unsupported/nohistory/reset gates fall back to exact annual seed and ensemble predictions. Use original per-group Series.mean aggregation. Preserve exact native_mse/training_frequency controls. Two new histories plus25 old on272 weeks.'
p['inference']='72 prespecified exploratory comparisons:2 candidates x3 references(corresponding R45 parent, original four-state quarter, annual) x3 primary methods x2 losses(directionerror,Brier) x2 windows. Circular8week10000seed20260910 centered two-sided p and Holm across72. Allhistory viewed, no blindholdout or correction for previous adaptive rounds.'
p['diagnostics']='Report six years, early/recent/pooled periods, all seeds and original four states including empty outcomes. Distinguish inherited parent fit support from required original child support; child support must exactly match original four-state recipe, so sample shortage is not solved by sharing coefficients. Report parent-versus-child acceptance transitions and future recoveries/regressions against all3 references; no selection from subgroup outcomes.'
p['verification']='Freeze11sources,protocol,inputs and5986oldfiles. Reconstruct original52/13 maturity splits, strict cutoff routing and child support. Freeze copied parent fits, freeze child validation, score, evaluate, independentlyverify588originalscalaroptima,736gates,validationpredictions,seed/ensemble predictions,exactfallback,oldhistories,allmetrics,diagnostics and72statistics. Poison nontraining inputs and postcutoff validation inputs at23cutoffs. Verify parent parameter byte identity, report/visual QA, then delivery.'
p['limitations']='All history previously viewed. This is retrospective testing of shared fitting with unchanged four-state validation support thresholds, not a new unseen training result. Parent pooled data may dilute state heterogeneity. Few qualified child validation cells limit coverage. Fixed sum-loss ridge has different per-observation strength when fitting groups differ. No baselinepromotion, PnL or deployment claims.'
p['budgets']=dict(source_files=11,old_files=5986,cutoffs=23,methods=4,seeds=3,states_per_family=4,new_policies=2,reused_head_records=1104,reused_scalar_fits=588,new_scalar_fits=0,new_neural_fits=0,gate_records=736,new_learned_seed_predictions=6528,new_total_model_rows=8704,total_model_rows=117504,total_ensemble_rows=44064,original_histories=25,total_histories=27,comparisons=72)
p['primary_comparisons_per_window']=[dict(history=h,candidate=m,reference_history=ref,reference=m,metric=metric) for family,h in p['policies'].items() for ref in [p['parent_policies'][family],'weekly_state_validated','rolling5_annual20'] for m in ['learned_vol_interaction','learned_order_extension','learned_order_offset'] for metric in ['direction_error','brier']]
put('protocol.json',json.dumps(p,indent=2,ensure_ascii=False))
put('README.md','''第46轮固定测试：复用第45轮趋势／波动两组校准参数，恢复原四状态分别验证启用。子状态训练至少10周、验证至少5周，Brier改善超过1e-12且正确数不减少。合并组原门控不参与新候选决定。

参数逐字节复用，不新增神经或标量拟合；原季度52／13成熟周、严格路由、年末重置和原集成算法保留。两个候选分别对照对应第45轮合并验证、原四状态季度及年度，预定72项探索性比较。全部历史已查看，没有盲测；原25条历史、5,986个旧文件保留。

运行 `research_v4/.venv_gpu/Scripts/python.exe -B -u research_v46/run46.py prepare contract fit validate score evaluate verify report`。图表和解读核对后单独运行 `delivery46.py`。所有11个源文件与协议在准备阶段冻结。
''')
print('R46 ten source files and protocol staged; report source still required before freeze.')

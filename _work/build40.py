from pathlib import Path
import calendar,json

project=Path('D:/ddos_v3');old=project/'research_v39';root=project/'research_v40'
assert not (root/'results/preparation_manifest.json').exists()
root.mkdir(exist_ok=True)
def replace(s,a,b):
    assert a in s,a
    return s.replace(a,b)
def source(name):return (old/f'{name}39.py').read_text(encoding='utf-8')
def write(name,s):(root/f'{name}40.py').write_text(s,encoding='utf-8')

s=source('common')
s=replace(s,"V38=PROJECT/'research_v38'","V39=PROJECT/'research_v39'")
s=replace(s,'str(V38));import common38','str(V39));import common39')
s=replace(s,'v37=previous_round.previous_round','v37=previous_round.v37')
s=replace(s,"POLICIES={'state_gated':'weekly_state_validated','state_direct':'weekly_state_ungated','global_gated':'weekly_global_validated','global_direct':'weekly_global_ungated'};NEW=list(POLICIES.values());FAMILIES=['state','global']","POLICIES={'state_gated':'monthly_state_validated','state_direct':'monthly_state_ungated'};NEW=list(POLICIES.values());FAMILIES=['state'];FAMILY_COMPONENTS=[('state',STATES)];QUARTER={'monthly_state_validated':'weekly_state_validated','monthly_state_ungated':'weekly_state_ungated'}")
s=replace(s,"paths=[V38/'protocol.json']+[V38/'results'/n for n in ['delivery_manifest.json','preparation_manifest.json','verification.json','结果解读与下一步.md','model_predictions.csv','ensemble_predictions.csv','ensemble_metrics.csv','routing.csv','weekly_context.csv','seed_correction_effects.csv']]","paths=[V39/'protocol.json']+[V39/'results'/n for n in ['delivery_manifest.json','preparation_manifest.json','verification.json','结果解读与下一步.md','model_predictions.csv','ensemble_predictions.csv','ensemble_metrics.csv','routing.csv','weekly_context.csv','source_seed_effects.csv','correction_heads.json','gate_decisions.csv','split_membership.csv','schedule.csv','weekly_signal_bank.csv','seed_routing.csv','weekly_policy_effects.csv']]")
s=s.replace('V38','V39').replace('5484','5554')
s=replace(s,"mode='annual_reset' if cutoff.endswith('12-31') else 'ready' if ready else 'insufficient_history'","mode='annual_reset' if cutoff.endswith('12-31') else 'q1_guard' if int(cutoff[5:7]) in (1,2) else 'ready' if ready else 'insufficient_history'")
s=replace(s,"[('state',STATES),('global',['all'])]",'FAMILY_COMPONENTS')
s += '''
def monthly_for(date):
    # The decision must be strictly earlier than the Friday signal, including month-end Fridays.
    return (pd.Timestamp(date)-pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')

def frames_equal_missing(a,b,**kwargs):
    # Only normalize truly undefined, entirely missing Brier columns; finite values retain their tolerances.
    a=a.copy();b=b.copy()
    for name in ['brier_difference','baseline_brier','candidate_brier']:
        if name in a and name in b and a[name].isna().all() and b[name].isna().all():
            a[name]=a[name].astype(float);b[name]=b[name].astype(float)
    pd.testing.assert_frame_equal(a,b,**kwargs)
'''
write('common',s)

s=source('prepare').replace('common39','common40').replace('V38','V39').replace('5484','5554').replace('R39','R40')
s=replace(s,"('routing.csv','routing.csv')","('routing.csv','quarterly_routing.csv')")
s=replace(s,"('seed_correction_effects.csv','source_seed_effects.csv')","('source_seed_effects.csv','source_seed_effects.csv'),('correction_heads.json','quarterly_heads.json'),('gate_decisions.csv','quarterly_gates.csv'),('split_membership.csv','quarterly_members.csv'),('schedule.csv','quarterly_schedule.csv'),('seed_routing.csv','quarterly_seed_routing.csv'),('weekly_policy_effects.csv','quarterly_weekly_effects.csv')")
s=replace(s,"    assert read(V39/'results/verification.json')", "    route=csv('quarterly_routing').rename(columns={'head_cutoff':'quarter_cutoff'});route['head_cutoff']=route.date.map(monthly_for);p=OUT/'routing.csv';route.to_csv(p,index=False);files.append(p)\n    assert set(route.head_cutoff).issubset(cfg()['decision_dates']) and route.head_cutoff.lt(route.date).all()\n    assert read(V39/'results/verification.json')")
write('prepare',s)

s=source('contract').replace('common39','common40')
s=replace(s," and h[4]['fit_eligible']",'')
s=replace(s,"(r.encoder_cutoff,r.head_cutoff)==independent_route(r.date)","(r.encoder_cutoff,r.quarter_cutoff)==independent_route(r.date);assert r.head_cutoff==monthly_for(r.date)<r.date")
s=replace(s,"    return ['purge_unmatured_adjacent_training_label'", "    assert monthly_for('2026-07-31')=='2026-06-30' and monthly_for('2026-08-01')=='2026-07-31' and monthly_for('2024-03-01')=='2024-02-29'\n    assert split(p,'2026-01-31')[2]['mode']=='q1_guard' and split(p,'2026-02-28')[2]['mode']=='q1_guard'\n    frames_equal_missing(pd.DataFrame({'brier_difference':[None]}),pd.DataFrame({'brier_difference':[np.nan]}),check_dtype=False)\n    for aa,bb in [(None,0.),(.1,.2)]:\n        try:frames_equal_missing(pd.DataFrame({'brier_difference':[aa]}),pd.DataFrame({'brier_difference':[bb]}),check_dtype=False)\n        except AssertionError:pass\n        else:raise AssertionError('Missing compatibility masked a substantive difference')\n    return ['month_end_friday_strict_cutoff','leap_month_end','January_February_Q1guard','missing_only_normalization','purge_unmatured_adjacent_training_label'")
s=s.replace('ready_quarters','ready_months')
write('contract',s)

s=source('fit').replace('common39','common40')
write('fit',s)
s=source('validate').replace('common39','common40')
s=replace(s,"[('state',STATES),('global',['all'])]",'FAMILY_COMPONENTS').replace('460','1088')
write('validate',s)
s=source('score').replace('common39','common40')
s=s.replace('17408','8704').replace('87040','95744').replace('32640','35904').replace('13056','6528').replace('Four fixed weekly','Two fixed monthly').replace('old16histories','old20histories')
write('score',s)

s=source('evaluate').replace('common39','common40').replace('4352','2176').replace('quarter_outcomes','month_outcomes').replace('84','60')
s=replace(s,"check_frozen();check_phase('scoring');", "check_frozen();check_phase('scoring');")
s=replace(s,"files=[]\n    for name,g in tables.items()", "files=[]\n    from diagnostics40 import diagnostics\n    tables.update(diagnostics(csv('ensemble_predictions'),tables['weekly_policy_effects']))\n    for name,g in tables.items()")
write('evaluate',s)

s=source('verify').replace('common39','common40').replace('contract39','contract40').replace('validate39','validate40').replace('ready_quarters','ready_months').replace('5484','5554').replace('84','60').replace('13056','6528').replace('quarter_outcomes','month_outcomes').replace('R39','R40')
s=replace(s,"[('state',STATES),('global',['all'])]",'FAMILY_COMPONENTS')
s=replace(s,"mode='annual_reset' if c.cutoff.endswith('12-31') else 'ready' if len(train)==52 and len(val)==13 else 'insufficient_history'","mode='annual_reset' if c.cutoff.endswith('12-31') else 'q1_guard' if int(c.cutoff[5:7]) in (1,2) else 'ready' if len(train)==52 and len(val)==13 else 'insufficient_history'")
s=replace(s,"pd.testing.assert_frame_equal(a,expected,check_dtype=False,atol=1e-14,rtol=0)","frames_equal_missing(a,expected,check_dtype=False,atol=1e-14,rtol=0)")
s=replace(s,"r.cutoff==source.head_cutoff<r.date", "r.cutoff==monthly_for(r.date)<r.date")
s=replace(s,"check_inference();assert old_evidence()", "check_inference();from audit_cadence40 import audit_cadence\n    cadence=audit_cadence();assert old_evidence()")
s=replace(s,"**fit,**pred,artifacts=", "**fit,**pred,**cadence,artifacts=")
write('verify',s)
write('run',source('run').replace("{phase}39.py","{phase}40.py"))
write('delivery',source('delivery').replace('common39','common40'))

c=json.loads((old/'protocol.json').read_text(encoding='utf-8'))
for key in ['primary_comparisons_per_window','budgets','fitting','validation','controls','scoring','sequencing','inference','split']:
    c.pop(key,None)
c.update(version=40,experiment='Monthly versus quarterly refit and held-out validation of mature-week state probability corrections')
c['decision_dates']=[f'{y}-{m:02d}-{calendar.monthrange(y,m)[1]}' for y in range(2020,2027) for m in range(1,13) if (y,m)>=(2020,12) and (y,m)<=(2026,7)]
assert len(c['decision_dates'])==68
c['policies']={'state_gated':'monthly_state_validated','state_direct':'monthly_state_ungated'}
c['split']='Same R39 purged52train/13validation count rule at each monthly calendar end. Decision strictly earlier than the signal: a Friday on month-end still uses previous month-end. Preserve archived annual neural/head models, signal-time annual-relative state labels, maturity and original weekly panel. Refit and revalidate together; this is not validation-only cadence. No backfill before2021. Same-quarter-end candidate inputs, offsets and gates must match R39 exactly.'
c['fitting']='State family only; four methods x three fixed seeds. Exactly R39 sum Bernoulli logloss plus20*d²/2, capabs.5,60bisection iterations, minstate train10 of52older mature weeks. No neural or feature fits, no hyperparameter search, no cumulative offsets.'
c['validation']='Frozen offsets first, then last13mature weekly validation signals; each state requires train10 and validation5, Brier decrease>1e-12 and ensemblecorrect count no lower. Three seeds equally averaged for gate and reporting. Strict>.5up. No refit with validation labels. Insufficient support or rejected gate gives exactannual probability.'
c['controls']='Two new monthly policies, validated and identical52train parameters ungated. Compare with immutable R39 quarterly state validated/ungated plus annual natural20baseline. December31 annual reset and January/February decisions both suppress correction to retain allQ1fallback. No new global family; all20old histories remain. Native_mse and training_frequency copy annual controls exactly.'
c['scoring']='Latest month-end strictly before each original Friday signal routes fixed offsets by original signal state. Decisioninputs jointmature<=cutoff, training labelmature<earliestvalidation signal. Allgates frozen before subsequent historicalscoring. Samequarter dates and firstmonth forecast subset independently match R39. Moreupdates change membersandcoefficients as well as gates; distinguish chronological policychange from pure stopping-only effect.'
c['sequencing']='Freeze12sources,protocol,inputs and5554oldfiles; contract; fit; validate; score once; evaluate60fixed comparisons and cadence diagnostics; independent verification; report and delivery. No after-score tuning. Initial nullcolumn compatibility is built into frozen verifier, only entirely undefined Briercolumns, unchanged numericaltolerances.'
c['inference']='60exploratory comparisons: monthlygated vsquarterlygated, monthlydirect vsquarterlydirect, monthlygated vsmonthlydirect, eachmonthly vsannual; each3primarymethods x2losses x2fixedwindows. Circular8week bootstrap10000seed20260910, centered2sidedp andHolm across60. No correction forprioradaptive rounds. R18diagnostic. Allsixyears,seeds,states,emptyfuturecells retained. No newblindholdout.'
c['diagnostics']='Fixed monthly-versus-quarterly weekly cases, Brierdifferences, probabilitychange coverage; annual-relative activation groups both,neither,monthlyonly,quarterlyonly. Gate transition register everycutoff/method/state, including suppressedQ1 andemptyfuturecells; extra-month stops/restarts compared with quarter gate. Acceptedfuture Brierworsening is retrospective, never an inputto gate. Counts of months/cells are not independenttrials or a statistical false-positive rate. More frequenttest opportunities may increasenoisyacceptances.'
c['budgets']=dict(old_files=5554,new_neural_fits=0,new_transform_fits=0,new_feature_inference=0,weeks=272,cutoffs=68,new_policies=2,maximum_scalar_fits=2400,head_cells=3264,gate_cells=1088,new_learned_seed_rows=6528,new_total_seed_rows_with_copied_controls=8704,total_model_rows=95744,total_ensemble_rows=35904,comparisons=60)
p=c['policies'];pairs=[]
for a,b in [(p['state_gated'],'weekly_state_validated'),(p['state_direct'],'weekly_state_ungated'),(p['state_gated'],p['state_direct']),(p['state_gated'],'rolling5_annual20'),(p['state_direct'],'rolling5_annual20')]:
    for m in ['learned_vol_interaction','learned_order_extension','learned_order_offset']:
        for loss in ['direction_error','brier']:pairs.append(dict(history=a,candidate=m,reference_history=b,reference=m,metric=loss))
c['primary_comparisons_per_window']=pairs
(root/'protocol.json').write_text(json.dumps(c,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('R40 draft created. Need diagnostics, independent cadence audit and report before freeze. No fits or forecasts yet.')

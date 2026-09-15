from pathlib import Path
import json
p=Path('D:/ddos_v3');r=p/'research_v41';r.mkdir(exist_ok=True);assert not (r/'results/preparation_manifest.json').exists()
old=p/'research_v40';c0=json.loads((old/'protocol.json').read_text(encoding='utf-8'))
c={k:c0[k] for k in ['label_end','windows','seeds','report_python','states','state_definition','settings']}
c.update(version=41,experiment='Read-only single-week influence, validation reuse and support attrition audit',
    scope='No neural fit, scalar fit, parameter change, policy rerouting, forecast, backtest or new significance test. Read frozenR39quarterly andR40monthly state-calibration validation probabilities, decisions, members and previouslyscored futureoutcomes. Preserve22histories and all5633oldfiles. R18diagnostic; R19/R23/R25 primary.',
    sensitivity='For every cadence/cutoff/method/state with ready mode, train_n>=10 and original validation_n>=5, remove each single validationweek in turn with allthree savedseed correctionparameters fixed. Compare Brier and directioncorrect differences on remainingweeks. Report metric-only acceptance with originalsupport eligibility held fixed, separately from fullgate acceptance which enforcesremaining_n>=5. Originaln5 fullgate failure is a mechanicalsupporteffect, not proof ofsingle-observation statisticaldominance. No retune, refit or hypotheticalfuture prediction. Every originaleligible rejectedcell is included too.',
    contributions='Mean saved3seed annual/candidateprobabilities, then perweek candidate-minusbaseline squaredloss and correctcountdifference. Sum/mean these to replayoriginalgate. Positive helpful Briercontribution means baseline_loss-candidate_loss>0; maximumhelpful share denominator is sumpositivehelpful contributions, not unstable netgain. Deleteone is a sensitivity diagnostic, not a jackknife confidenceinterval or robustnessqualification.',
    overlap='Within eachcadence, compare every adjacent pair ofdecisioncutoffs and bothroles(training,validation), for allsignals and each4signal-time annual-relative states. Store retained,lost,added rows and exact identities n_current=n_previous-lost+added, overlap/current, Jaccard. Classify dropped rows as olderthan currentrolewindowfirstsignal. Preserve mode and resetperiods. Maincomparableoverlap summary uses adjacentpairs where bothcutoffs ready; allpairs retained.',
    transitions='Join statevalidationflows to every adjacentgatepair. Distinguish accepted->rejected by currentreason, ready-both transitions versus forcedannual/Q1fallback, and supportcrossing previous_n>=5/current_n<5. Record lossesandnewcomers; do not infer economicregimefailure merely from supportdropout. Currentrefitparametersmaychange, so this is accounting, not a causal decomposition ofallgatechanges.',
    runs='For every maximal consecutive accepted run per cadence/method/state, includinglengthone, count decisions,totalvalidationrowuses,uniquevalidationrowids,repeateduses and uses/unique. Consecutivepasses can sharelabels andchangeparameters; reuse is not an effectiveindependent sample-size estimator. Every acceptedcell belongs to exactlyone run; states preserve originalyear-relativelabels.',
    future_role='Attach archived direct/ungated candidate outcome cell n,Brier_vsannual,recoveries/regressions for alloriginalgates; accepted cells equal correspondingvalidated policy outcomes. Emptyfuturecells retained. Summarize accepted metricfragile/stable cohorts by methodandcadence with counts and weekweightedpreviouslyscored Brier. This retrospectiveassociation never changesa gate or definesa newforecastpolicy; repetitions acrossmodels,months andfrequencies are not independentreplications.',
    summary_windows=[dict(name='decision_2021_2023',start='2021-01-01',end='2023-12-31'),dict(name='decision_2024_2026',start='2024-01-01',end='2026-07-31'),dict(name='decision_all_2020_2026',start='2020-12-31',end='2026-07-31')],
    limitations='Allhistoricaloutcomes previouslyseen; no newblindholdout. Singledeletion can expose fragileconditions but neither provesbadfuture performance nor qualifiesa newgate. No removalchosen fornewpredictions. Stability/cohorttables descriptive, no newpvalues. Keep originalR25recent73/12558.4%, natural20annualR25 71/12556.8%,R39quarterR23 73/12558.4%, andR37year2025gain separate.',
    budgets=dict(old_files=5633,source_files=8,cadences=2,quarterly_cutoffs=23,monthly_cutoffs=68,gate_cells=1456,role_flow_cells=890,gate_transition_cells=1424,maximum_leave_one_out_cases=18928,new_fits=0,new_predictions=0,new_pvalues=0),
    sequencing='Freeze8sources,protocol andallinputs before diagnostics. Preparecopies,contractandvalidationcontributions,diagnostics,independentarithmetic/membership/run/cohortverification,report/visual/humanfacts,delivery. No afterresult tuning.')
(r/'protocol.json').write_text(json.dumps(c,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
s=(old/'common40.py').read_text(encoding='utf-8')
start=s.index('def manifest(');end=s.index('def split(')
shared=s[start:end]
prefix='''"""Read-only audit of archived state correction validation evidence."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';V40=PROJECT/'research_v40';V39=PROJECT/'research_v39'
sys.path.insert(0,str(V40));import common40 as previous_round
prior=previous_round.prior;np=prior.np;pd=prior.pd;read=prior.read;save=prior.save;sha=prior.sha;data=prior.data
STATES=previous_round.STATES;LEARNED=previous_round.LEARNED;PRIMARY=previous_round.PRIMARY;ANNUAL=previous_round.ANNUAL;CADENCES=['monthly','quarterly']
HISTORY={'monthly':('monthly_state_validated','monthly_state_ungated'),'quarterly':('weekly_state_validated','weekly_state_ungated')}
COPIES=[(V40/'results'/f'{n}.csv',f'{n}.csv') for n in ['model_predictions','ensemble_predictions','ensemble_metrics','seed_metrics','state_metrics','weekly_signal_bank']]
for source,target in [('gate_decisions','gates'),('schedule','schedule'),('split_membership','members'),('validation_predictions','validation_predictions'),('month_outcomes','outcomes')]:COPIES.append((V40/'results'/f'{source}.csv',f'monthly_{target}.csv'))
COPIES.append((V40/'results/correction_heads.json','monthly_heads.json'))
for source,target in [('quarterly_gates','gates'),('quarterly_schedule','schedule'),('quarterly_members','members')]:COPIES.append((V40/'results'/f'{source}.csv',f'quarterly_{target}.csv'))
COPIES.extend([(V40/'results/quarterly_heads.json','quarterly_heads.json'),(V39/'results/validation_predictions.csv','quarterly_validation_predictions.csv'),(V39/'results/quarter_outcomes.csv','quarterly_outcomes.csv')])
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def source_hashes():
    x=previous_round.source_hashes();x.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.glob('*.py')});return x
def input_hashes():
    x=previous_round.input_hashes();paths=[a for a,b in COPIES]+[V40/'protocol.json']+[V40/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','verification.json','结果解读与下一步.md']];x.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return x
def old_evidence():
    x=dict(read(V40/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V40/'results/delivery_manifest.json')['files'].items():x[str((V40/n).relative_to(PROJECT))]=d
    p=V40/'results/delivery_manifest.json';x[str(p.relative_to(PROJECT))]=sha(p);assert len(x)==5633
    for n,d in x.items():assert sha(PROJECT/n)==d,n
    return x
'''
tail='''
def cell_gate(mode,train_n,n,brier_difference,correct_difference):
    reason=mode if mode!='ready' else 'insufficient_state_train' if train_n<10 else 'insufficient_validation' if n<5 else 'brier_not_better' if not brier_difference < -1e-12 else 'direction_worse' if correct_difference<0 else 'accepted'
    return reason=='accepted',reason
def eq(a,b,tol=1e-12):
    if a is None or pd.isna(a):assert b is None or pd.isna(b),(a,b)
    elif isinstance(a,(str,bool)):assert a==b,(a,b)
    else:assert abs(a-b)<tol,(a,b)
'''
(r/'common41.py').write_text(prefix+shared+tail,encoding='utf-8')
(r/'prepare41.py').write_text('''from common41 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run);files=[OUT/'initial_freeze.json']
    assert read(V40/'results/verification.json')['status']=='PASS'
    for src,name in COPIES:
        dest=OUT/name;dest.write_bytes(src.read_bytes());files.append(dest)
    finish(run,files,old_files=5633,new_fits=0,new_predictions=0);print('R41 read-only inputs,8sources,protocol and5633oldfiles frozen.',flush=True)
if __name__=='__main__':main()
''',encoding='utf-8')
s=(old/'run40.py').read_text(encoding='utf-8').replace("['prepare','contract','fit','validate','score','evaluate','verify','report']","['prepare','contract','diagnose','verify','report']").replace('{phase}40.py','{phase}41.py')
(r/'run41.py').write_text(s,encoding='utf-8')
s=(old/'delivery40.py').read_text(encoding='utf-8').replace('common40','common41').replace("['fitting','validation','scoring','evaluation','report']","['diagnosis','report']")
(r/'delivery41.py').write_text(s,encoding='utf-8')
print('R41 draft scaffolding and protocol created; no diagnostic results calculated.')

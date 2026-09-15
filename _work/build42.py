from pathlib import Path
import json
p=Path('D:/ddos_v3');r=p/'research_v42';r.mkdir(exist_ok=True);assert not (r/'results/preparation_manifest.json').exists();v40=p/'research_v40';v41=p/'research_v41'
c0=json.loads((v40/'protocol.json').read_text(encoding='utf-8'));c={k:c0[k] for k in ['label_end','windows','seeds','report_python','states','state_definition','settings','decision_dates']}
c.update(version=42,experiment='Bounded retention of previously accepted state correction under validation support shortage',
    policy='monthly_state_retained',references=['monthly_state_validated','weekly_state_validated','rolling5_annual20'],
    scope='One new historicalrouting policy over frozen R40monthly candidate heads and gates. No neural fit, scalar refit, feature inference, cadence/state/window/lambda/cap/threshold search. Original22histories and5691oldfiles preserved. Fourlearnedmethods,3fixedseeds;R18diagnostic,3primarymethods.',
    retention='A formal R40accepted gate saves its current3seed stateoffsets and sourcecutoff, with immutable expiry equal to the firstcalendarquarter-end strictly after sourcecutoff. It replaces any previous record. Only a current ready gate with training_n>=10, validation_n<5 and originalreason insufficient_validation may reuse an unexpired record from the same annualmodel/stateboundary year. Carrying does not update sourcecutoff or expiry. All other rejections clear the record, so later shortages cannot resurrect a rejected/expired record. Annualreset and January/Februaryguard clear everything. Enough currentvalidation with Brier_not_better or direction_worse clears rather than carries.',
    boundaries='A record covers signals strictly after its sourcecutoff and on or before expiry. At a decisioncutoff D>=expiry the oldrecord is expired for subsequent signals. A new accepted gate at D may save a newrecord. A Friday exactly on a month/quarterend uses latestmonthlydecision strictlybefore its signaldate, thus can still use oldrecord through inclusiveexpiryday; nextday cannot. Expiry anchored to lastformalacceptance, never extended by carry. Sourceannualyear must match signalencoder and currentdecision year. Firstquarter fallback remains exactannual.',
    sequencing='Freeze10sources,protocol,inputs andoldfiles; source/maturity/expirycontract; buildandfreeze all1088retentiondecisioncells using originalgates only; subsequentlyapply selected originalheadcoefficients to originalannual logits on272weeklysignals once; evaluate36comparisons; independently verify state machine, informationisolation, coefficients, predictions,metrics andstatistics; reportanddelivery. No futureoutcomes drive retention decisions.',
    controls='Compare fixednewcandidate with immutable R40monthly immediatefallback, R39quarterlyvalidated and natural20annual. R40monthlydirect remains descriptive. Original native_mse andtraining_frequency copiedfromannual exactly. Sourceheadsandgates never edited. The rule does not prevent wellvalidated but laterharmfulfreshacceptances; it specifically tests handling of temporaryvalidationcountshortage.',
    inference='36exploratory comparisons: newcandidate vs3references x3primarymethods x2losses(directionerror,Brier) x2fixedwindows. Circular8weekbootstrap10000seed20260910, centered2sidedp, Holm across36. No adjustment forprioradaptive research; no newblindholdout. Retention-triggering historicalcases alreadyexamined inR40/R41.',
    diagnostics='Record everydecision previousrecord,source,expiry,actionandresetcause; everyseed selectedcoefficientsource; weeklydifferences versusmonthly andquarterly, retained signalcoverage and sourceage; futureemptydecisioncells retained. Distinguish carried statecells, coveredweeks, actualprobabilitychanges anddirectionchanges. Include all6years,pooled,seeds,states,R18diagnostic.',
    limitations='Previouslyviewed272outcomes; this is newhistoricalrouting, not newblindtest or liveprediction. Retainedparameters can be stale; countshortage is notproof oldcorrection remainsvalid. The maxquarterhorizon was chosen frompriorresearch proposal, not optimized here. OldR25recent73/12558.4%, natural20annualR25 71/12556.8%,R39quarterR23 73/12558.4%,R37year2025gain and2026regressions kept separate.',
    budgets=dict(old_files=5691,source_files=10,new_policies=1,weeks=272,cutoffs=68,decision_cells=1088,new_neural_fits=0,new_scalar_fits=0,new_feature_inference=0,new_learned_seed_rows=3264,new_total_model_rows=4352,total_model_rows=100096,total_ensemble_rows=37536,comparisons=36))
pairs=[]
for ref in c['references']:
    for m in ['learned_vol_interaction','learned_order_extension','learned_order_offset']:
        for loss in ['direction_error','brier']:pairs.append(dict(history=c['policy'],candidate=m,reference_history=ref,reference=m,metric=loss))
c['primary_comparisons_per_window']=pairs
(r/'protocol.json').write_text(json.dumps(c,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
old=(v41/'common41.py').read_text(encoding='utf-8');shared=old[old.index('def manifest('):old.index('def cell_gate(')]
prefix='''"""Bounded historical retention with immutable source candidates and gates."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';V41=PROJECT/'research_v41';V40=PROJECT/'research_v40'
sys.path.insert(0,str(V41));import common41 as previous_round
v40=previous_round.previous_round;v37=v40.v37;prior=v40.prior;np=prior.np;pd=prior.pd;read=prior.read;save=prior.save;sha=prior.sha;data=prior.data;metric=prior.metric;statistics=prior.statistics;probability=prior.probability
STATES=v40.STATES;LEARNED=v40.LEARNED;PRIMARY=v40.PRIMARY;ANNUAL=v40.ANNUAL;MONTHLY='monthly_state_validated';QUARTER='weekly_state_validated';NEW=['monthly_state_retained'];REFS=[MONTHLY,QUARTER,ANNUAL];REPORT_HIST=[ANNUAL,QUARTER,MONTHLY]+NEW
COPIES=[(V41/'results'/f'{s}.csv',f'{d}.csv') for s,d in [('model_predictions','baseline_model_predictions'),('ensemble_predictions','baseline_ensemble_predictions'),('ensemble_metrics','baseline_metrics'),('weekly_signal_bank','weekly_signal_bank'),('monthly_gates','gate_decisions'),('monthly_schedule','schedule'),('monthly_members','split_membership'),('monthly_validation_predictions','validation_predictions')]]
COPIES.append((V41/'results/monthly_heads.json','correction_heads.json'))
COPIES.extend([(V40/'results'/f'{s}.csv',f'{d}.csv') for s,d in [('routing','routing'),('weekly_context','weekly_context'),('seed_routing','original_monthly_seed_routing'),('weekly_policy_effects','original_monthly_weekly_effects')]])
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def source_hashes():
    r=previous_round.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.glob('*.py')});return r
def input_hashes():
    r=previous_round.input_hashes();paths=[a for a,b in COPIES]+[V41/'protocol.json']+[V41/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','verification.json','结果解读与下一步.md']];r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V41/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V41/'results/delivery_manifest.json')['files'].items():r[str((V41/n).relative_to(PROJECT))]=d
    p=V41/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==5691
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
'''
tail='''
def expiry_after(cutoff):return (pd.Timestamp(cutoff)+pd.offsets.QuarterEnd(1)).strftime('%Y-%m-%d')
def annual_for(cutoff):return f'{int(cutoff[:4])-1}-12-31'
def eq(a,b,tol=1e-12):
    if a is None or pd.isna(a):assert b is None or pd.isna(b),(a,b)
    elif isinstance(a,(str,bool)):assert a==b,(a,b)
    else:assert abs(a-b)<tol,(a,b)
def usable(source_cutoff,expiry,signal_date):return isinstance(source_cutoff,str) and source_cutoff<signal_date<=expiry and source_cutoff[:4]==signal_date[:4]
def retention_decisions(gates,dates):
    lookup=gates[gates.family.eq('state')].set_index(['cutoff','method','component']);records=[]
    for method in LEARNED:
        for state in STATES:
            stored=None
            for cutoff in dates:
                g=lookup.loc[(cutoff,method,state)];old=dict(stored) if stored else None;expired=bool(stored and cutoff>=stored['expiry'])
                if expired:stored=None
                if g.accepted:
                    assert g['mode']=='ready' and g.training_n>=10 and g.validation_n>=5
                    stored=dict(source_cutoff=cutoff,expiry=expiry_after(cutoff),source_encoder_cutoff=annual_for(cutoff));action='fresh';reason='accepted_current'
                elif g['mode']=='ready' and g.reason=='insufficient_validation' and g.training_n>=10 and g.validation_n<5:
                    if stored and stored['source_encoder_cutoff']==annual_for(cutoff):action='carry';reason='retained_validation_shortage'
                    else:stored=None;action='fallback';reason='retention_expired' if expired else 'no_valid_saved_acceptance'
                else:stored=None;action='fallback';reason='cleared_'+g.reason
                records.append(dict(cutoff=cutoff,method=method,state=state,mode=g['mode'],original_accepted=bool(g.accepted),original_reason=g.reason,training_n=int(g.training_n),validation_n=int(g.validation_n),previous_source_cutoff=old['source_cutoff'] if old else None,previous_expiry=old['expiry'] if old else None,previous_expired=expired,action=action,retention_reason=reason,source_cutoff=stored['source_cutoff'] if stored else None,expiry=stored['expiry'] if stored else None,source_encoder_cutoff=stored['source_encoder_cutoff'] if stored else None))
    return pd.DataFrame(records)
'''
(r/'common42.py').write_text(prefix+shared+tail,encoding='utf-8')
(r/'prepare42.py').write_text('''from common42 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run);files=[OUT/'initial_freeze.json'];assert read(V41/'results/verification.json')['status']=='PASS'
    for src,name in COPIES:
        p=OUT/name;p.write_bytes(src.read_bytes());files.append(p)
    finish(run,files,old_files=5691,new_fits=0,new_predictions=0);print('R42 sources,protocol,inputs and5691oldfiles frozen.',flush=True)
if __name__=='__main__':main()
''',encoding='utf-8')
s=(v41/'run41.py').read_text(encoding='utf-8').replace("['prepare','contract','diagnose','verify','report']","['prepare','contract','route','score','evaluate','verify','report']").replace('{phase}41.py','{phase}42.py');(r/'run42.py').write_text(s,encoding='utf-8')
s=(v41/'delivery41.py').read_text(encoding='utf-8').replace('common41','common42').replace("['diagnosis','report']","['routing','scoring','evaluation','report']");(r/'delivery42.py').write_text(s,encoding='utf-8')
print('R42 draft protocol and scaffolding prepared; no retention decisions or new forecasts generated.')

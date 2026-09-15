from common31 import *
import math

def coverage_checks():
    obs,_,_=data();expected=[]
    for cutoff in cfg()['decision_dates']:
        year=int(cutoff[:4])+(1 if cutoff.endswith('12-31') else 0);end=min(f'{year}-12-31',cfg()['label_end'])
        for i,r in obs.iterrows():
            if cutoff<r.date<=end and r.weekday==4 and isinstance(r.joint_completed,str) and r.joint_completed<=cfg()['label_end']:expected.append((cutoff,i,r.date,r.joint_completed))
    cover=csv('coverage');pd.testing.assert_frame_equal(pd.DataFrame(expected,columns=cover.columns),cover,check_dtype=False);unique=csv('universal_model_predictions');pd.testing.assert_frame_equal(unique,archive_predictions(),check_dtype=False,check_exact=True)
    keys=['cutoff','row_index','date','joint_completed'];actual=unique[keys].drop_duplicates().sort_values(['cutoff','row_index']).reset_index(drop=True);pd.testing.assert_frame_equal(actual,cover,check_dtype=False);assert len(cover)==671 and len(unique)==10736
    assert unique.groupby(['cutoff','date']).size().eq(16).all() and not unique.duplicated(['cutoff','date','method','seed']).any()
    assert read(V30/'results/verification.json')['status']=='PASS' and read(V28/'results/verification.json')['status']=='PASS';assert len(read(OUT/'source_models.json'))==69
    for r in read(OUT/'source_models.json'):assert sha(PROJECT/r['project_file'])==r['sha256']
    return dict(coverage_model_weeks=671,universal_model_rows=10736,canonical_weeks=272,source_networks=69)

def independent_simulation(feedback):
    source={(r.cutoff,r.date):r._asdict() for r in feedback.itertuples(index=False)};dates=sorted(feedback.date.unique());events=[];ledger=[];members=[];inc=cha=start=None
    def emit(lo,hi,old,new):
        for date in dates:
            if lo<date<=hi:
                a=source[(old,date)];b=source[(new,date)];ledger.append(dict(row_index=a['row_index'],date=date,joint_completed=a['joint_completed'],trial_start=lo,incumbent_cutoff=old,challenger_cutoff=new,incumbent_probability=a['prediction_probability'],challenger_probability=b['prediction_probability'],actual_up=a['actual_up']))
    for decision in cfg()['decision_dates']:
        if start is not None:emit(start,decision,inc,cha)
        visible=[r for r in ledger if r['trial_start']==start and r['incumbent_cutoff']==inc and r['challenger_cutoff']==cha and r['joint_completed']<=decision];mandatory=decision.endswith('12-31');n=len(visible);ready=False;delta=np.nan;gain=0;promote=False
        if mandatory:reason='annual';selected=decision
        else:
            losses=[];errors=[]
            for r in visible:
                a=r['incumbent_probability'];b=r['challenger_probability'];y=r['actual_up'];loss=(b-y)**2-(a-y)**2;err=int((b>.5)!=y)-int((a>.5)!=y);losses.append(loss);errors.append(err);members.append(dict(decision_cutoff=decision,**r,challenger_minus_incumbent_brier=loss,challenger_minus_incumbent_error=err))
            delta=math.fsum(losses)/n if n else np.nan;gain=-sum(errors);ready=inc!=cha and n>=8;promote=bool(ready and delta<0 and gain>0);reason='same_model' if inc==cha else 'warmup' if not ready else 'promote' if promote else 'hold';selected=cha if promote else inc
        events.append(dict(cutoff=decision,trial_start=start or '',incumbent_cutoff=inc or '',evaluated_challenger_cutoff=cha or '',selected_model_cutoff=selected,next_challenger_cutoff=decision,mandatory=mandatory,paired_mature_weeks=n,ready=ready,challenger_minus_incumbent_brier=delta,challenger_correct_gain=gain,promote=promote,reason=reason));inc=selected;cha=decision;start=decision
    emit(start,cfg()['label_end'],inc,cha)
    return pd.DataFrame(events),pd.DataFrame(ledger,columns=PAIR_COLUMNS),pd.DataFrame(members,columns=['decision_cutoff']+PAIR_COLUMNS+['challenger_minus_incumbent_brier','challenger_minus_incumbent_error'])

def synthetic_cases():
    dates=pd.date_range('2021-04-02',periods=8,freq='W-FRI');start='2021-03-31';inc='2020-12-31';cha=start;decision='2021-06-30'
    base=pd.DataFrame(dict(row_index=np.arange(8),date=dates.strftime('%Y-%m-%d'),joint_completed=(dates+pd.Timedelta(days=10)).strftime('%Y-%m-%d'),trial_start=start,incumbent_cutoff=inc,challenger_cutoff=cha,incumbent_probability=.75,challenger_probability=.25,actual_up=0))[PAIR_COLUMNS];cases=[]
    def test(b):return decide(b,start,inc,cha,decision)[0]
    assert test(base)['promote'];cases.append('both_strictly_better_promotes')
    assert not test(base.iloc[:7])['ready'];cases.append('seven_pairs_hold')
    b=base.copy();b['incumbent_probability']=.25;b['challenger_probability']=.125;assert not test(b)['promote'];cases.append('brier_better_direction_tie_hold')
    b=base.copy();b['incumbent_probability']=[.75,0,0,0]*2;b['challenger_probability']=.375;r=test(b);assert r['challenger_minus_incumbent_brier']==0 and r['challenger_correct_gain']==2 and not r['promote'];cases.append('exact_brier_tie_hold')
    b=base.copy();b['incumbent_probability']=[.51]*4+[.01]*4;b['challenger_probability']=.49;r=test(b);assert r['challenger_correct_gain']>0 and r['challenger_minus_incumbent_brier']>0 and not r['promote'];cases.append('direction_better_brier_worse_hold')
    b=base.copy();b['incumbent_probability']=.49;b['challenger_probability']=[.51]+[0]*7;r=test(b);assert r['challenger_correct_gain']<0 and r['challenger_minus_incumbent_brier']<0 and not r['promote'];cases.append('brier_better_direction_worse_hold')
    b=base.copy();b['challenger_probability']=b.incumbent_probability;assert not test(b)['promote'];cases.append('identical_forecasts_hold')
    b=base.copy();b['incumbent_cutoff']=cha;assert not decide(b,start,cha,cha,decision)[0]['ready'];cases.append('same_model_not_eligible')
    b=base.copy();b.loc[7,'joint_completed']='2021-07-01';assert not test(visible_pairs(b,start,inc,cha,decision))['ready'];cases.append('immature_pair_excluded')
    b.loc[7,'joint_completed']=decision;assert test(visible_pairs(b,start,inc,cha,decision))['promote'];cases.append('maturity_equality_included')
    b=pd.concat([base,base.assign(trial_start='2020-12-31'),base.assign(challenger_cutoff='2020-09-30')],ignore_index=True);pd.testing.assert_frame_equal(visible_pairs(b,start,inc,cha,decision).reset_index(drop=True),base);cases.append('stale_trial_and_other_challenger_excluded')
    b=base.copy();b.loc[0,'date']=start;assert len(visible_pairs(b,start,inc,cha,decision))==7;cases.append('signal_must_follow_trial_start')
    b=base.copy();b['challenger_probability']=.5;assert test(b)['promote'];cases.append('probability_half_is_down')
    # Use calendar metadata, never historical returns or probabilities.
    obs,_,_=data();cover=previous_round.coverage(obs);f=cover.copy();f['prediction_probability']=[.75 if c.endswith('12-31') else .25 if c.endswith('03-31') else .125 for c in f.cutoff];f['frequency_probability']=.5;f['actual_up']=0;f=f[previous_round.FEEDBACK_COLUMNS]
    e,l,m=simulate(f);a,b,c=independent_simulation(f)
    for x,y in [(e,a),(l,b),(m,c)]:pd.testing.assert_frame_equal(x,y,check_dtype=False,atol=1e-14,rtol=0)
    assert e[e.cutoff.eq('2021-06-30')].selected_model_cutoff.iloc[0]=='2021-03-31';assert l[l.date.eq('2023-06-30')].incumbent_cutoff.iloc[0]=='2022-12-31';assert e[e.mandatory].selected_model_cutoff.tolist()==e[e.mandatory].cutoff.tolist();cases += ['promote_tested_model_not_new_launch','decision_day_signal_uses_incumbent','annual_reset_overrides_trial']
    return cases

def audit_policy(feedback,events,ledger,members):
    e,l,m=independent_simulation(feedback);events=events.fillna({'trial_start':'','incumbent_cutoff':'','evaluated_challenger_cutoff':''})
    for a,b in [(events,e),(ledger,l),(members,m)]:pd.testing.assert_frame_equal(a,b,check_dtype=False,atol=1e-13,rtol=0)
    audits=[]
    for e in events[~events.mandatory].itertuples():
        bad=feedback.copy();future=bad.joint_completed.gt(e.cutoff);bad.loc[future,'actual_up']=1-bad.loc[future,'actual_up'];bad.loc[future,'prediction_probability']=.999
        a,_,am=simulate(bad,stop_at=e.cutoff);b,_,bm=simulate(feedback,stop_at=e.cutoff);pd.testing.assert_frame_equal(a,b,check_exact=True);pd.testing.assert_frame_equal(am,bm,check_exact=True)
        issued=ledger[ledger.date.le(e.cutoff)];used={(r.incumbent_cutoff,r.date) for r in issued.itertuples()}|{(r.challenger_cutoff,r.date) for r in issued.itertuples()};bad=feedback.copy();unused=[(r.cutoff,r.date) not in used for r in bad.itertuples()];bad.loc[unused,'prediction_probability']=.001
        a,_,am=simulate(bad,stop_at=e.cutoff);pd.testing.assert_frame_equal(a,b,check_exact=True);pd.testing.assert_frame_equal(am,bm,check_exact=True)
        bad=ledger.copy();stale=bad.trial_start.ne(e.trial_start);bad.loc[stale,'actual_up']=1-bad.loc[stale,'actual_up'];bad.loc[stale,'challenger_probability']=.999
        left=visible_pairs(bad,e.trial_start,e.incumbent_cutoff,e.evaluated_challenger_cutoff,e.cutoff);right=visible_pairs(ledger,e.trial_start,e.incumbent_cutoff,e.evaluated_challenger_cutoff,e.cutoff);pd.testing.assert_frame_equal(left,right,check_exact=True)
        r,_=decide(left,e.trial_start,e.incumbent_cutoff,e.evaluated_challenger_cutoff,e.cutoff);assert r['promote']==e.promote and r['paired_mature_weeks']==e.paired_mature_weeks
        audits.append(dict(cutoff=e.cutoff,paired_mature_weeks=e.paired_mature_weeks,ready=e.ready,promoted=e.promote,future_labels_probs_excluded=True,unissued_model_probs_excluded=True,stale_trial_excluded=True))
    assert len(ledger)==272 and len(events)==23 and len(audits)==17;return pd.DataFrame(audits)

def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();started=now();coverage=coverage_checks();cases=synthetic_cases();assert len(cases)==16;check_frozen(False)
    save(OUT/'contract_verification.json',dict(status='PASS',started_utc=started,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],coverage=coverage,synthetic_cases=cases,actual_policy_simulations=0));print('Contract PASS: frozen coverage and 16 synthetic maturity, dual-gate and chronology cases.',flush=True)
if __name__=='__main__':main()

from common30 import *

def coverage_checks():
    obs,_,_=data();cover=csv('coverage');missing=csv('extension_membership');expected=[]
    for cutoff in cfg()['decision_dates']:
        year=int(cutoff[:4])+(1 if cutoff.endswith('12-31') else 0);end=min(pd.Timestamp(year,12,31),pd.Timestamp(cfg()['label_end']))
        for i,r in obs.iterrows():
            if pd.Timestamp(cutoff)<pd.Timestamp(r.date)<=end and r.weekday==4 and pd.notna(r.joint_completed) and pd.Timestamp(r.joint_completed)<=pd.Timestamp(cfg()['label_end']):expected.append((cutoff,i,r.date,r.joint_completed))
    pd.testing.assert_frame_equal(pd.DataFrame(expected,columns=['cutoff','row_index','date','joint_completed']),cover,check_dtype=False);pd.testing.assert_frame_equal(extensions(cover),missing,check_dtype=False)
    assert not cover.duplicated(['cutoff','date']).any() and cover.cutoff.lt(cover.date).all()
    models=read(OUT/'source_models.json');heads=read(OUT/'source_heads.json');assert len(models)==69 and len(heads)==276
    for r in models:
        assert sha(PROJECT/r['project_file'])==r['sha256'];tr=training_rows(obs,r['cutoff']);np.testing.assert_array_equal(tr,previous_round.previous_round.indices(obs,fold_for(r['cutoff']),'rolling5')[0])
    assert read(V28/'results/verification.json')['status']=='PASS' and read(V29/'results/verification.json')['status']=='PASS'
    return dict(coverage_model_weeks=len(cover),missing_model_weeks=len(missing),source_networks=len(models),source_heads=len(heads))

def synthetic_cases():
    dates=pd.date_range('2021-01-08',periods=17,freq='W-FRI');base=pd.DataFrame(dict(row_index=np.arange(17),date=dates.strftime('%Y-%m-%d'),joint_completed=(dates+pd.Timedelta(days=10)).strftime('%Y-%m-%d'),cutoff='2020-12-31',prediction_probability=.9,frequency_probability=.5,actual_up=0))[FEEDBACK_COLUMNS];active='2020-12-31';decision='2021-06-30';cases=[]
    a=base.iloc[:16].copy();r,_=decide(a,active,decision);assert r['refit'];cases.append('two_positive_blocks_trigger')
    r,_=decide(a.iloc[:15],active,decision);assert not r['refit'] and not r['ready'];cases.append('15_weeks_hold')
    b=a.copy();b['prediction_probability']=.5;r,_=decide(b,active,decision);assert not r['refit'] and r['older8_excess']==r['recent8_excess']==0;cases.append('strict_zero_hold')
    b=a.copy();b.loc[b.index[:8],'prediction_probability']=0.;r,_=decide(b,active,decision);assert not r['refit'] and r['older8_excess']<0<r['recent8_excess'];cases.append('only_recent_block_bad_hold')
    b=a.copy();b.loc[b.index[-8:],'prediction_probability']=0.;r,_=decide(b,active,decision);assert not r['refit'];cases.append('only_older_block_bad_hold')
    b=a.copy();b.loc[b.index[-1],'joint_completed']='2021-07-01';r,_=decide(visible_feedback(b,active,decision),active,decision);assert not r['ready'];cases.append('immature_label_excluded')
    b=a.copy();b.loc[b.index[-1],'joint_completed']=decision;r,_=decide(visible_feedback(b,active,decision),active,decision);assert r['ready'];cases.append('maturity_equality_included')
    b=pd.concat([a,a.assign(cutoff='2019-12-31',prediction_probability=0.)],ignore_index=True);r,_=decide(visible_feedback(b,active,decision),active,decision);assert r['refit'] and r['available_mature_weeks']==16;cases.append('old_model_errors_excluded')
    r,_=decide(visible_feedback(b,'2021-03-31',decision),'2021-03-31',decision);assert not r['ready'] and r['available_mature_weeks']==0;cases.append('refit_resets_membership')
    b=base.copy();b.loc[b.index[0],'prediction_probability']=0.;r,tail=decide(b,active,decision);assert r['refit'] and tail.row_index.tolist()==list(range(1,17));cases.append('latest16_only')
    return cases

def independent_simulation(feedback):
    rows=feedback.to_dict('records');dates=cfg()['decision_dates'];all_dates=sorted({r['date'] for r in rows});lookup={(r['cutoff'],r['date']):r for r in rows};active=None;issued=[];events=[];previous=None;members=[]
    for decision in dates:
        if active is not None:
            for date in all_dates:
                if previous<date<=decision:issued.append(lookup[(active,date)].copy())
        visible=[r for r in issued if r['cutoff']==active and r['joint_completed']<=decision];mandatory=decision.endswith('12-31');ready=False;a=b=np.nan
        if mandatory:trigger=True;reason='annual'
        elif len(visible)<16:trigger=False;reason='warmup'
        else:
            ready=True;recent=sorted(visible,key=lambda r:r['date'])[-16:];delta=[(r['prediction_probability']-r['actual_up'])**2-(r['frequency_probability']-r['actual_up'])**2 for r in recent];a=sum(delta[:8])/8;b=sum(delta[8:])/8;trigger=min(a,b)>0;reason='persistent_underperformance' if trigger else 'no_persistent_underperformance'
        if not mandatory:
            tail=sorted(visible,key=lambda r:r['date'])[-16:]
            for j,r in enumerate(tail):members.append(dict(decision_cutoff=decision,**r,block=('older8' if j<8 else 'recent8') if ready else 'warmup',excess_brier=(r['prediction_probability']-r['actual_up'])**2-(r['frequency_probability']-r['actual_up'])**2))
        selected=decision if trigger else active;events.append(dict(cutoff=decision,previous_model_cutoff=active or '',selected_model_cutoff=selected,mandatory=mandatory,available_mature_weeks=len(visible),ready=ready,older8_excess=a,recent8_excess=b,refit=trigger,reason=reason));active=selected;previous=decision
    for date in all_dates:
        if previous<date<=cfg()['label_end']:issued.append(lookup[(active,date)].copy())
    return pd.DataFrame(events),pd.DataFrame(issued,columns=FEEDBACK_COLUMNS),pd.DataFrame(members,columns=['decision_cutoff']+FEEDBACK_COLUMNS+['block','excess_brier'])

def audit_policy(feedback,events,ledger,members):
    independent_events,independent_ledger,independent_members=independent_simulation(feedback);events=events.fillna({'previous_model_cutoff':''})
    for a,b in [(events,independent_events),(ledger,independent_ledger),(members,independent_members)]:pd.testing.assert_frame_equal(a,b,check_dtype=False,atol=1e-13,rtol=0)
    audit=[]
    for e in events[~events.mandatory].itertuples():
        poisoned=feedback.copy();future=poisoned.joint_completed.gt(e.cutoff);poisoned.loc[future,'actual_up']=1-poisoned.loc[future,'actual_up'];poisoned.loc[future,'prediction_probability']=.999;poisoned.loc[poisoned.date.gt(e.cutoff),'frequency_probability']=.001
        a,_,am=simulate(poisoned,stop_at=e.cutoff);b,_,bm=simulate(feedback,stop_at=e.cutoff);pd.testing.assert_frame_equal(a,b,check_exact=True);pd.testing.assert_frame_equal(am,bm,check_exact=True)
        issued=ledger[ledger.date.le(e.cutoff)].copy();other=issued.cutoff.ne(e.previous_model_cutoff);issued.loc[other,'actual_up']=1-issued.loc[other,'actual_up'];issued.loc[other,'prediction_probability']=.999
        left=visible_feedback(issued,e.previous_model_cutoff,e.cutoff);right=visible_feedback(ledger,e.previous_model_cutoff,e.cutoff);pd.testing.assert_frame_equal(left.reset_index(drop=True),right.reset_index(drop=True),check_exact=True)
        a,_=decide(left,e.previous_model_cutoff,e.cutoff);assert a['refit']==e.refit and a['available_mature_weeks']==e.available_mature_weeks
        audit.append(dict(cutoff=e.cutoff,active_model=e.previous_model_cutoff,mature_weeks=e.available_mature_weeks,ready=e.ready,triggered=e.refit,future_label_probability_invariant=True,other_model_errors_excluded=True))
    assert len(ledger)==272 and len(events)==23 and len(audit)==17;return pd.DataFrame(audit)

def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();started=now();result=coverage_checks();cases=synthetic_cases();check_frozen(False)
    save(OUT/'contract_verification.json',dict(status='PASS',started_utc=started,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],**result,synthetic_cases=cases,actual_feedback_trigger_simulations=0));print('Contract PASS: calendar coverage and ten synthetic maturity/reset/strict-trigger cases.',flush=True)
if __name__=='__main__':main()

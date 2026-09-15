from common44 import *

def synthetic():
    assert expiry_after('2024-03-31')=='2024-06-30' and expiry_after('2024-05-31')=='2024-06-30' and expiry_after('2024-12-31')=='2025-03-31'
    assert checkpoints('2024-03-31','2024-06-30')==['2024-04-30','2024-05-31','2024-06-30']
    meta=pd.DataFrame([dict(row_index=i,date=f'2024-04-{i+1:02d}',joint_completed=f'2024-04-{i+10:02d}',state='negative_low',year=2024,encoder_cutoff='2023-12-31') for i in range(6)])
    meta.loc[0,'date']='2024-03-29'
    assert len(count_members(meta,'2024-03-31','negative_low','2024-04-13'))==4
    assert len(count_members(meta,'2024-03-31','negative_low','2024-04-14'))==5
    assert len(count_members(meta,'2024-04-10','negative_low','2024-04-14'))==4
    try:count_members(pd.concat([meta,meta.iloc[:1]]),'2024-03-31','negative_low','2024-04-30')
    except AssertionError:pass
    else:raise AssertionError('Duplicate metadata was accepted')
    extra=pd.DataFrame([dict(row_index=20+i,date=f'2024-05-{i+1:02d}',joint_completed=f'2024-05-{i+10:02d}',state='nonnegative_high',year=2024,encoder_cutoff='2023-12-31') for i in range(13)])
    z=count_members(pd.concat([meta,extra]),'2024-03-31','negative_low','2024-05-31');assert len(z)==6 and not z.in_rolling13.any()
    rows=[dict(checkpoint='2024-04-30',observed=True,cumulative_n=4,rolling13_n=4),dict(checkpoint='2024-05-31',observed=True,cumulative_n=5,rolling13_n=5),dict(checkpoint='2024-06-30',observed=True,cumulative_n=6,rolling13_n=4)]
    assert classify_track(rows,'2024-06-30','2024-06-30','rolling13')[0]=='reached_before_expiry'
    assert classify_track(rows[1:],'2024-05-31','2024-05-31','cumulative')[0]=='reached_only_at_expiry'
    partial=[rows[0],dict(checkpoint='2024-05-31',observed=False,cumulative_n=None,rolling13_n=None)]
    assert classify_track(partial,'2024-06-30','2024-04-30','cumulative')[0]=='observation_censored'
    return ['strict_post_source_and_inclusive_maturity','signal_before_source_but_new_maturity','duplicate_metadata_rejected','rolling13_can_drop_cumulative_evidence','strict_next_quarter_expiry','expiry_day_not_actionable','first_readiness_survives_later_window_drop','unknown_future_not_zero']
def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();tests=synthetic()
    bank=csv('weekly_signal_bank');meta=canonical_metadata(bank);gates=csv('gate_decisions');heads=read(OUT/'correction_heads.json');accepted=gates[gates.accepted].sort_values(['cutoff','method','component'])
    assert len(meta)==272 and len(bank)==3264 and len(accepted)==84 and sum(accepted.method.isin(PRIMARY))==65
    assert bank.groupby('row_index').size().eq(12).all() and bank.joint_completed.le(cfg()['label_end']).all()
    records=[]
    for r in accepted.itertuples():
        assert r.mode=='ready' and r.training_n>=10 and r.validation_n>=5 and r.reason=='accepted'
        h=[x for x in heads if x['cutoff']==r.cutoff and x['method']==r.method and x['component']==r.component]
        assert sorted(x['seed'] for x in h)==cfg()['seeds'] and all(x['fit_eligible'] for x in h)
        end=expiry_after(r.cutoff)
        records.append(dict(origin_id=f'{r.cutoff}:{r.method}:{r.component}',event_id=f'{r.cutoff}:{r.component}',source_cutoff=r.cutoff,method=r.method,state=r.component,expiry=end,source_year=int(r.cutoff[:4]),source_training_n=int(r.training_n),source_validation_n=int(r.validation_n),horizon_days=int((pd.Timestamp(end)-pd.Timestamp(r.cutoff)).days),planned_checkpoints=len(checkpoints(r.cutoff,end)),planned_preexpiry_checkpoints=sum(d<end for d in checkpoints(r.cutoff,end))))
    files=[]
    for name,g in [('origins',pd.DataFrame(records)),('weekly_metadata',meta)]:
        p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen(False);save(OUT/'contract_verification.json',dict(status='PASS',completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],frozen_origins=84,primary_origins=65,weekly_metadata_rows=272,synthetic_cases=tests,labels_or_probabilities_used=False,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('Contract PASS: all 84 original acceptances and 272 metadata rows frozen; maturity/expiry/censoring boundaries verified.',flush=True)
if __name__=='__main__':main()

"""Validate source history and freeze the stock experiment before model fitting."""
from stock import *

def prepared_csv(p,f):
    if Path(p).exists():assert Path(p).read_bytes()==f.to_csv(index=False,lineterminator='\n').encode('utf-8')
    else:csv(p,f)

def prepared_save(p,obj):
    if Path(p).exists():assert read(p)==obj
    else:save(p,obj)

def main():
    OUT.mkdir(exist_ok=True)
    raw=load_csv(ROOT/'data/raw.csv');vendor=load_csv(ROOT/'data/hfq.csv');qfq=load_csv(ROOT/'data/qfq.csv')
    assert raw.date.equals(vendor.date) and raw.date.equals(qfq.date)
    cal=load_csv(CALENDAR).date
    cal=cal[cal.between(raw.date.min(),cfg()['data_end'])].reset_index(drop=True)
    assert cal.iloc[-1]==cfg()['data_end'] and raw.date.isin(cal).all()
    f=build_frame(raw,cal);observed=f[f.valid_ohlc]
    np.testing.assert_array_equal(observed.date,raw.date)
    gap=float(np.max(abs(observed[['open','high','low','close']].to_numpy()-vendor[['open','high','low','close']].to_numpy())))
    # Tencent rounds intermediate corporate-action accumulators to 3 decimals;
    # allow 0.005 for accumulated rounding across nine actions (not model fit).
    assert gap<.005,('Corporate-action reconstruction mismatch',gap)
    # A second vendor lists these dates and distributions independently.
    prepared_csv(ROOT/'data/actions.csv',pd.DataFrame(ACTIONS,columns=['ex_date','share_multiplier','cash_per_pre_action_share']))
    prepared_csv(ROOT/'data/model_frame.csv',f)
    _,obs,_=observations(f,cfg()['data_end']);prepared_csv(OUT/'observations.csv',obs)
    memberships=[]
    for cutoff in cfg()['cutoffs']:
        a=annual_interface(f,cutoff);g=a['observations'].iloc[a['training_rows']]
        memberships.append(dict(cutoff=cutoff,lower=a['lower'],train_n=len(g),first_anchor=g.date.min(),last_anchor=g.date.max(),last_maturity=g.joint_completed.max()))
    missing=f.loc[~f.valid_ohlc,'date'];prepared_csv(OUT/'missing_stock_sessions.csv',missing.to_frame())
    prepared_save(OUT/'data_validation.json',dict(status='PASS',raw_rows=len(raw),calendar_rows=len(cal),missing_stock_sessions=len(missing),
        start=raw.date.min(),end=raw.date.max(),corporate_actions=len(ACTIONS),vendor_hfq_max_difference=gap,
        training_windows=memberships,missing_year_counts=missing.str[:4].value_counts().sort_index().to_dict(),
        mature_friday_counts=obs[obs.weekday.eq(4)&obs.date.ge('2023-01-01')].date.str[:4].value_counts().sort_index().to_dict(),
        adjustment_sources=['https://www.etnet.com.hk/www/sc/ashares/quote_dividend.php?code=603986',
            'https://pdf.dfcfw.com/pdf/H2_AN201805141143166795_1.pdf'],
        limitation='Raw prices are a retrospective vendor vintage; forward corporate-action construction is causal, but cannot prove the vendor never revised raw history. Missing stock sessions are excluded without imputing a reason or fills.'))
    r,_,_=modules()
    files=set(ROOT.glob('*.py'))|{ROOT/'protocol.json',CALENDAR}
    files.update(p for p in (ROOT/'data').iterdir() if p.is_file())
    # Freeze actual imported local numerical sources, and all legacy protocol
    # files (read-only) that their nested cfg functions might access.
    for m in list(sys.modules.values()):
        name=getattr(m,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.is_relative_to(PROJECT) and '.venv' not in str(p) and p.suffix=='.py' and p.is_file():files.add(p)
    files.update(PROJECT.glob('research*/protocol.json'))
    save(OUT/'freeze.json',dict(created_utc=now(),stock='603986',files={relative(p):sha(p) for p in sorted(files)}))
    print(json.dumps(memberships,indent=2),flush=True)

if __name__=='__main__':main()

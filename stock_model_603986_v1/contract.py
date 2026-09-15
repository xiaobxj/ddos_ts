"""Causal stock data tests, independent of prediction performance."""
from stock import *

def main():
    freeze=read(OUT/'freeze.json')
    for name,digest in freeze['files'].items():assert sha(PROJECT/name)==digest,name
    f=data();raw=load_csv(ROOT/'data/raw.csv');checks=[]
    # Corporate actions after a prefix cannot change that prefix.
    for cutoff in cfg()['cutoffs']+['2026-05-25','2026-09-14']:
        prefix=build_frame(raw[raw.date.le(cutoff)],f.loc[f.date.le(cutoff),'date'].reset_index(drop=True))
        full=f[f.date.le(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix,full,check_exact=False,atol=1e-10,rtol=1e-14)
    checks.append('Six forward-adjustment prefixes reproduce full vintage without future actions')
    _,allobs,targets=observations(f,cfg()['data_end'])
    for cutoff in cfg()['cutoffs']:
        a=annual_interface(f,cutoff);subset=allobs[allobs.joint_completed.le(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(a['observations'],subset,check_exact=True)
        g=a['observations'].iloc[a['training_rows']]
        assert g.date.ge(a['lower']).all() and g.joint_completed.le(cutoff).all()
        modified=f.copy();modified.loc[modified.date.gt(cutoff),['open','high','low','close','volume']]*=7
        b=annual_interface(modified,cutoff)
        for key in a['values']:np.testing.assert_array_equal(a['values'][key],b['values'][key])
        assert a['scales']==b['scales']
    checks.append('Four annual mature-label memberships and tensors invariant to future perturbation')
    # Independently sum actual dividends and split shares for every target.
    maxgap=0.
    for row in allobs.itertuples():
        assert f.valid_ohlc.iloc[row.anchor-124:max(row.anchor+5,row.exit)+1].all()
        shares=1.;cash=0.
        for date,multiplier,dividend in ACTIONS:
            if row.entry_date < date <= row.exit_date:
                cash+=shares*dividend;shares*=multiplier
        direct=(shares*f.raw_open.iloc[row.exit]+cash)/f.raw_open.iloc[row.entry]-1
        maxgap=max(maxgap,abs(direct-row.actual_return))
        assert abs(direct-row.actual_return)<1e-12
        assert pd.Timestamp(f.date.iloc[row.exit-1]).dayofweek==row.weekday
    checks.append('Every gross holding-period label independently reconstructed; all gap windows excluded')
    # A deliberately injected missing input must remove affected samples.
    lastanchor=int(allobs.anchor.iloc[-1]);bad=f.copy();bad.loc[lastanchor-50,'valid_ohlc']=False
    _,masked,_=observations(bad,cfg()['data_end']);assert lastanchor not in set(masked.anchor)
    checks.append('Injected missing stock session removes affected observations')
    r,_,_=modules(True);model=r.legacy.make_model('combined',SEEDS[0]);assert sum(p.numel() for p in model.parameters())==38551
    a=annual_interface(f,cfg()['cutoffs'][0]);v={k:r.torch.from_numpy(x[:3]).cuda() for k,x in a['values'].items()}
    y={k:r.torch.from_numpy(x[:3]).cuda() for k,x in a['labels'].items()}
    original=r.training.model_hash(model);optim=r.training.optimizer_for(model)
    loss,ret,aux,norm=r.training.update_step(model,optim,v,y,np.arange(3))
    assert np.isfinite([loss,ret,aux,norm]).all() and norm>0 and r.training.model_hash(model)!=original
    checks.append('Fresh 38551-parameter stock network has finite nonzero training update (contract model discarded)')
    save(OUT/'contract.json',dict(status='PASS',created_utc=now(),freeze_sha256=sha(OUT/'freeze.json'),checks=checks,
        label_reconstruction_max_difference=maxgap,contract_model_saved=False))
    print('Stock contract PASS:',len(checks),'checks',flush=True)

if __name__=='__main__':main()

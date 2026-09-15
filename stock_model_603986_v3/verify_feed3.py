"""Live public feed and inference replay checks before the first genuine event."""
from experiment3 import *
import daily3 as d
import copy

def main():
    root=OUT/'live_feed_probe';root.mkdir(exist_ok=False)
    fetched=d.download(root,d.utc_clock().astimezone(d.CST));frame,snapshot=d.frames(root,[],fetched)
    sealed=data();pd.testing.assert_frame_equal(frame,sealed,check_exact=True)
    changed=copy.deepcopy(fetched);changed['hfq'].loc[changed['hfq'].index[-1],'close']+=.1
    try:d.frames(root,[],changed)
    except AssertionError:pass
    else:raise AssertionError('Unknown corporate-action mismatch was accepted')
    incomplete=copy.deepcopy(fetched);incomplete['market']=incomplete['market'].iloc[:-1]
    try:d.frames(root,[],incomplete)
    except AssertionError:pass
    else:raise AssertionError('Incomplete exchange calendar was accepted')
    rows=d.infer(frame);checks=[]
    for row in rows:
        if row['name'].startswith('h5.') and row['name']!='h5.frequency':
            name=row['name'].replace('h5.','').replace('.raw','.vol.raw').replace('.platt','.vol.platt').replace('.shrink','.vol.shrink')
            source=load_csv(OUT/'slow_predictions.csv');v=source[source.name.eq(name)&source.date.eq(cfg()['data_end'])].probability.iloc[0]
        elif row['name'].startswith('h1.') and row['name']!='h1.frequency':
            folder=fit_folder(row['training_cutoff'],1);obs,_=observations(frame,1);idx=obs.index[obs.date.eq(cfg()['data_end'])][0]
            probabilities=[]
            for seed in SEEDS:
                with np.load(folder/f'cache_{seed}_e20.npz') as z:probabilities.append(z['p__vol.U'][idx])
            v=float(np.mean(probabilities))
        else:continue
        gap=abs(row['probability']-v);assert gap<5e-5
        checks.append(dict(name=row['name'],fresh_probability=row['probability'],cached_probability=float(v),delta=gap))
    save(OUT/'daily_feed_verified.json',dict(status='PASS',completed_utc=now(),frame_replay='bit exact after round-trip CSV read',snapshot=snapshot,
        checks=checks,max_delta=max(c['delta'] for c in checks),guards=['unknown corporate action rejected','incomplete exchange calendar rejected'],
        no_forecast_events_created=True))

if __name__=='__main__':main()

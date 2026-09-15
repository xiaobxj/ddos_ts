from experiment3 import *
from scipy.special import expit
import daily3 as d

def main():
    r,_,_=modules();f=data();checks=[]
    for task in tasks():
        a=interface(f,task);ids=a['ids'];folder=fit_folder(task['cutoff']);done=read(folder/'completed.json')
        assert done['training_frequency']==float(a['y'].mean())
        with np.load(folder/'controls.npz') as z:cache={k:z[k] for k in z.files}
        np.testing.assert_array_equal(cache['market'],a['market']);np.testing.assert_array_equal(cache['gate'],a['gate'])
        fit=r.fit_clip(a['market'][ids])
        for k in fit:
            if k!='standardized':np.testing.assert_array_equal(fit[k],cache[k])
        x=r.transform(a['market'],cache);theta=cache['market_theta'];z=np.c_[x,np.ones(len(x))];p=expit(z@theta)
        np.testing.assert_allclose(p,cache['market_probability'],rtol=0,atol=2e-15)
        grad=z[ids].T@(p[ids]-a['y'])/len(ids)+.01*np.r_[theta[:-1],0.]
        assert abs(grad).max()<3e-9
        checks.append(dict(cutoff=task['cutoff'],training_n=len(ids),gradient_inf=float(abs(grad).max())))
    d.package_check();events=d.chain(d.LIVE);forecasts=[e for e in events if e['kind']=='forecast'];labels=[e for e in events if e['kind']=='label']
    assert len(forecasts)==1 and len(labels)==0 and len(events)==4
    first=forecasts[0];assert first['payload']['signal_date']=='2026-09-15'
    assert set(v['name'] for v in first['payload']['rows'])==set(d.NAMES)
    assert first['payload']['targets']=={'1':'2026-09-16','5':'2026-09-22'}
    for e in events:
        if e['kind']=='snapshot':
            for k in ['raw_blob','calendar_blob','frame_blob']:d.getblob(d.LIVE,e['payload'][k])
            for source in e['payload']['sources'].values():d.getblob(d.LIVE,source['blob'])
    save(OUT/'final_audit.json',dict(status='PASS',completed_utc=now(),new_control_checks=checks,
        genuine_forecast_days=len(forecasts),matured_labels=0,event_count=len(events),duplicate_run_preserved_forecast=True,
        first_forecast_utc=first['recorded_utc'],first_forecast_sha256=first['sha256'],last_chain_sha256=events[-1]['sha256'],
        initial_live_files={rel(p):sha(p) for p in sorted(d.LIVE.rglob('*')) if p.is_file()}))

if __name__=='__main__':main()

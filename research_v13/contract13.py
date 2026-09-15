from common13 import *

def main():
    legacy.initialize();check_frozen(False);tests=[]
    obs=pd.read_csv(OUT/'observation_table.csv');scales=read(OUT/'training_scales.json')
    with np.load(V5/'cache/targets.npz') as raw,np.load(V5/'cache/packed_raw.npz') as packed:
        for cutoff in cfg()['new_training_cutoffs']:
            with np.load(CACHE/f'training_{cutoff}.npz') as data:
                tr=np.flatnonzero(obs.joint_completed.le(cutoff).to_numpy())
                np.testing.assert_array_equal(data['row_index'],tr)
                for k in ['patches','geometry','valid']:np.testing.assert_array_equal(data[k],packed[k][tr])
                for k in ['returns','auxiliary']:
                    y=raw[k][tr].astype(float);m=y.mean(axis=0);s=np.maximum(y.std(axis=0),1e-6)
                    np.testing.assert_array_equal(m,np.array(scales[cutoff][k+'_mean']))
                    np.testing.assert_array_equal(s,np.array(scales[cutoff][k+'_sd']))
                    np.testing.assert_array_equal(data[k],((y-m)/s).astype(np.float32))
    member=pd.read_csv(OUT/'calibration_membership.csv')
    for cutoff,g in member.groupby('outer_cutoff'):
        expected=np.flatnonzero((obs.date.ge(f'{int(cutoff[:4])-2}-01-01')&obs.date.le(cutoff)&
            obs.weekday.eq(4)&obs.joint_completed.le(cutoff)).to_numpy())
        np.testing.assert_array_equal(g.row_index,expected)
        assert g.joint_completed.le(cutoff).all() and g.inner_cutoff.lt(g.date).all()
    tests.append('causal training/rolling membership, labels, packed features and exact scales')
    dates=pd.date_range('2010-01-01',periods=156,freq='7D').strftime('%Y-%m-%d')
    x=np.linspace(-2,2,156);m=np.full(156,.25)
    g=pd.DataFrame(dict(date=dates,predicted_return=x+m,training_mean=m,actual=.3*x+m))
    assert abs(coefficient(g)['alpha']-.3)<1e-14
    for slope,expected in [(-2,0.),(2,1.),(0,0.)]:
        h=g.copy();h['actual']=slope*x+m;assert coefficient(h)['alpha']==expected
    h=g.copy();h['predicted_return']=m;assert coefficient(h)['fallback'] and coefficient(h)['alpha']==0
    assert coefficient(g.iloc[:20])['fallback']
    tests.append('least-squares recovery, clipping endpoints, zero denominator and insufficient history')
    p=np.stack([x,x+.4,x-.8]);alpha=.37
    np.testing.assert_allclose(transform(p,m,alpha).mean(axis=0),transform(p.mean(axis=0),m,alpha),rtol=0,atol=5e-16)
    np.testing.assert_array_equal(transform(x,m,0.),m)
    np.testing.assert_allclose(transform(x,m,1.),x,rtol=0,atol=5e-16)
    tests.append('training-mean centering, endpoint identities and equal-seed ensemble commutation')
    # One real dropout update against the literal original round6 expression.
    v,labels,tr,s=load_training('2014-12-31');seed=cfg()['seeds'][0]
    a=legacy.make_model('combined',seed);oa=prior.optimizer_for(a);a.train()
    idx=np.arange(128);prior.update_step(a,oa,v,labels,idx)
    ha=object_hash(a.state_dict());hoa=object_hash(oa.state_dict());ra=torch.cuda.get_rng_state().clone()
    b=legacy.make_model('combined',seed);ob=prior.optimizer_for(b);b.train()
    batch=torch.tensor(idx,device='cuda');ob.zero_grad(set_to_none=True)
    out,aux=legacy.predict(b,{k:t[batch] for k,t in v.items()},True)
    loss=((out-labels['returns'][batch])**2).mean()
    loss=loss+.1*((aux-labels['auxiliary'][batch])**2).mean();loss.backward()
    torch.nn.utils.clip_grad_norm_(b.parameters(),1.,error_if_nonfinite=True);ob.step()
    assert ha==object_hash(b.state_dict()) and hoa==object_hash(ob.state_dict())
    assert torch.equal(ra,torch.cuda.get_rng_state())
    tests.append('bit-exact real-model weights/AdamW/dropout state versus the original MSE update')
    result=dict(status='PASS',tests=tests,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),completed_utc=now())
    save(OUT/'contract_verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

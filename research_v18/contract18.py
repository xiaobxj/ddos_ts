from common18 import *

def main():
    check_frozen(False);assert not (OUT/'contract_verification.json').exists();tests=[]
    obs,price,returns=data();bars=base16.raw_bars(price);parents=read(OUT/'parent_heads.json');sources={h['job']:h for h in read(V17/'results/source_heads.json')}
    max_duplicate=0.;max_standardized=0.
    for fold in cfg()['folds']:
        tr,te=indices(obs,fold);f=market_rows('training',fold['cutoff'],tr);d=fit_market(f)
        np.testing.assert_allclose(f,scalar_descriptors(bars,obs.anchor.iloc[tr].to_numpy()),rtol=0,atol=1e-12)
        h=next(h for h in parents if h['cutoff']==fold['cutoff'] and h['method']=='raw25_clip');raw=load_npz(sources[h['source_job']]);normal=load_npz(h)
        for mi,ri in DUPLICATES.items():
            err=float(np.max(np.abs(f[:,mi]-raw['features'][:,ri])));zerr=float(np.max(np.abs(d['standardized'][:,mi]-normal['standardized'][:,ri])))
            assert err<1e-14 and zerr<1e-12;max_duplicate=max(max_duplicate,err);max_standardized=max(max_standardized,zerr)
        cutoff_anchor=int(np.flatnonzero(price.date.le(fold['cutoff']).to_numpy())[-1]);changed=bars.copy();changed[cutoff_anchor+1:,:4]*=9;changed[cutoff_anchor+1:,4]+=1e9
        np.testing.assert_array_equal(descriptors(bars,obs.anchor.iloc[tr].to_numpy()),descriptors(changed,obs.anchor.iloc[tr].to_numpy()))
    tests.append('all six training masks and scalar market formulas; post-cutoff price invariance; explicit three-column raw/standardized duplicate parity')
    rng=np.random.default_rng(20261800);y=(rng.random(101)>.48).astype(float)
    for dim in [4,26,29]:
        f=rng.normal(size=(101,dim));d=fit_market(f);theta,trace=fit_newton(d['standardized'],y,cfg()['probe']);v,g,h=objective(theta,design(d['standardized']),y,.01)
        assert len(theta)==dim+1 and np.max(np.abs(g))<=1e-9 and np.linalg.eigvalsh(h).min()>0
        assert all(b['objective']<=a['objective']+1e-13 for a,b in zip(trace,trace[1:]))
    known=np.column_stack([np.arange(101.),np.ones(101)]);d=fit_market(known)
    np.testing.assert_array_equal(d['lower'],[1.,1.]);np.testing.assert_array_equal(d['upper'],[99.,1.]);assert d['sd'][1]==1e-6
    tests.append('solver convergence in4/26/29dimensions; exact known quantile bounds and constant-column SD floor')
    # Adding zero coefficients preserves each parent score and penalty exactly.
    for parent in parents:
        d=load_npz(parent);old=np.asarray(parent['coefficients']);extra=4 if parent['method']=='learned_clip' else 1
        context=np.zeros((len(d['standardized']),extra));a=np.column_stack([d['standardized'],context]);theta=np.r_[old[:-1],np.zeros(extra),old[-1]]
        np.testing.assert_allclose(design(a)@theta,design(d['standardized'])@old,rtol=0,atol=1e-12)
        v,_,_=objective(theta,design(a),d['direction'],.01);assert abs(v-parent['objective'])<1e-12
    tests.append('24parent models nested at zero appended slopes with identical standardized representation and ridge objective')
    save(OUT/'contract_verification.json',dict(status='PASS',completed_utc=now(),tests=tests,maximum_training_raw_duplicate_error=max_duplicate,
        maximum_training_standardized_duplicate_error=max_standardized,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes()))
    print(json.dumps(dict(status='PASS',tests=tests,max_raw_duplicate_error=max_duplicate,max_standardized_duplicate_error=max_standardized)),flush=True)

if __name__=='__main__':main()

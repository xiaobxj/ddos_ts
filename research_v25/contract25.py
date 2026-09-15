from common25 import *
from contract24 import provenance as inherited_provenance

def synthetic():
    rng=np.random.default_rng(20262500);n=180;base=rng.normal(0,.5,n);feature=rng.normal(0,1,n);outcome=(rng.random(n)>.45).astype(float)
    cases=[('ordinary',base,feature,outcome),('zero_feature',base,np.zeros(n),outcome),('constant_feature',base,np.ones(n),outcome),('all_up',base,feature,np.ones(n)),('all_down',base,feature,np.zeros(n)),('extreme_offset',np.where(np.arange(n)%2,80.,-80.),feature,outcome)];checks=[]
    for name,offset,h,y in cases:
        v,g,hh=offset_objective(.17,offset,h,y);eps=1e-5
        numeric=(offset_objective(.17+eps,offset,h,y)[0]-offset_objective(.17-eps,offset,h,y)[0])/(2*eps)
        numeric_h=(offset_objective(.17+eps,offset,h,y)[1]-offset_objective(.17-eps,offset,h,y)[1])/(2*eps)
        assert abs(numeric-g)<1e-8 and abs(numeric_h-hh)<1e-8
        gamma,trace=fit_offset(offset,h,y,cfg()['probe']);alt=independent_root(offset,h,y,cfg()['probe']);assert abs(gamma-alt['gamma'])<1e-7 and abs(offset_objective(gamma,offset,h,y)[1])<=1e-10
        flipped,_=fit_offset(offset,-h,y,cfg()['probe']);assert abs(gamma+flipped)<1e-12;np.testing.assert_allclose(offset+gamma*h,offset+flipped*(-h),rtol=0,atol=1e-12)
        if name=='zero_feature':assert gamma==0 and trace[-1]['iteration']==0
        checks.append(dict(case=name,gamma=gamma,root_gap=abs(gamma-alt['gamma']),finite_difference_gradient_gap=abs(numeric-g),finite_difference_hessian_gap=abs(numeric_h-hh)))
    return checks

def main():
    check_frozen(False);assert not (OUT/'contract_verification.json').exists();checks=synthetic();chain=pd.DataFrame(inherited_provenance());pd.testing.assert_frame_equal(chain,csv('temporal_provenance'),check_dtype=False)
    obs,price,returns=data();parents={h['job']:h for h in read(V19/'results/heads.json')};nested=[]
    for source in read(OUT/'source_heads.json'):
        parent=parents[source['source_job']];x,tr=inputs(source,'training');pt=np.asarray(parent['coefficients']);offset=design(x[:,:-1])@pt;constant=.005*np.square(pt[:-1]).sum();zero=offset_objective(0.,offset,x[:,-1],(returns[tr]>0).astype(float))[0]+constant
        assert abs(zero-parent['objective'])<1e-12;np.testing.assert_array_equal(x[:,:-1],load_npz(parent)['standardized']);nested.append(dict(source_job=source['job'],train_n=len(tr),frozen_parent_coordinates=len(pt),zero_gamma_objective_gap=abs(zero-parent['objective'])))
    files=[]
    for name,rows in [('synthetic_offset_checks',checks),('nesting_contract',nested)]:path=OUT/f'{name}.csv';pd.DataFrame(rows).to_csv(path,index=False);files.append(path)
    tests=['Six scalar problems: ordinary,zero/constant feature,all-up/all-down labels,extreme offsets','Gradient and Hessian finite differences; Newton vs bracketed root','Feature sign flip symmetry and zero-feature exact parent recovery','24training parent inputs and zero-gamma penalized objectives match R19','24upstream temporal provenance chains unchanged; no real candidate fits or heldout scoring in contract']
    save(OUT/'contract_verification.json',dict(status='PASS',completed_utc=now(),tests=tests,source_sha256=source_hashes(),protocol_sha256=sha(ROOT/'protocol.json'),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},new_candidate_fits=0,new_heldout_predictions=0));print(json.dumps(dict(status='PASS',tests=tests)),flush=True)
if __name__=='__main__':main()

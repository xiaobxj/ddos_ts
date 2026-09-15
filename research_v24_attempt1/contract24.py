from common24 import *
from scipy.optimize import minimize
from scipy.special import expit

def synthetic():
    rng=np.random.default_rng(20262400);checks=[]
    for dim in [28,31]:
        x=rng.normal(size=(140,dim));y=(rng.random(140)>.47).astype(float);beta=rng.normal(0,.15,dim+1)
        for factor in [1,4,16]:
            xx=scaled_inputs(x,factor);theta=beta.copy();theta[-2]*=np.sqrt(factor)
            np.testing.assert_array_equal(xx[:,:-1],x[:,:-1]);np.testing.assert_allclose(design(xx)@theta,design(x)@beta,rtol=0,atol=1e-14)
            a,g,h=objective(theta,design(xx),y,.01);b,gg=direct_objective(beta,x,y,factor);assert abs(a-b)<1e-14
            expected=gg.copy();expected[-2]/=np.sqrt(factor);np.testing.assert_allclose(g,expected,rtol=0,atol=1e-14)
            eps=1e-6
            for j in [0,dim-2,dim-1,dim]:
                delta=np.zeros(dim+1);delta[j]=eps
                numeric=(direct_objective(beta+delta,x,y,factor)[0]-direct_objective(beta-delta,x,y,factor)[0])/(2*eps)
                assert abs(numeric-gg[j])<2e-9
            fitted,trace=fit_newton(xx,y,cfg()['probe']);original=original_coefficients(fitted,factor)
            initial=np.zeros(dim+1);initial[-1]=np.log(y.mean()/(1-y.mean()))
            alternate=minimize(direct_objective,initial,args=(x,y,factor),jac=True,method='L-BFGS-B',options=cfg()['independent_solver']['options'])
            value,grad=direct_objective(original,x,y,factor);other,og=direct_objective(alternate.x,x,y,factor)
            assert abs(value-other)<1e-10 and abs(expit(design(x)@original)-expit(design(x)@alternate.x)).max()<1e-5
            assert abs(og).max()<5e-7 and abs(grad).max()<5e-9
            checks.append(dict(dimensions=dim,multiplier=factor,objective_gap=abs(value-other),finite_difference_pass=True))
    return checks

def provenance():
    obs,price,returns=data();r19={h['job']:h for h in read(V19/'results/heads.json')};r18={h['job']:h for h in read(V18/'results/heads.json')};r17={h['job']:h for h in read(V17/'results/heads.json')};r16={h['job']:h for h in read(V17/'results/source_heads.json')};records=[]
    for s in read(OUT/'source_heads.json'):
        parent=r19[s['source_job']];additive=r18[parent['source_job']];clipped=r17[additive['parent_job']];upstream=r16[clipped['source_job']]
        f=next(f for f in cfg()['folds'] if f['cutoff']==s['cutoff']);tr,te=indices(obs,f)
        np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(s['cutoff'])))
        assert len(tr)==f['train_n'] and obs.joint_completed.iloc[tr].max()<=s['cutoff']
        for source in [s,parent,additive,clipped,upstream]:
            assert source['cutoff']==s['cutoff'] and source['seed']==s['seed'];cache=load_npz(source)
            np.testing.assert_array_equal(cache['row_index'],tr)
        x,rows=inputs(s,'training');np.testing.assert_array_equal(rows,tr);np.testing.assert_array_equal(x[:,:-1],load_npz(parent)['standardized'])
        np.testing.assert_array_equal(load_npz(s)['ray'],np.asarray(additive['coefficients'])[:25])
        if s['method'].startswith('learned'):
            assert sha(PROJECT/upstream['project_file'])==upstream['sha256']
        records.append(dict(job=s['job'],parent_job=parent['job'],additive_job=additive['job'],clipped_job=clipped['job'],feature_source_job=upstream['job'],cutoff=s['cutoff'],seed=s['seed'],train_n=len(tr),last_training_signal=obs.date.iloc[tr].max(),last_training_maturity=obs.joint_completed.iloc[tr].max(),first_nextyear_signal=obs.date.iloc[te].min(),supervised_features_training_in_sample=True,full_pipeline_oof=False,new_neural_execution=False))
    assert len(records)==24;return records

def main():
    check_frozen(False);assert not (OUT/'contract_verification.json').exists();toy=synthetic();chain=provenance();files=[]
    for name,rows in [('synthetic_penalty_checks',toy),('temporal_provenance',chain)]:path=OUT/f'{name}.csv';pd.DataFrame(rows).to_csv(path,index=False);files.append(path)
    tests=['Scaling preserves all original parent coordinates and logits under coefficient mapping','Explicit diagonal penalty and gradients equal scaled-coordinate ridge; finite differences agree','Six synthetic convex fits agree with independent original-coordinate L-BFGS-B','24upstream job chains, cache row indices and mature-label cutoffs match; learned checkpoints hash unchanged','No new heldout probabilities or full-pipeline OOF claim']
    result=dict(status='PASS',completed_utc=now(),tests=tests,source_sha256=source_hashes(),protocol_sha256=sha(ROOT/'protocol.json'),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},new_candidate_fits=0,new_heldout_predictions=0)
    save(OUT/'contract_verification.json',result);print(json.dumps(dict(status='PASS',tests=tests)),flush=True)
if __name__=='__main__':main()

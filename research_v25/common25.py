"""Freeze the R19 parent and learn one scalar order correction."""
from pathlib import Path
import sys,json,time,shutil
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache'
V24=PROJECT/'research_v24';V23=PROJECT/'research_v23';V19=PROJECT/'research_v19';V18=PROJECT/'research_v18';V17=PROJECT/'research_v17'
sys.path.insert(0,str(V24));import common24 as previous
np=previous.np;pd=previous.pd;read=previous.read;save=previous.save;sha=previous.sha;load_npz=previous.load_npz
probability=previous.probability;design=previous.design;metric=previous.metric;probability_losses=previous.probability_losses;data=previous.data;indices=previous.indices;reference=previous.reference;inputs=previous.inputs
CORE=['common25.py','prepare25.py','contract25.py','train25.py','score25.py','evaluate25.py','verify25.py']
CANDIDATES={'learned_order_offset':'learned_vol_interaction','raw_order_offset':'raw_vol_interaction'}
WEAK={'learned_order_offset':'learned_order_extension','raw_order_offset':'raw_order_extension'}
OLD_INTERACTIONS={'learned_order_offset':'learned_market','raw_order_offset':'raw_trend'}
GATE_MAP={m:'excess_order' for m in CANDIDATES};CONTROLS={'excess_order':'order_only'}
METHODS=previous.METHODS+list(CANDIDATES);SEED_METHODS=previous.SEED_METHODS+['learned_order_offset'];PARTITIONS=previous.PARTITIONS
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')
def source_hashes():
    r=previous.source_hashes();r.update({str((ROOT/n).relative_to(PROJECT)):sha(ROOT/n) for n in CORE});return r
def old_evidence():
    r=dict(read(V24/'results/preparation_manifest.json')['old_evidence'])
    for name,digest in read(V24/'results/delivery_manifest.json')['files'].items():r[str((V24/name).relative_to(PROJECT))]=digest
    path=V24/'results/delivery_manifest.json';r[str(path.relative_to(PROJECT))]=sha(path);assert len(r)==3585
    for name,digest in r.items():assert sha(PROJECT/name)==digest,name
    return r
def input_hashes():
    r=previous.input_hashes();paths=[V24/'protocol.json']+[V24/'results'/n for n in ['heads.json','source_heads.json','temporal_provenance.csv','model_predictions.csv','ensemble_predictions.csv','verification.json','delivery_manifest.json']]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__,device='CPU float64; one scalar correction; no neural execution')
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    r=read(OUT/'preparation_manifest.json');assert r['protocol_sha256']==sha(ROOT/'protocol.json') and r['source_sha256']==source_hashes() and r['input_sha256']==input_hashes()
    for name,digest in r['artifacts'].items():assert sha(ROOT/name)==digest,name
    if contract:
        c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and c['source_sha256']==r['source_sha256'] and c['protocol_sha256']==r['protocol_sha256']
        for name,digest in c['artifacts'].items():assert sha(ROOT/name)==digest,name
    return r
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for name,digest in r['artifacts'].items():assert sha(ROOT/name)==digest,name
    return r
def offset_objective(gamma,offset,h,y,penalty=.01):
    z=offset+gamma*h;p=probability(z)
    return (float(np.mean(np.logaddexp(0,z)-y*z)+.5*penalty*gamma*gamma),
        float(np.mean(h*(p-y))+penalty*gamma),float(np.mean(h*h*p*(1-p))+penalty))
def fit_offset(offset,h,y,settings):
    offset=np.asarray(offset,float);h=np.asarray(h,float);y=np.asarray(y,float);assert offset.shape==h.shape==y.shape and len(y)>0 and np.isfinite(offset).all() and np.isfinite(h).all()
    gamma=0.;trace=[]
    for iteration in range(settings['max_iterations']+1):
        value,grad,hess=offset_objective(gamma,offset,h,y,settings['l2_lambda']);assert hess>=settings['l2_lambda'] and np.isfinite([value,grad,hess]).all()
        trace.append(dict(iteration=iteration,gamma=gamma,objective=value,gradient=grad,hessian=hess))
        if abs(grad)<=settings['gradient_absolute_tolerance']:return gamma,trace
        step=grad/hess;scale=1.
        for backtrack in range(settings['line_search_max_steps']):
            proposed=gamma-scale*step;trial=offset_objective(proposed,offset,h,y,settings['l2_lambda'])[0]
            if trial<=value-settings['armijo']*scale*grad*step+settings['objective_roundoff_allowance']:gamma=float(proposed);break
            scale*=.5
        else:raise AssertionError('Offset Newton line search failed')
    raise AssertionError('Offset Newton did not converge')
def independent_root(offset,h,y,settings):
    from scipy.optimize import brentq
    from scipy.special import expit
    penalty=settings['l2_lambda'];bound=float(np.mean(np.abs(h))/penalty+1.)
    def gradient(gamma):return float(np.dot(h,expit(offset+gamma*h)-y)/len(y)+penalty*gamma)
    assert gradient(-bound)<0<gradient(bound)
    root,result=brentq(gradient,-bound,bound,xtol=1e-13,rtol=1e-14,maxiter=200,full_output=True)
    z=offset+root*h;loss=float(np.mean(np.logaddexp(0,-z)*y+np.logaddexp(0,z)*(1-y))+.5*penalty*root*root)
    return dict(gamma=float(root),objective=loss,gradient=gradient(root),converged=bool(result.converged),iterations=int(result.iterations),lower=-bound,upper=bound)

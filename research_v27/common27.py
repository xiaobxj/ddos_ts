"""Matched-update training-memory experiment; immutable R26 feature recipes."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache';V26=PROJECT/'research_v26'
sys.path.insert(0,str(V26));import common26 as prior
np=prior.np;pd=prior.pd;torch=prior.torch;legacy=prior.legacy;training=prior.training;neural=prior.neural
read=prior.read;save=prior.save;sha=prior.sha;metric=prior.metric;design=prior.design;probability=prior.probability;objective=prior.objective
data=prior.data;scales_for=prior.scales_for;arrays=prior.arrays;ref=prior.ref;market=prior.market;gates=prior.gates
fit_pipeline=prior.fit_pipeline;apply_pipeline=prior.apply_pipeline;fit_clip=prior.fit_clip;load_model=prior.load_model;forecasts=prior.forecasts
statistics=prior.statistics;METHODS=prior.METHODS;LEARNED=prior.LEARNED;PRIMARY=prior.PRIMARY
HISTORIES=['full','rolling5','rolling3'];NEW=HISTORIES[1:]
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')
def source_hashes():
    r=prior.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.rglob('*.py')});return r
def input_hashes():
    r=prior.input_hashes();paths=[V26/'protocol.json']+[V26/'results'/n for n in ['models.json','heads.json','model_predictions.csv','ensemble_predictions.csv','training_curves.csv','verification.json','delivery_manifest.json']]
    paths += [PROJECT/'research_v25/results/model_predictions.csv']
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V26/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V26/'results/delivery_manifest.json')['files'].items():r[str((V26/n).relative_to(PROJECT))]=d
    p=V26/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==3813
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json')
    assert p['protocol_sha256']==sha(ROOT/'protocol.json') and p['source_sha256']==source_hashes() and p['input_sha256']==input_hashes()
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:assert read(OUT/'contract_verification.json')['status']=='PASS'
    return p
def check_phase(phase):
    p=read(OUT/f'{phase}_manifest.json');assert p.get('finished_utc')
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    return p
def indices(obs,fold,history):
    full,te=prior.indices(obs,fold)
    if history=='full':return full,te
    years=int(history.removeprefix('rolling'));lower=f"{int(fold['cutoff'][:4])-years+1}-01-01"
    tr=full[obs.date.iloc[full].ge(lower).to_numpy()]
    expected=cfg()['training_counts'][history][cfg()['folds'].index(fold)];assert len(tr)==expected and not len(np.intersect1d(tr,te))
    return tr,te
def load_training(history,cutoff):
    with np.load(CACHE/f'training_{history}_{cutoff}.npz') as d:
        values={k:torch.from_numpy(d[k].copy()).cuda() for k in ['patches','geometry','valid']};labels={k:torch.from_numpy(d[k].copy()).cuda() for k in ['returns','auxiliary']}
        return values,labels,d['row_index'].copy()
def schedule(count,presentations,rng):return legacy.epoch_order(count,presentations,rng)
def predict_rows(obs,rows,history,cutoff,method,seed,score,prob=None):
    r=prior.rows_for_predictions(obs,rows,cutoff,method,seed,score,prob);r.insert(0,'history',history);return r
def ensemble_from(models):
    result=[]
    for history,g in models.groupby('history',sort=False):
        e=prior.ensemble_from(g.drop(columns='history'));e.insert(0,'history',history);result.append(e)
    return pd.concat(result,ignore_index=True)
def frozen_baselines():
    m=pd.read_csv(V26/'results/model_predictions.csv',float_precision='round_trip');m.insert(0,'history','full')
    e=pd.read_csv(V26/'results/ensemble_predictions.csv',float_precision='round_trip');e.insert(0,'history','full');return m,e
def calibration_availability(obs):
    # Membership/provenance audit only; no fitted calibration or performance selection.
    old=pd.read_csv(PROJECT/'research_v25/results/model_predictions.csv');later=pd.read_csv(V26/'results/model_predictions.csv')
    rows=[];members=[];old=old[old.method.isin(LEARNED)];later=later[later.method.isin(LEARNED)]
    pool=pd.concat([old[['row_index','date','joint_completed','cutoff','method','seed']],later[['row_index','date','joint_completed','cutoff','method','seed']]],ignore_index=True)
    assert not pool.duplicated(['method','seed','date']).any()
    for fold in cfg()['folds']:
        cutoff=fold['cutoff'];lower=f'{int(cutoff[:4])-2}-01-01';s=pool[pool.date.ge(lower)&pool.date.le(cutoff)&pool.joint_completed.le(cutoff)]
        assert s.cutoff.lt(s.date).all();weeksets=[]
        for (method,seed),g in s.groupby(['method','seed']):weeksets.append(set(g.date))
        assert len(weeksets)==12 and all(x==weeksets[0] for x in weeksets)
        for history in HISTORIES:
            if history=='full':g=s[s.method.eq(LEARNED[0])&s.seed.eq(cfg()['seeds'][0])][['row_index','date','joint_completed','cutoff']]
            else:
                parts=[]
                for past in cfg()['folds']:
                    if past['cutoff']>=cutoff:continue
                    _,ids=indices(obs,past,history);eligible=obs.iloc[ids];eligible=eligible[eligible.date.ge(lower)&eligible.date.le(cutoff)&eligible.joint_completed.le(cutoff)]
                    part=eligible[['date','joint_completed']].copy();part['row_index']=part.index;part['cutoff']=past['cutoff'];parts.append(part)
                g=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['row_index','date','joint_completed','cutoff'])
            rows.append(dict(history=history,prediction_year=int(cutoff[:4])+1,cutoff=cutoff,window_start=lower,available_weeks=len(g),available_signal_years=int(g.date.str[:4].nunique()),three_signal_years_available=bool(g.date.str[:4].nunique()==3),status='frozen_existing_forecasts' if history=='full' else 'planned_forecasts_from_current_fixed_folds',calibration_fits=0))
            members.extend(dict(history=history,outer_cutoff=cutoff,**r) for r in g.to_dict('records'))
    return pd.DataFrame(rows),pd.DataFrame(members)

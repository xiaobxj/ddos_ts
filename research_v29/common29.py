"""One prespecified, causal market-state trigger using immutable R28 models."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache';V28=PROJECT/'research_v28'
sys.path.insert(0,str(V28));import common28 as previous_round
prior=previous_round.prior
np=prior.np;pd=prior.pd;torch=prior.torch;legacy=prior.legacy;training=prior.training;neural=prior.neural
read=prior.read;save=prior.save;sha=prior.sha;metric=prior.metric;design=prior.design;probability=prior.probability
data=prior.data;arrays=prior.arrays;ref=prior.ref;market=prior.market;gates=prior.gates;fit_clip=prior.fit_clip;load_model=prior.load_model;forecasts=prior.forecasts;apply_pipeline=prior.apply_pipeline
statistics=prior.statistics;METHODS=prior.METHODS;LEARNED=prior.LEARNED;PRIMARY=prior.PRIMARY
HISTORIES=['rolling5_matched','rolling5_annual20','rolling5_quarterly20','state90'];NEW=['state90']
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')
def source_hashes():
    r=previous_round.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.rglob('*.py')});return r
def input_hashes():
    r=previous_round.input_hashes();paths=[V28/'protocol.json']+[V28/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','models.json','heads.json','training_scales.json','budgets.csv','verification.json','unique_model_predictions.csv','model_predictions.csv','ensemble_predictions.csv']]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V28/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V28/'results/delivery_manifest.json')['files'].items():r[str((V28/n).relative_to(PROJECT))]=d
    p=V28/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==4522
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json');assert p['protocol_sha256']==sha(ROOT/'protocol.json') and p['source_sha256']==source_hashes() and p['input_sha256']==input_hashes()
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:assert read(OUT/'contract_verification.json')['status']=='PASS'
    return p
def check_phase(phase):
    p=read(OUT/f'{phase}_manifest.json');assert p.get('finished_utc')
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    return p
def fold_for(cutoff):return next(f for f in previous_round.cfg()['folds'] if f['cutoff']==cutoff)
def training_rows(obs,cutoff):
    lower=str((pd.Timestamp(cutoff)-pd.DateOffset(years=5)+pd.Timedelta(days=1)).date())
    return np.flatnonzero(obs.date.ge(lower)&obs.joint_completed.le(cutoff))
def transformed(mf,scaler):return (np.clip(mf,scaler['lower'],scaler['upper'])-scaler['mean'])/scaler['sd']
def rolling_means(z):
    n=cfg()['state']['observations'];cs=np.vstack([np.zeros((1,z.shape[1])),np.cumsum(z,axis=0)]);return (cs[n:]-cs[:-n])/n
def assess(obs,mf,active,decision):
    tr=training_rows(obs,active);scaler=fit_clip(mf[tr]);lower=fold_for(active)['lower']['rolling5'];hist=np.flatnonzero(obs.date.ge(lower)&obs.date.le(active));current=np.flatnonzero(obs.date.le(decision))[-cfg()['state']['observations']:]
    lag=int(current[-1]-hist[-1]);assert lag>0 and len(current)==60
    z=transformed(mf[hist],scaler);means=rolling_means(z);dif=means[lag:]-means[:-lag];distances=np.mean(dif*dif,axis=1);assert len(distances)>=252
    refmean=z[-60:].mean(0);currentmean=transformed(mf[current],scaler).mean(0);distance=float(np.mean((currentmean-refmean)**2));threshold=float(np.quantile(distances,.9,method='linear'))
    saved={k:scaler[k] for k in ['lower','upper','mean','sd']};saved.update(training_rows=tr,history_rows=hist,current_rows=current,reference_rows=hist[-60:],calibration_end_rows=hist[59+lag:],calibration_start_rows=hist[59:-lag],calibration_distances=distances,reference_mean=refmean,current_mean=currentmean)
    event=dict(cutoff=decision,previous_model_cutoff=active,selected_model_cutoff=decision if distance>threshold else active,mandatory=False,refit=distance>threshold,distance=distance,threshold=threshold,ratio=distance/threshold,lag_observations=lag,calibration_n=len(distances),historical_percentile=float((distances<distance).mean()),reference_last_date=obs.date.iloc[hist[-1]],current_last_date=obs.date.iloc[current[-1]])
    return event,saved
def build_policy(obs,mf):
    # This interface receives date/maturity metadata and observed descriptors only.
    assert set(obs.columns)=={'date','joint_completed'}
    active=None;events=[];states={}
    for cutoff in cfg()['decision_dates']:
        if cutoff.endswith('12-31'):
            event=dict(cutoff=cutoff,previous_model_cutoff=active or '',selected_model_cutoff=cutoff,mandatory=True,refit=True,distance=np.nan,threshold=np.nan,ratio=np.nan,lag_observations=0,calibration_n=0,historical_percentile=np.nan,reference_last_date='',current_last_date='');active=cutoff
        else:
            event,d=assess(obs,mf,active,cutoff);states[cutoff]=d;active=event['selected_model_cutoff']
        events.append(event)
    return pd.DataFrame(events),states
def route_from_events(obs,events):
    ids=np.flatnonzero(obs.date.ge('2021-01-01')&obs.date.le(cfg()['label_end'])&obs.weekday.eq(4)&obs.joint_completed.le(cfg()['label_end']));assert len(ids)==272
    rows=[]
    for i in ids:
        e=events[events.cutoff.lt(obs.date.iloc[i])].iloc[-1];rows.append(dict(history='state90',row_index=int(i),date=obs.date.iloc[i],decision_cutoff=e.cutoff,cutoff=e.selected_model_cutoff))
    return pd.DataFrame(rows)
def baselines():
    m=pd.read_csv(V28/'results/model_predictions.csv',float_precision='round_trip');e=pd.read_csv(V28/'results/ensemble_predictions.csv',float_precision='round_trip')
    return m[m.history.isin(HISTORIES[:-1])].reset_index(drop=True),e[e.history.isin(HISTORIES[:-1])].reset_index(drop=True)
def archive_predictions():
    d=pd.read_csv(V28/'results/unique_model_predictions.csv',float_precision='round_trip');return d[d.history.eq('rolling5')].reset_index(drop=True)
def required_extensions(obs,events):
    archive=pd.read_csv(V28/'results/unique_model_predictions.csv',usecols=['history','cutoff','row_index']);archive=archive[archive.history.eq('rolling5')];available=set(zip(archive.cutoff,archive.row_index));route=route_from_events(obs,events)
    return route[[((c,int(i)) not in available) for c,i in zip(route.cutoff,route.row_index)]][['cutoff','row_index','date']].reset_index(drop=True)
def predict_rows(obs,rows,cutoff,method,seed,score,prob=None):return previous_round.predict_rows(obs,rows,'rolling5',cutoff,method,seed,score,prob)
def route_predictions(unique):
    route=csv('routing');g=unique.drop(columns='history').merge(route[['row_index','date','cutoff']],on=['row_index','date','cutoff'],how='inner',validate='many_to_one');g.insert(0,'history','state90');assert len(g)==4352 and not g.duplicated(['method','seed','date']).any();return g
def ensemble_from(models):return previous_round.ensemble_from(models)

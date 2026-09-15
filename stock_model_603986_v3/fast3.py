"""Small H1 context experiment: matched members, no neural changes."""
from experiment3 import *
from scipy.special import expit
import requests

def prepare_context():
    folder=ROOT/'data';cni=read(folder/'probes/cni980017_official.json');d=cni['data']
    assert cni['code']==200 and d['indexCode']=='980017' and d['indexName']=='国证芯片'
    sf=pd.DataFrame(d['data'],columns=d['item'])[['timestamp','close']].rename(columns={'timestamp':'date'}).sort_values('date').reset_index(drop=True)
    assert len(sf)==cni['total'] and sf.date.is_unique and sf.close.gt(0).all()
    broad=load_csv(old.CALENDAR)[['date','close']]
    recent=read(folder/'probes/sh000300_2026.json')['data']['sh000300']['day']
    rr=pd.DataFrame([(v[0],float(v[2])) for v in recent],columns=['date','close'])
    common=broad[broad.date.isin(rr.date)].merge(rr,on='date',suffixes=['_old','_new'])
    np.testing.assert_allclose(common.close_old,common.close_new,rtol=0,atol=1e-9)
    broad=pd.concat([broad,rr[rr.date.gt(broad.date.max())]],ignore_index=True)
    frame=data();external=pd.DataFrame({'date':frame.date}).merge(broad.rename(columns={'close':'csi300'}),on='date',how='left').merge(sf.rename(columns={'close':'chips980017'}),on='date',how='left')
    assert external[['csi300','chips980017']].notna().all().all() and external.date.iloc[-1]==cfg()['data_end']
    csv(folder/'external_close.csv',external)
    spec=dict(created_utc=now(),sector_name=d['indexName'],sector_code=d['indexCode'],sector_source='https://hq.cnindex.com.cn/market/market/getIndexDailyDataWithDataFormat',
        sector_identity='https://www.cnindex.com.cn/docs/gz_980017.pdf',sector_rows=len(sf),first_sector_date=sf.date.min(),last_sector_date=sf.date.max(),
        source_vintage='current official historical index series; no reconstruction from present constituents; historical revisions cannot be excluded',
        features=dict(own=['trend60','vol20','range20','volume_change20','return1','return5','today_gap','today_intraday'],
            each_external=['index_return1','index_trend60','index_vol20','beta60_adjusted_relative_strength20']),
        relative='beta=trailing60 cov(stock,index)/var(index); residual standardized over same trailing60, last20 sum / (std60*sqrt20)',
        members='same input-valid H1 rows with finite all-context features; lower 5-year anchor bound; mature by annual cutoff',
        explicit_no_duplicate='no redundant raw-return difference columns appended alongside their two original returns',
        before_fit=True,files={rel(folder/'external_close.csv'):sha(folder/'external_close.csv'),rel(folder/'probes/cni980017_official.json'):sha(folder/'probes/cni980017_official.json'),rel(ROOT/'fast3.py'):sha(ROOT/'fast3.py')})
    save(OUT/'fast_freeze.json',spec)

def feature_matrix(frame,external,obs):
    r,_,_=modules();anchors=obs.anchor.to_numpy(int);own=r.market(frame,obs,np.arange(len(obs)))
    stock=frame.close.to_numpy(float);op=frame.open.to_numpy(float)
    rr=np.r_[np.nan,np.diff(np.log(stock))];more=[];contexts={'market':[],'sector':[]}
    cols={'market':'csi300','sector':'chips980017'}
    for t in anchors:
        more.append([rr[t],rr[t-4:t+1].sum(),np.log(op[t]/stock[t-1]),np.log(stock[t]/op[t])])
        sr=rr[t-59:t+1]
        for name,col in cols.items():
            close=external[col].to_numpy(float);ir=np.diff(np.log(close[t-60:t+1]))
            sd=max(float(ir.std()),1e-8);beta=float(np.mean((sr-sr.mean())*(ir-ir.mean()))/(sd*sd))
            resid=sr-beta*ir;rsd=max(float(resid.std()),1e-8)
            contexts[name].append([ir[-1],ir.sum()/(sd*np.sqrt(60)),float(ir[-20:].std()),resid[-20:].sum()/(rsd*np.sqrt(20))])
    x=np.c_[own,np.asarray(more)];market=np.asarray(contexts['market']);sector=np.asarray(contexts['sector'])
    return dict(own=x,market=np.c_[x,market],sector=np.c_[x,sector],both=np.c_[x,market,sector])

def main():
    check_freeze();prepare_context();f=data();obs,_=observations(f,1);ext=load_csv(ROOT/'data/external_close.csv')
    features=feature_matrix(f,ext,obs);finite=np.isfinite(features['both']).all(1);r,_,_=modules()
    fitrefs=[];outputs=[];checks=[]
    # All fits first; outcomes are scored only after the fitted parameters are saved.
    for cutoff in [f'{y}-12-31' for y in range(2022,2026)]:
        ids,_=membership(obs,cutoff,5);ids=ids[finite[ids]];assert len(ids)>=400
        y=obs.actual_up.iloc[ids].to_numpy(float);test=np.flatnonzero(obs.date.gt(cutoff)&obs.date.le(f'{int(cutoff[:4])+1}-12-31')&finite)
        folder=OUT/'fast'/cutoff;folder.mkdir(parents=True,exist_ok=False);csv(folder/'members.csv',obs.iloc[ids])
        prefix=f[f.date.le(cutoff)].reset_index(drop=True);po,_=observations(prefix,1);pf=feature_matrix(prefix,ext.iloc[:len(prefix)],po)
        for name,x in features.items():
            np.testing.assert_array_equal(x[ids],pf[name][ids])
            d=r.fit_clip(x[ids]);theta,_=r.fit_newton(d['standardized'],y,r.cfg()['probe']);xt=r.transform(x,d)
            grad=np.c_[d['standardized'],np.ones(len(ids))].T@(expit(np.c_[d['standardized'],np.ones(len(ids))]@theta)-y)/len(ids)+.01*np.r_[theta[:-1],0.]
            assert abs(grad).max()<3e-9
            np.savez_compressed(folder/(name+'.npz'),theta=theta,ids=ids,**d)
            out=obs.iloc[test].copy();out['name']=name;out['cutoff']=cutoff;out['probability']=expit(np.c_[xt[test],np.ones(len(test))]@theta);outputs.append(out)
            fitrefs.append(dict(cutoff=cutoff,name=name,n=len(ids),dimension=x.shape[1],rank=int(np.linalg.matrix_rank(np.c_[d['standardized'],np.ones(len(ids))])),gradient_inf=float(abs(grad).max())))
        out=obs.iloc[test].copy();out['name']='frequency';out['cutoff']=cutoff;out['probability']=y.mean();outputs.append(out)
    save(OUT/'fast_fits_completed.json',dict(completed_utc=now(),fits=fitrefs))
    from slow3 import metric
    pred=pd.concat(outputs,ignore_index=True);metrics=[];years=[]
    common=obs[obs.matured&obs.date.ge('2023-01-01')&finite].date.tolist()
    for name,g in pred[pred.matured].groupby('name'):
        assert g.date.tolist()==common
        metrics.append(dict(name=name,**metric(g.actual_up,g.probability)))
        for year,gg in g.groupby('year'):years.append(dict(name=name,year=int(year),**metric(gg.actual_up,gg.probability)))
    csv(OUT/'fast_predictions.csv',pred);csv(OUT/'fast_metrics.csv',pd.DataFrame(metrics));csv(OUT/'fast_years.csv',pd.DataFrame(years))
    save(OUT/'fast_verified.json',dict(status='PASS',completed_utc=now(),train_prefix_checks=16,fit_gradient_checks=16,common_test_n=len(common),
        all_training_rows_have_context=bool(finite[obs.matured].all()),source='official CNI close prices and Tencent CSI300; OHLC/volume missing in older CNI records unused'))

if __name__=='__main__':main()

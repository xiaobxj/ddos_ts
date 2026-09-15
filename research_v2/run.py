"""Frozen second-round development experiment. Does not overwrite v1 evidence."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
from core import ROOT,V1,observation_table,static_features,prototype_features,fold_masks,ridge_path,logistic_path

OUT=ROOT/'results'
METHODS=[
    dict(name='a_close_annual',feature='legacy',target='close_return',memory=0,cadence='annual',model='ridge'),
    dict(name='b_exec_annual',feature='legacy',target='exec_return',memory=0,cadence='annual',model='ridge'),
    dict(name='c_econ_annual',feature='econ',target='exec_return',memory=0,cadence='annual',model='ridge'),
    dict(name='d_multi_exp_annual',feature='multi',target='exec_return',memory=0,cadence='annual',model='ridge'),
    dict(name='e_adaptive_annual',feature='adaptive',target='exec_return',memory=0,cadence='annual',model='ridge'),
    dict(name='f_multi_exp_quarter',feature='multi',target='exec_return',memory=0,cadence='quarter',model='ridge'),
    dict(name='g_multi_3y_annual',feature='multi',target='exec_return',memory=3,cadence='annual',model='ridge'),
    dict(name='h_multi_5y_annual',feature='multi',target='exec_return',memory=5,cadence='annual',model='ridge'),
    dict(name='i_multi_3y_quarter',feature='multi',target='exec_return',memory=3,cadence='quarter',model='ridge'),
    dict(name='j_multi_5y_quarter',feature='multi',target='exec_return',memory=5,cadence='quarter',model='ridge'),
    dict(name='k_logit_exp_annual',feature='multi',target='exec_return',memory=0,cadence='annual',model='logistic'),
    dict(name='l_logit_5y_quarter',feature='multi',target='exec_return',memory=5,cadence='quarter',model='logistic'),
]


def hash_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def previous_evidence():
    return {str(p.relative_to(V1)):hash_file(p) for p in V1.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}


def periods(start_year,end_year,cadence):
    for year in range(start_year,end_year+1):
        for month in ([1] if cadence=='annual' else [1,4,7,10]):
            start=pd.Timestamp(year,month,1)
            end=start+pd.DateOffset(months=12 if cadence=='annual' else 3)-pd.Timedelta(days=1)
            yield (start-pd.Timedelta(days=1)).strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'),year


def main():
    OUT.mkdir(exist_ok=True)
    started=time.time()
    cfg=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    saved=dict(started_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol=cfg,methods=METHODS,
               protocol_sha256=hash_file(ROOT/'protocol.json'),
               code_sha256={p.name:hash_file(p) for p in ROOT.glob('*.py')},
               v1_sha256=previous_evidence(),python=sys.version,executable=sys.executable,
               platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__)
    (OUT/'run_manifest.json').write_text(json.dumps(saved,indent=2),encoding='utf-8')
    frame=pd.read_csv(V1/'data/1_000300.csv')
    obs=observation_table(frame)
    obs.to_csv(OUT/'observation_table.csv',index=False)
    assert np.isfinite(obs[['close_return','exec_return']].to_numpy()).all()
    base_features=static_features(frame,obs)
    cache={}
    def features(feature,year,phase):
        if feature in base_features:
            return base_features[feature]
        key='validation' if phase=='validation' else year
        if key not in cache:
            filename='validation_prototypes.npz' if key=='validation' else f'prototypes_{year}.npz'
            prototypes=np.load(V1/'results'/filename)['prototypes']
            cache[key]=prototype_features(frame,obs,prototypes)
        return cache[key][feature]
    old=pd.read_csv(V1/'results/predictions.csv')
    evaluation_anchors=old[old.method=='adaptive_normalized'].anchor.to_numpy(int)
    assert len(evaluation_anchors)==272
    assert set(evaluation_anchors).issubset(set(obs.anchor))
    development_allowed=obs.anchor.isin(evaluation_anchors).to_numpy()
    validation_allowed=((obs.date>='2018-01-01')&(obs.date<='2020-12-31')&(obs.completed<='2020-12-31')).to_numpy()
    selected,all_validation,fold_audit,solver_audit={},[],[],[]
    # Compute validation paths for every arm before inspecting any development score.
    for method in METHODS:
        records={str(l):[] for l in cfg['lambdas']}
        for cutoff,end,year in periods(2018,2020,method['cadence']):
            tr,te=fold_masks(obs,cutoff,end,method['memory'])
            te &= validation_allowed
            if not te.any():
                continue
            x=features(method['feature'],year,'validation')
            y=obs[method['target']].to_numpy()
            if method['model']=='ridge':
                path=ridge_path(x[tr],y[tr],x[te],cfg['lambdas'])
            else:
                path,solver=logistic_path(x[tr],y[tr],x[te],cfg['lambdas'])
                solver_audit.extend([dict(phase='validation',method=method['name'],cutoff=cutoff,**s) for s in solver])
            train_exec=obs.exec_return.to_numpy()[tr]
            up=train_exec[train_exec>0].mean()
            down=train_exec[train_exec<=0].mean()
            for lam,prediction in path.items():
                for ix,value in zip(np.flatnonzero(te),prediction):
                    probability=float(value) if method['model']=='logistic' else np.nan
                    pred_return=probability*up+(1-probability)*down if method['model']=='logistic' else float(value)
                    position=int(value>.5) if method['model']=='logistic' else int(value>0)
                    records[lam].append(dict(method=method['name'],lam=float(lam),anchor=int(obs.anchor.iloc[ix]),
                                             date=obs.date.iloc[ix],actual=float(obs.exec_return.iloc[ix]),
                                             predicted_return=pred_return,probability=probability,position=position,
                                             cutoff=cutoff,train_n=int(tr.sum())))
            fold_audit.append(dict(phase='validation',method=method['name'],cutoff=cutoff,period_end=end,
                                   train_n=int(tr.sum()),first_train=obs.date[tr].min(),last_label=obs.completed[tr].max(),
                                   first_signal=obs.date[te].min(),last_signal=obs.date[te].max(),test_n=int(te.sum()),
                                   feature_count=x.shape[1],memory_years=method['memory'],cadence=method['cadence']))
        rankings=[]
        for lam,rows in records.items():
            g=pd.DataFrame(rows).sort_values('date')
            mse=float(np.mean((g.predicted_return-g.actual)**2))
            correct=float((g.position.to_numpy()==(g.actual.to_numpy()>0)).mean())
            p=g.probability.to_numpy()
            label=(g.actual>0).to_numpy(float)
            pc=np.clip(p,1e-12,1-1e-12)
            logloss=float(-np.mean(label*np.log(pc)+(1-label)*np.log1p(-pc))) if method['model']=='logistic' else np.nan
            criterion=logloss if method['model']=='logistic' else mse
            rankings.append((criterion,-float(lam),float(lam),mse))
            all_validation.extend(rows)
            print(f"validation {method['name']} lambda={lam}: rmse={np.sqrt(mse):.5f}, accuracy={correct:.3f}",flush=True)
        best=sorted(rankings)[0]
        selected[method['name']]=dict(lam=best[2],criterion=float(best[0]),return_rmse=float(np.sqrt(best[3])),
                                     feature=method['feature'],model=method['model'],memory=method['memory'],cadence=method['cadence'])
    selectable=[m for m in METHODS if m['model']=='ridge' and m['feature']!='legacy']
    chosen=min(selectable,key=lambda m:(selected[m['name']]['return_rmse'],m['name']))['name']
    decision=dict(selected_lambdas=selected,validation_selected_pipeline=chosen,
                  selected_before_development=pd.Timestamp.now(tz='UTC').isoformat())
    (OUT/'selection.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    pd.DataFrame(all_validation).to_csv(OUT/'validation_predictions.csv',index=False)
    print(f"FROZEN validation-selected pipeline: {chosen}; elapsed {time.time()-started:.1f}s",flush=True)
    # No tuning beyond this point.
    records=[]
    for method in METHODS:
        lam=selected[method['name']]['lam']
        for cutoff,end,year in periods(2021,2026,method['cadence']):
            tr,te=fold_masks(obs,cutoff,end,method['memory'])
            te &= development_allowed
            if not te.any():
                continue
            x=features(method['feature'],year,'development')
            y=obs[method['target']].to_numpy()
            if method['model']=='ridge':
                prediction=ridge_path(x[tr],y[tr],x[te],[lam])[str(lam)]
            else:
                path,solver=logistic_path(x[tr],y[tr],x[te],[lam])
                prediction=path[str(lam)]
                solver_audit.extend([dict(phase='development',method=method['name'],cutoff=cutoff,**s) for s in solver])
            train_exec=obs.exec_return.to_numpy()[tr]
            up=train_exec[train_exec>0].mean()
            down=train_exec[train_exec<=0].mean()
            for ix,value in zip(np.flatnonzero(te),prediction):
                row=obs.iloc[ix]
                probability=float(value) if method['model']=='logistic' else np.nan
                pred_return=probability*up+(1-probability)*down if method['model']=='logistic' else float(value)
                position=int(value>.5) if method['model']=='logistic' else int(value>0)
                records.append(dict(method=method['name'],date=row.date,anchor=int(row.anchor),year=int(row.date[:4]),
                                     entry=int(row.entry),exit=int(row.exit),exit_date=row.exit_date,
                                     actual=float(row.exec_return),close_actual=float(row.close_return),
                                     predicted_return=pred_return,probability=probability,position=position,
                                     correct=int(position==int(row.exec_return>0)),cutoff=cutoff,lam=lam,train_n=int(tr.sum())))
            fold_audit.append(dict(phase='development',method=method['name'],cutoff=cutoff,period_end=end,
                                   train_n=int(tr.sum()),first_train=obs.date[tr].min(),last_label=obs.completed[tr].max(),
                                   first_signal=obs.date[te].min(),last_signal=obs.date[te].max(),test_n=int(te.sum()),
                                   feature_count=x.shape[1],memory_years=method['memory'],cadence=method['cadence']))
        print(f"development {method['name']} complete; elapsed {time.time()-started:.1f}s",flush=True)
    template=pd.DataFrame(records).query("method=='b_exec_annual'").copy().sort_values('date')
    for name,pos,pred in [('always_up',np.ones(len(template)),np.full(len(template),1e-9)),
                           ('cash',np.zeros(len(template)),np.zeros(len(template)))]:
        g=template.copy()
        g['method']=name
        g['position']=pos.astype(int)
        g['predicted_return']=pred
        g['correct']=(g.position==(g.actual>0)).astype(int)
        g['lam']=np.nan
        records.extend(g.to_dict('records'))
    for source in ['adaptive_normalized','fixed_10']:
        prior=old[old.method==source].set_index('anchor').loc[template.anchor]
        g=template.copy()
        g['method']='v1_'+source
        g['position']=prior.position.to_numpy(int)
        g['predicted_return']=prior.prediction.to_numpy(float)
        g['correct']=(g.position==(g.actual>0)).astype(int)
        g['lam']=np.nan
        # Preserve original refit provenance instead of falsely attaching v2 sample counts.
        g['cutoff']=prior.train_cutoff.to_numpy()
        g['train_n']=np.nan
        records.extend(g.to_dict('records'))
    all_predictions=pd.DataFrame(records)
    chosen_rows=all_predictions[all_predictions.method==chosen].copy()
    for bps in [10,20,40]:
        g=chosen_rows.copy()
        g['method']=f'selected_threshold_{bps}bps'
        g['position']=(g.predicted_return>bps/10000).astype(int)
        g['correct']=(g.position==(g.actual>0)).astype(int)
        all_predictions=pd.concat([all_predictions,g],ignore_index=True)
    all_predictions.to_csv(OUT/'predictions.csv',index=False)
    (OUT/'folds.json').write_text(json.dumps(fold_audit,indent=2),encoding='utf-8')
    (OUT/'solver_audit.json').write_text(json.dumps(solver_audit,indent=2),encoding='utf-8')
    assert previous_evidence()==saved['v1_sha256'],'v1 evidence was changed'
    saved['finished_utc']=pd.Timestamp.now(tz='UTC').isoformat()
    saved['elapsed_seconds']=time.time()-started
    saved['v1_preserved']=True
    (OUT/'run_manifest.json').write_text(json.dumps(saved,indent=2),encoding='utf-8')
    print(f"Completed {len(all_predictions)} predictions; v1 preserved",flush=True)


if __name__=='__main__':
    main()

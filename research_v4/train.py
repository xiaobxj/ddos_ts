"""Chronological epoch selection then frozen seven-arm Crossformer experiment."""
from util import *
from data import packed_path
from architecture import SegmentedCrossformer
import torch
import time
import platform
from contextlib import nullcontext


def architecture_config():
    config=cfg()['crossformer']
    return {key:config[key] for key in ['d_model','d_ff','n_heads','e_layers','factor','dropout']}


def select_rows(obs,cutoff,end,allowed):
    train=(obs.completed<=cutoff).to_numpy()
    test=((obs.date>cutoff)&(obs.date<=end)&(obs.weekday==4)).to_numpy() & allowed
    assert train.any() and test.any()
    return np.flatnonzero(train),np.flatnonzero(test)


def fit(arm,arrays,obs,train_ids,test_ids,seed,epochs,checkpoints,cutoff,phase):
    settings=cfg()['training']
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    network=SegmentedCrossformer(arm['dimensions'],**architecture_config()).cuda()
    optimizer=torch.optim.AdamW(network.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    yt=obs.exec_return.to_numpy(float)[train_ids]
    mean=float(yt.mean());sd=float(yt.std())
    assert sd>1e-8
    y=torch.tensor((yt-mean)/sd,dtype=torch.float32,device='cuda')
    inputs={key:torch.from_numpy(arrays[key][train_ids]).cuda() for key in ['patches','geometry','valid']}
    testing={key:torch.from_numpy(arrays[key][test_ids]).cuda() for key in inputs}
    permutation=np.random.default_rng(seed)
    predictions={};curves=[];start=time.time()
    for epoch in range(1,epochs+1):
        network.train();ids=permutation.permutation(len(train_ids));total=0.
        for offset in range(0,len(ids),settings['batch_size']):
            batch=torch.as_tensor(ids[offset:offset+settings['batch_size']],device='cuda')
            optimizer.zero_grad(set_to_none=True)
            output=network(inputs['patches'][batch],inputs['geometry'][batch],inputs['valid'][batch])
            loss=((output-y[batch])**2).mean()
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),settings['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step()
            total+=float(loss.detach())*len(batch)
        record=dict(phase=phase,method=arm['name'],cutoff=cutoff,seed=seed,epoch=epoch,
                    train_mse_standardized=total/len(train_ids),elapsed_seconds=time.time()-start)
        if epoch in checkpoints:
            network.eval()
            with torch.inference_mode():
                output=network(testing['patches'],testing['geometry'],testing['valid']).cpu().numpy().astype(float)*sd+mean
            assert np.isfinite(output).all()
            predictions[epoch]=output
            record['evaluation_rmse']=float(np.sqrt(np.mean((output-obs.exec_return.to_numpy()[test_ids])**2))) if phase=='validation' else None
        curves.append(record)
    checkpoint=OUT/'checkpoints'/f"{phase}_{arm['name']}_{cutoff}_{seed}.pt"
    torch.save(dict(state_dict=network.cpu().state_dict(),dimensions=arm['dimensions'],architecture=architecture_config(),
                    target_mean=mean,target_sd=sd,seed=seed,epochs=epochs,cutoff=cutoff,train_n=len(train_ids),
                    protocol_sha256=sha(ROOT/'protocol.json')),checkpoint)
    del network,optimizer,inputs,testing,y;torch.cuda.empty_cache()
    return predictions,curves,dict(checkpoint=str(checkpoint.relative_to(ROOT)),sha256=sha(checkpoint),
                                  target_mean=mean,target_sd=sd,parameters=138441 if arm['dimensions']==5 else 150996)


def main():
    initialize_torch()
    (OUT/'checkpoints').mkdir(exist_ok=True)
    config=cfg();start=time.time()
    preparation=json.loads((OUT/'preparation_manifest.json').read_text(encoding='utf-8'))
    assert preparation.get('finished_utc')
    assert preparation['protocol_sha256']==sha(ROOT/'protocol.json')
    assert preparation['previous_evidence']==previous_evidence()
    manifest=dict(started_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),
                  source_sha256={name:sha(ROOT/name) for name in ['train.py','architecture.py','util.py','data.py']},
                  official_crossformer_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'vendor/Crossformer/cross_models').glob('*.py')},
                  previous_evidence=preparation['previous_evidence'],preparation_manifest_sha256=sha(OUT/'preparation_manifest.json'),
                  torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,device=torch.cuda.get_device_name(),
                  python=sys.version,executable=sys.executable,platform=platform.platform())
    save_json(OUT/'training_manifest.json',manifest)
    obs=pd.read_csv(OUT/'observation_table.csv')
    validation=((obs.date>='2018-01-01')&(obs.date<='2020-12-31')&(obs.completed<='2020-12-31')).to_numpy()
    prior=pd.read_csv(V3/'results/predictions.csv')
    anchors=prior[(prior.method=='ridge_target')&(prior.date>='2024-07-01')].anchor
    allowed=obs.anchor.isin(anchors).to_numpy()
    assert allowed.sum()==103
    arms=config['arms'];settings=config['training'];candidates=settings['epoch_candidates']
    raw_fixed=next(a for a in arms if a['name']=='raw_fixed')
    arrays=dict(np.load(packed_path(raw_fixed,2024)))
    validation_rows=[];curve_rows=[];folds=[];checkpoint_records=[];val_mean=[]
    for year in [2018,2019,2020]:
        cutoff=f'{year-1}-12-31';end=f'{year}-12-31'
        tr,te=select_rows(obs,cutoff,end,validation)
        for seed in settings['seeds']:
            paths,curves,checkpoint=fit(raw_fixed,arrays,obs,tr,te,seed,max(candidates),candidates,cutoff,'validation')
            for epoch,prediction in paths.items():
                for i,value in zip(te,prediction):
                    validation_rows.append(dict(epoch=epoch,**prediction_row(obs.iloc[i],'raw_fixed',value,cutoff,len(tr),seed)))
            curve_rows.extend(curves);checkpoint_records.append(checkpoint)
            print(f'Validation {year} seed {seed} finished; {time.time()-start:.1f}s',flush=True)
        folds.append(dict(phase='validation',method='raw_fixed',cutoff=cutoff,end=end,train_n=len(tr),test_n=len(te),
                          first_train=obs.date.iloc[tr].min(),last_label=obs.completed.iloc[tr].max(),
                          first_signal=obs.date.iloc[te].min(),last_signal=obs.date.iloc[te].max()))
        for i in te:val_mean.append(dict(date=obs.date.iloc[i],actual=obs.exec_return.iloc[i],prediction=obs.exec_return.iloc[tr].mean()))
    val=pd.DataFrame(validation_rows)
    val.to_csv(OUT/'validation_seed_predictions.csv',index=False)
    choices=[]
    for epoch in candidates:
        ensemble=val[val.epoch==epoch].groupby('date').agg(predicted_return=('predicted_return','mean'),actual=('actual','first'))
        assert len(ensemble)==141
        choices.append(dict(epochs=epoch,rmse=float(np.sqrt(np.mean((ensemble.predicted_return-ensemble.actual)**2))),
                            accuracy=float(((ensemble.predicted_return>0)==(ensemble.actual>0)).mean())))
    chosen=min(choices,key=lambda x:(x['rmse'],x['epochs']))
    vm=pd.DataFrame(val_mean)
    selection=dict(frozen_utc=pd.Timestamp.now(tz='UTC').isoformat(),selected_epochs=chosen['epochs'],rankings=choices,
                   training_mean_validation_rmse=float(np.sqrt(np.mean((vm.prediction-vm.actual)**2))),
                   zero_validation_rmse=float(np.sqrt(np.mean(vm.actual**2))),selection_arm='raw_fixed',
                   seeds=settings['seeds'],rule=settings['selection'])
    save_json(OUT/'selection.json',selection)
    print(f'FROZEN epoch count {chosen["epochs"]}; validation candidates {choices}',flush=True)
    records=[];benchmarks=[]
    economic=np.load(V3/'results/features.npz')['target']
    np.testing.assert_array_equal(pd.read_csv(V3/'results/observation_table.csv').anchor,obs.anchor)
    for fold in config['development_folds']:
        tr,te=select_rows(obs,fold['cutoff'],fold['end'],allowed)
        ridge=ridge_path(economic[tr],obs.exec_return.to_numpy()[tr],economic[te],[10.0])['10.0']
        for name,values in [('ridge_matched_schedule',ridge),('training_mean',np.full(len(te),obs.exec_return.iloc[tr].mean())),
                            ('buy_hold',np.full(len(te),1e-9)),('zero_return_cash',np.zeros(len(te)))]:
            for i,value in zip(te,values):benchmarks.append(prediction_row(obs.iloc[i],name,value,fold['cutoff'],len(tr)))
        for arm in arms:
            arrays=dict(np.load(packed_path(arm,fold['prototype_year'])))
            for seed in settings['seeds']:
                paths,curves,checkpoint=fit(arm,arrays,obs,tr,te,seed,chosen['epochs'],[chosen['epochs']],fold['cutoff'],'development')
                for i,value in zip(te,paths[chosen['epochs']]):
                    records.append(prediction_row(obs.iloc[i],arm['name'],value,fold['cutoff'],len(tr),seed))
                curve_rows.extend(curves);checkpoint_records.append(checkpoint)
                pd.DataFrame(records).to_csv(OUT/'development_seed_predictions.csv',index=False)
                pd.DataFrame(curve_rows).to_csv(OUT/'training_curves.csv',index=False)
                print(f"Development {fold['cutoff']} {arm['name']} seed {seed}; {time.time()-start:.1f}s",flush=True)
            folds.append(dict(phase='development',method=arm['name'],**fold,train_n=len(tr),test_n=len(te),
                              first_train=obs.date.iloc[tr].min(),last_label=obs.completed.iloc[tr].max(),
                              first_signal=obs.date.iloc[te].min(),last_signal=obs.date.iloc[te].max()))
    seed_frame=pd.DataFrame(records)
    ensemble=[]
    for (name,anchor),g in seed_frame.groupby(['method','anchor'],sort=False):
        assert len(g)==3
        row=obs[obs.anchor==anchor].iloc[0]
        ensemble.append(prediction_row(row,name,g.predicted_return.mean(),g.cutoff.iloc[0],g.train_n.iloc[0]))
    for source,name in [('ridge_target','round3_quarterly_ridge'),('kronos_ohlcv','round3_direct_kronos')]:
        old=prior[(prior.method==source)&(prior.date>='2024-07-01')].copy()
        for r in old.itertuples():
            row=obs[obs.anchor==r.anchor].iloc[0]
            benchmarks.append(prediction_row(row,name,r.predicted_return,r.cutoff,int(r.train_n)))
    pd.DataFrame(ensemble+benchmarks).to_csv(OUT/'predictions.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(OUT/'training_curves.csv',index=False)
    save_json(OUT/'folds.json',folds);save_json(OUT/'checkpoint_manifest.json',checkpoint_records)
    assert previous_evidence()==manifest['previous_evidence']
    manifest.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),elapsed_seconds=time.time()-start,
                    selected_epochs=chosen['epochs'],previous_preserved=True,predictions=len(ensemble+benchmarks),
                    seed_predictions=len(seed_frame),seed_checkpoints=len(checkpoint_records),
                    max_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated())
    save_json(OUT/'training_manifest.json',manifest)
    print('Fourth-round model training and predictions completed.',flush=True)


if __name__=='__main__':
    main()

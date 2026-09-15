"""Replay every new/reused checkpoint; reconstruct masks/scales/order hashes."""
from common7 import *
import subprocess


def reconstruct(obs, cutoff, history):
    full = np.flatnonzero((obs.joint_completed<=cutoff).to_numpy())
    end = next(f['end'] for f in cfg()['folds'] if f['cutoff']==cutoff)
    te = np.flatnonzero(((obs.date>cutoff)&(obs.date<=end)&(obs.weekday==4)&(obs.joint_completed<='2020-12-31')).to_numpy())
    if history == 'full':
        return full, full, te
    lengths = np.full(5, len(full)//5); lengths[:len(full)%5] += 1
    edges = np.r_[0, np.cumsum(lengths)]
    index = {'drop_early': 0, 'drop_middle': 2, 'drop_recent': 4}[history]
    keep = np.ones(len(full), bool); keep[edges[index]:edges[index+1]] = False
    return full[keep], full, te


def independent_order(count, total, generator):
    parts = []; size = 0
    while size < total:
        p = generator.permutation(count)
        p = p[:total-size]
        parts.extend(p.tolist()); size += len(p)
    return np.asarray(parts, dtype=np.int64)


def verify_scores():
    import evaluate7 as scoring
    seed = pd.read_csv(OUT / 'all_seed_predictions.csv')
    ensemble = pd.read_csv(OUT / 'ensemble_predictions.csv')
    reference = seed.groupby(['role','history','date'])[['predicted_return','reassigned_prediction']].mean().sort_index()
    measured = ensemble.set_index(['role','history','date']).sort_index()
    np.testing.assert_allclose(reference.predicted_return, measured.predicted_return, rtol=0, atol=1e-12)
    np.testing.assert_allclose(reference.reassigned_prediction, measured.reassigned_prediction, rtol=0, atol=1e-12)
    assert len(ensemble)==2256 and seed.groupby(['role','history','date']).size().eq(3).all()
    metrics = pd.read_csv(OUT / 'ensemble_metrics.csv')
    for r in metrics.itertuples():
        g = ensemble[(ensemble.role==r.role)&(ensemble.history==r.history)]
        for name, value in scoring.metric(g).items():
            if value is not None:
                assert abs(value-getattr(r,name))<1e-12, (r.role,r.history,name)
    checks = pd.read_csv(OUT / 'candidate_comparisons.csv')
    deleted = checks[checks.history!='full']
    flags = dict(absolute_advantage=bool((deleted.mse_skill_vs_mean>0).all()),
                 relative_advantage=bool((deleted.mse_skill_vs_baseline10>0).all()),
                 seed_consistency=bool((deleted.candidate_seeds_better_than_mean>=2).all()))
    assessment = json.loads((OUT / 'stability_assessment.json').read_text(encoding='utf-8'))
    assert flags==assessment['flags'] and all(flags.values())==assessment['consistency_screen_pass']
    assert assessment['no_candidate_selection']
    return dict(ensemble_predictions=len(ensemble), ensemble_method_history_rows=len(metrics),
                candidate_selection_performed=False, consistency_flags_recomputed=True)


def main():
    legacy.initialize(); start=time.time()
    test = subprocess.run([sys.executable,str(ROOT/'test_contracts7.py')], capture_output=True,
                          text=True, encoding='utf-8', errors='replace')
    (OUT/'contract_test_output.txt').write_text(test.stdout+test.stderr,encoding='utf-8')
    assert test.returncode==0, test.stdout+test.stderr
    old = old_evidence()
    for phase in ['preparation','baseline','combined']:
        run = json.loads((OUT/f'{phase}_manifest.json').read_text(encoding='utf-8'))
        assert run.get('finished_utc') and run['protocol_sha256']==sha(ROOT/'protocol.json')
        assert run['source_sha256']==source_hashes() and run['input_sha256']==input_hashes()
        assert run['old_evidence']==old
        if phase!='preparation':
            assert run['preparation_manifest_sha256']==sha(OUT/'preparation_manifest.json')
    prep = json.loads((OUT/'preparation_manifest.json').read_text(encoding='utf-8'))
    for name,digest in prep['local_sha256'].items():
        assert sha(ROOT/name)==digest,name
    obs = pd.read_csv(OUT/'observation_table.csv')
    targets = dict(np.load(V5/'cache/targets.npz'))
    reused = pd.read_csv(OUT/'reused_full_predictions.csv')
    new_frames = [pd.read_csv(OUT/f'{arm}_predictions.csv') for arm in ['baseline','combined']]
    new = pd.concat(new_frames,ignore_index=True)
    predictions = pd.concat([reused,new],ignore_index=True)
    assert len(new)==5076 and len(reused)==1692 and len(predictions)==6768
    checkpoints = json.loads((OUT/'reused_checkpoint_manifest.json').read_text(encoding='utf-8'))
    for arm in ['baseline','combined']:
        checkpoints += json.loads((OUT/f'{arm}_checkpoint_manifest.json').read_text(encoding='utf-8'))
    assert len(checkpoints)==144
    prior_cp = json.loads((V6/'results/blocks_checkpoint_manifest.json').read_text(encoding='utf-8'))
    prior_pred = pd.read_csv(V6/'results/blocks_seed_predictions.csv')
    prefix_count=0; prefix_rows=0; max_prefix_error=0.; maximum=0.; replayed=0
    for fold in cfg()['folds']:
        for history in cfg()['histories']:
            tr,full,te=reconstruct(obs,fold['cutoff'],history)
            assert obs.joint_completed.iloc[tr].max()<=fold['cutoff']<obs.date.iloc[te].min()
            with np.load(CACHE/f'masks_{fold["cutoff"]}.npz') as masks:
                np.testing.assert_array_equal(masks[history],tr)
                np.testing.assert_array_equal(masks['full'],full)
                np.testing.assert_array_equal(masks['testing'],te)
            testing=legacy.batch_tensors(te)
            shifted={k:torch.roll(v,1,dims=0) for k,v in testing.items()}
            selected=[r for r in checkpoints if r['cutoff']==fold['cutoff'] and r['history']==history]
            assert len(selected)==12
            for r in selected:
                path=PROJECT/r['project_file']; assert sha(path)==r['sha256']
                saved=torch.load(path,map_location='cpu',weights_only=True)
                expected_protocol=sha(V6/'protocol.json') if history=='full' else sha(ROOT/'protocol.json')
                assert saved['protocol_sha256']==expected_protocol
                assert saved['train_n']==len(tr) and saved['full_train_n']==len(full)
                for key in ['arm','history','seed','epoch','cutoff']:
                    assert saved[key]==r[key]
                for key in ['returns','auxiliary']:
                    np.testing.assert_allclose(saved['scales'][key+'_mean'], targets[key][tr].mean(axis=0),rtol=0,atol=1e-12)
                    np.testing.assert_allclose(saved['scales'][key+'_sd'],np.maximum(targets[key][tr].std(axis=0),1e-6),rtol=0,atol=1e-12)
                network=legacy.make_model(r['arm'],r['seed']).eval(); network.load_state_dict(saved['state_dict'])
                with torch.inference_mode():
                    output,_=legacy.predict(network,testing,False)
                    altered,_=legacy.predict(network,shifted,False)
                p=output.cpu().numpy().astype(float)*saved['scales']['returns_sd']+saved['scales']['returns_mean']
                q=altered.cpu().numpy().astype(float)*saved['scales']['returns_sd']+saved['scales']['returns_mean']
                reference=predictions[(predictions.arm==r['arm'])&(predictions.epoch==r['epoch'])&
                                     (predictions.seed==r['seed'])&(predictions.cutoff==r['cutoff'])&(predictions.history==history)]
                np.testing.assert_array_equal(reference.row_index,te)
                np.testing.assert_allclose(reference.actual,targets['returns'][te],rtol=0,atol=1e-12)
                np.testing.assert_allclose(reference.predicted_return,p,rtol=0,atol=1e-10)
                np.testing.assert_allclose(reference.reassigned_prediction,q,rtol=0,atol=1e-10)
                maximum=max(maximum,float(np.max(np.abs(p-reference.predicted_return))),float(np.max(np.abs(q-reference.reassigned_prediction))))
                if history!='full' and r['seed']==20260910 and r['epoch']==10:
                    item=next(x for x in prior_cp if all(x[k]==r[k] for k in ['arm','history','cutoff','seed','epoch']))
                    prior=torch.load(V6/item['file'],map_location='cpu',weights_only=True)
                    assert saved['state_dict'].keys()==prior['state_dict'].keys()
                    for key,value in saved['state_dict'].items():
                        assert torch.equal(value,prior['state_dict'][key]),(item['file'],key)
                    original=prior_pred[(prior_pred.arm==r['arm'])&(prior_pred.history==history)&
                                        (prior_pred.seed==r['seed'])&(prior_pred.cutoff==r['cutoff'])]
                    np.testing.assert_allclose(p,original.predicted_return,rtol=0,atol=1e-10)
                    max_prefix_error=max(max_prefix_error,float(np.max(np.abs(p-original.predicted_return))))
                    prefix_count+=1; prefix_rows+=len(original)
                replayed+=1; del network; torch.cuda.empty_cache()
            del testing,shifted
            print(f'Replayed {replayed}/144: {fold["cutoff"]}, {history}',flush=True)
    assert prefix_count==18 and prefix_rows==846
    order_count=0; paired_orders=[]
    for arm in ['baseline','combined']:
        curves=pd.read_csv(OUT/f'{arm}_training_curves.csv')
        assert len(curves)==540
        for (cutoff,history,seed),g in curves.groupby(['cutoff','history','seed']):
            tr,full,_=reconstruct(obs,cutoff,history); rng=np.random.default_rng(seed)
            assert g.epoch.tolist()==list(range(1,21))
            for r in g.itertuples():
                order=independent_order(len(tr),len(full),rng)
                assert order_hash(tr[order])==r.training_order_sha256
                assert r.presentations==len(full) and r.train_n==len(tr)
                assert r.optimizer_steps==int(np.ceil(len(full)/128))
                order_count+=1
        paired_orders.append(curves.set_index(['cutoff','history','seed','epoch']).training_order_sha256.sort_index())
    np.testing.assert_array_equal(paired_orders[0],paired_orders[1])
    scores=verify_scores()
    result=dict(status='PASS',contract_tests=3,new_fits=54,new_checkpoints_replayed=108,
                reused_checkpoints_replayed=36,total_checkpoints_replayed=replayed,
                new_seed_predictions=len(new),reused_seed_predictions=len(reused),
                maximum_replay_error=maximum,prior_prefix_checkpoints_with_identical_weights=prefix_count,
                prior_prefix_predictions_matched=prefix_rows,maximum_prefix_prediction_error=max_prefix_error,
                training_epoch_order_hashes_recomputed=order_count,paired_arm_sampling_orders_identical=True,
                prior_evidence_files_preserved=len(old),all_source_input_and_cache_hashes_match=True,
                all_retained_label_scales_and_maturity_verified=True,**scores,
                checked_utc=pd.Timestamp.now(tz='UTC').isoformat(),elapsed_seconds=time.time()-start)
    save(OUT/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

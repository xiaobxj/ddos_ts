"""Frozen full-history ablation and retained-anchor block sensitivity fits."""
from models6 import *
import argparse


def main(phase):
    initialize()
    start = time.time()
    config = cfg()
    settings = config['training']
    path_manifest = OUT / f'{phase}_manifest.json'
    assert not path_manifest.exists(), 'Run already started; preserve existing evidence'
    prep = json.loads((OUT / 'preparation_manifest.json').read_text(encoding='utf-8'))
    assert prep.get('finished_utc') and prep['protocol_sha256'] == sha(ROOT / 'protocol.json')
    assert prep['source_sha256'] == source_hashes() and prep['input_sha256'] == input_hashes()
    tests = json.loads((OUT / 'contract_verification.json').read_text(encoding='utf-8'))
    assert tests['status'] == 'PASS' and tests['protocol_sha256'] == sha(ROOT / 'protocol.json')
    assert tests['source_sha256'] == source_hashes()
    for name, digest in prep['cache_sha256'].items():
        assert sha(ROOT / name) == digest, name
    manifest = start_manifest(phase)
    manifest['preparation_manifest_sha256'] = sha(OUT / 'preparation_manifest.json')
    save(path_manifest, manifest)
    checkpoint_dir = OUT / f'{phase}_checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    obs = pd.read_csv(OUT / 'observation_table.csv')
    records = []
    curves = []
    checkpoints = []
    baselines = []
    fits = 0
    for fold in config['folds']:
        histories, te = fold_indices(obs, fold)
        history_names = ['full'] if phase == 'main' else config['training_block_sensitivity']['masks']
        seeds = settings['seeds'] if phase == 'main' else [config['training_block_sensitivity']['seed']]
        maximum = settings['max_epochs'] if phase == 'main' else config['training_block_sensitivity']['epochs']
        checkpoints_at = settings['epochs'] if phase == 'main' else [maximum]
        testing = batch_tensors(te)
        for history in history_names:
            tr = histories[history]
            values = batch_tensors(tr)
            labels, scales = standardized_targets(tr)
            ymean = scales['returns_mean']
            ysd = scales['returns_sd']
            full_count = len(histories['full'])
            for i in te:
                baselines.append(dict(history=history, date=obs.date.iloc[i], actual=float(obs.exec_return.iloc[i]),
                                      cutoff=fold['cutoff'], train_n=len(tr), full_train_n=full_count,
                                      training_mean=ymean, zero_return=0.))
            for arm in config['arms']:
                for seed in seeds:
                    network = make_model(arm['name'], seed)
                    optimizer = torch.optim.AdamW(network.parameters(), lr=settings['learning_rate'],
                                                  weight_decay=arm['weight_decay'])
                    rng = np.random.default_rng(seed)
                    for epoch in range(1, maximum + 1):
                        network.train()
                        order = epoch_order(len(tr), full_count, rng)
                        running = 0.
                        for lower in range(0, full_count, settings['batch_size']):
                            batch = torch.tensor(order[lower:lower + settings['batch_size']], device='cuda')
                            v = {k: t[batch] for k, t in values.items()}
                            optimizer.zero_grad(set_to_none=True)
                            output, aux = predict(network, v, True)
                            loss = ((output - labels['returns'][batch]) ** 2).mean()
                            loss = loss + settings['auxiliary_weight'] * ((aux - labels['auxiliary'][batch]) ** 2).mean()
                            assert torch.isfinite(loss)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(network.parameters(), settings['gradient_clip_norm'],
                                                           error_if_nonfinite=True)
                            optimizer.step()
                            running += float(loss.detach()) * len(batch)
                        curve = dict(arm=arm['name'], history=history, cutoff=fold['cutoff'], seed=seed,
                                     epoch=epoch, train_n=len(tr), presentations=full_count,
                                     optimizer_steps=int(np.ceil(full_count / settings['batch_size'])),
                                     online_joint_loss=running / full_count)
                        if epoch in checkpoints_at:
                            network.eval()
                            with torch.inference_mode():
                                out, _ = predict(network, testing, False)
                            estimates = out.cpu().numpy().astype(float) * ysd + ymean
                            for i, estimate in zip(te, estimates):
                                records.append(dict(arm=arm['name'], history=history, epoch=epoch, seed=seed,
                                                    cutoff=fold['cutoff'], train_n=len(tr), row_index=int(i),
                                                    date=obs.date.iloc[i], actual=float(obs.exec_return.iloc[i]),
                                                    predicted_return=float(estimate)))
                            train_mse, train_aux = evaluate_training(network, values, labels)
                            curve.update(final_model_train_return_mse=train_mse,
                                         final_model_train_auxiliary_mse=train_aux,
                                         validation_mse=float(np.mean((estimates - obs.exec_return.iloc[te].to_numpy()) ** 2)))
                            path = checkpoint_dir / f'{arm["name"]}_{history}_{fold["cutoff"]}_{seed}_epoch{epoch}.pt'
                            torch.save(dict(state_dict={k: t.detach().cpu() for k, t in network.state_dict().items()},
                                            arm=arm['name'], history=history, seed=seed, epoch=epoch, cutoff=fold['cutoff'],
                                            train_n=len(tr), full_train_n=full_count, scales=scales,
                                            protocol_sha256=sha(ROOT / 'protocol.json')), path)
                            checkpoints.append(dict(file=str(path.relative_to(ROOT)), sha256=sha(path),
                                                    arm=arm['name'], history=history, cutoff=fold['cutoff'], seed=seed, epoch=epoch))
                        curves.append(curve)
                    fits += 1
                    pd.DataFrame(records).to_csv(OUT / f'{phase}_seed_predictions.csv', index=False)
                    pd.DataFrame(curves).to_csv(OUT / f'{phase}_training_curves.csv', index=False)
                    save(OUT / f'{phase}_checkpoint_manifest.json', checkpoints)
                    print(f'{phase}: {fits}/45 fits; {fold["cutoff"]} {history} {arm["name"]} seed {seed}; {time.time()-start:.1f}s', flush=True)
                    del network, optimizer
                    torch.cuda.empty_cache()
            del values, labels
            torch.cuda.empty_cache()
        del testing
    pd.DataFrame(baselines).to_csv(OUT / f'{phase}_baselines.csv', index=False)
    assert old_evidence() == manifest['old_evidence']
    assert source_hashes() == manifest['source_sha256'] and input_hashes() == manifest['input_sha256']
    manifest.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(), elapsed_seconds=time.time()-start,
                    fits=fits, checkpoints=len(checkpoints), predictions=len(records), old_preserved=True)
    save(path_manifest, manifest)
    print(f'Finished {phase}: {fits} fits, {len(checkpoints)} checkpoints.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['main', 'blocks'], required=True)
    main(parser.parse_args().phase)

"""Training-only interval and fitted-loss attribution; no new validation selection."""
from common8 import *


def concurrency(left, right_exclusive):
    delta = np.zeros(int(max(right_exclusive))+1, dtype=np.int64)
    np.add.at(delta, left, 1); np.add.at(delta, right_exclusive, -1)
    counts = np.cumsum(delta)
    active = counts[counts>0]
    return dict(active_bars=len(active), mean_concurrency=float(active.mean()), maximum_concurrency=int(active.max()),
                adjacent_intervals_share_bars=float(np.mean(left[1:]<right_exclusive[:-1])))


def main():
    legacy.initialize(); check_frozen()
    destination = OUT / 'audit_manifest.json'; assert not destination.exists()
    run = manifest('training_only_audit'); save(destination, run)
    obs = pd.read_csv(OUT / 'observation_table.csv')
    refs = json.loads((OUT / 'reused_checkpoint_manifest.json').read_text(encoding='utf-8'))
    reused = pd.read_csv(OUT / 'reused_full_predictions.csv')
    per_seed = []; summary = []; groups = []; overlap = []; max_replay = 0.
    for fold in cfg()['folds']:
        tr, weights, te, _ = load_policy(fold['cutoff'], 'full')
        values = legacy.batch_tensors(tr); testing = legacy.batch_tensors(te)
        labels, scales = targets(tr, weights)
        y = obs.exec_return.iloc[tr].to_numpy()
        table = obs.iloc[tr][['date','anchor','entry','exit']].copy().reset_index(drop=True)
        table['row_index'] = tr; table['actual'] = y
        table['mean_loss'] = (y-scales['returns_mean'])**2
        fitted_losses = []; auxiliary_losses = []; fitted_predictions = []
        for r in [r for r in refs if r['cutoff']==fold['cutoff']]:
            saved = torch.load(PROJECT / r['project_file'], map_location='cpu', weights_only=True)
            assert saved['scales'] == scales
            model = legacy.make_model('combined', r['seed']).eval(); model.load_state_dict(saved['state_dict'])
            p = []; a = []
            with torch.inference_mode():
                for lo in range(0, len(tr), 128):
                    raw, aux = legacy.predict(model, {k:t[lo:lo+128] for k,t in values.items()}, True)
                    p.extend(raw.cpu().numpy().astype(float)*scales['returns_sd']+scales['returns_mean'])
                    a.extend(((aux-labels['auxiliary'][lo:lo+128])**2).mean(dim=1).cpu().numpy().astype(float))
                out, _ = legacy.predict(model, testing, False)
            pv = out.cpu().numpy().astype(float)*scales['returns_sd']+scales['returns_mean']
            expected = reused[(reused.cutoff==fold['cutoff']) & (reused.seed==r['seed'])].sort_values('row_index')
            np.testing.assert_allclose(pv, expected.predicted_return, rtol=0, atol=1e-10)
            max_replay = max(max_replay, float(np.max(np.abs(pv-expected.predicted_return))))
            loss = (np.asarray(p)-y)**2
            fitted_losses.append(loss); auxiliary_losses.append(a); fitted_predictions.append(p)
            for index, pred, error, aux in zip(tr, p, loss, a):
                per_seed.append(dict(cutoff=fold['cutoff'], seed=r['seed'], row_index=int(index), date=obs.date.iloc[index],
                    predicted_return=float(pred), actual=float(obs.exec_return.iloc[index]), fitted_return_loss=float(error),
                    fitted_auxiliary_standardized_loss=float(aux)))
            del model
        table['fitted_loss'] = np.mean(fitted_losses, axis=0)
        table['fitted_auxiliary_standardized_loss'] = np.mean(auxiliary_losses, axis=0)
        table['fitted_ensemble_loss'] = (np.mean(fitted_predictions, axis=0)-y)**2
        table['year'] = table.date.str[:4]
        table['quarter'] = pd.to_datetime(table.date).dt.to_period('Q').astype(str)
        table['index_block'] = np.concatenate([np.full(len(v), str(i)) for i,v in enumerate(np.array_split(tr,5))])
        for grouping in ['year', 'quarter', 'index_block']:
            for key, g in table.groupby(grouping):
                groups.append(dict(cutoff=fold['cutoff'], grouping=grouping, period=str(key), rows=len(g),
                    sample_share=len(g)/len(table), mean_loss_share=float(g.mean_loss.sum()/table.mean_loss.sum()),
                    fitted_loss_share=float(g.fitted_loss.sum()/table.fitted_loss.sum()),
                    fitted_auxiliary_loss_share=float(g.fitted_auxiliary_standardized_loss.sum()/table.fitted_auxiliary_standardized_loss.sum()),
                    mean_return=float(g.actual.mean()), return_sd=float(g.actual.std(ddof=0)),
                    fitted_mse=float(g.fitted_loss.mean()), fitted_ensemble_mse=float(g.fitted_ensemble_loss.mean())))
        count = int(np.ceil(len(tr)*.01))
        summary.append(dict(cutoff=fold['cutoff'], train_n=len(tr), adjacent_return_correlation=float(np.corrcoef(y[:-1],y[1:])[0,1]),
            top_one_percent_rows=count, top_one_percent_mean_loss_share=float(table.mean_loss.nlargest(count).sum()/table.mean_loss.sum()),
            top_one_percent_fitted_loss_share=float(table.fitted_loss.nlargest(count).sum()/table.fitted_loss.sum()),
            training_mean_mse=float(table.mean_loss.mean()), fitted_mean_seed_mse=float(table.fitted_loss.mean()),
            fitted_ensemble_mse=float(table.fitted_ensemble_loss.mean())))
        left = table.entry.to_numpy(dtype=int)
        for kind, right in [('return_increments', table.exit.to_numpy(dtype=int)),
                            ('joint_future_bars', np.maximum(table.exit, table.anchor+5).to_numpy(dtype=int)+1)]:
            overlap.append(dict(cutoff=fold['cutoff'], support=kind, **concurrency(left, right)))
        table.insert(0, 'cutoff', fold['cutoff'])
        table.to_csv(OUT / f'training_loss_rows_{fold["cutoff"]}.csv', index=False)
        del values, testing, labels
        torch.cuda.empty_cache()
    pd.DataFrame(per_seed).to_csv(OUT / 'frozen_model_training_predictions.csv', index=False)
    pd.DataFrame(summary).to_csv(OUT / 'training_audit_summary.csv', index=False)
    pd.DataFrame(groups).to_csv(OUT / 'training_loss_attribution.csv', index=False)
    pd.DataFrame(overlap).to_csv(OUT / 'label_overlap.csv', index=False)
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(), reused_checkpoints=9, training_predictions=len(per_seed),
               maximum_validation_replay_error=max_replay, used_new_validation_results=False)
    save(destination, run)
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(overlap).to_string(index=False))


if __name__ == '__main__':
    main()

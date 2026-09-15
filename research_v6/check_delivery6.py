"""Read-only final artifact, metric and provenance integrity check."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def main():
    manifest = read('delivery_manifest.json')
    for name, digest in manifest['files'].items():
        with (ROOT / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
    report = (ROOT / manifest['report']).read_text(encoding='utf-8')
    assert report.startswith('# 第六轮方法测试：')
    assert '\ufffd' not in report and 'NotImplementedError' not in report
    for target in re.findall(r'\]\(([^)]+)\)', report):
        if not target.startswith(('https://', 'http://')):
            assert (OUT / target).resolve().exists(), target
    audit = read('verification.json')
    assert audit['status'] == 'PASS' and audit['total_checkpoints_replayed'] == 225
    ensemble = pd.read_csv(OUT / 'main_ensemble_predictions.csv')
    metrics = pd.read_csv(OUT / 'main_metrics.csv')
    for r in metrics.itertuples():
        g = ensemble[(ensemble.arm == r.arm) & (ensemble.epoch == r.epoch)]
        assert len(g) == 141
        assert abs(np.sqrt(np.mean((g.predicted_return-g.actual)**2))-r.rmse) < 1e-12
        assert abs(((g.predicted_return>0)==(g.actual>0)).mean()-r.accuracy) < 1e-12
    block = pd.read_csv(OUT / 'block_comparison_predictions.csv')
    block_metrics = pd.read_csv(OUT / 'block_metrics.csv')
    for r in block_metrics.itertuples():
        g = block[(block.arm == r.arm) & (block.history == r.history)]
        assert len(g) == 141
        assert abs(np.sqrt(np.mean((g.predicted_return-g.actual)**2))-r.rmse) < 1e-12
        ref = block[(block.arm == 'baseline') & (block.history == r.history)]
        delta = float(np.mean((g.predicted_return-g.actual)**2) - np.mean((ref.predicted_return-ref.actual)**2))
        assert abs(delta-r.mse_difference_vs_matched_baseline) < 1e-12
    sensitivity = pd.read_csv(OUT / 'block_prediction_sensitivity.csv')
    for r in sensitivity.itertuples():
        g = block[(block.arm == r.arm) & (block.history == r.history)].sort_values('date')
        ref = block[(block.arm == r.arm) & (block.history == 'full')].sort_values('date')
        p = g.predicted_return.to_numpy()
        p0 = ref.predicted_return.to_numpy()
        assert abs(float(np.sqrt(np.mean((p-p0)**2)))-r.prediction_rms_shift) < 1e-12
        assert int(((p>0)!=(p0>0)).sum()) == r.changed_direction_weeks
        mu = g.training_mean.to_numpy()
        mu0 = ref.training_mean.to_numpy()
        assert abs(float(np.sqrt(np.mean((mu-mu0)**2)))-r.training_mean_rms_shift) < 1e-12
        assert abs(float(np.sqrt(np.mean(((p-mu)-(p0-mu0))**2)))-r.prediction_rms_shift_after_removing_training_means) < 1e-12
    observations = pd.read_csv(OUT / 'observation_table.csv')
    for r in pd.read_csv(OUT / 'training_history_label_statistics.csv').itertuples():
        with np.load(ROOT / 'cache' / f'masks_{r.cutoff}.npz') as masks:
            y = observations.exec_return.iloc[masks[r.history]].to_numpy()
            assert len(y) == r.training_n
            assert abs(float(y.mean())-r.label_mean) < 1e-12
            assert abs(float(y.std())-r.label_std) < 1e-12
    selection = read('selection.json')['selected']
    protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    order = {a['name']: i for i, a in enumerate(protocol['arms'])}
    winner = min(metrics.to_dict('records'), key=lambda r: (r['rmse'], r['epoch'], order[r['arm']]))
    assert (winner['arm'], winner['epoch']) == (selection['arm'], selection['epoch'])
    print(json.dumps(dict(status='PASS', hashed_files=len(manifest['files']), report_characters=len(report),
                         validation_candidates=len(metrics), training_history_comparisons=len(block_metrics),
                         fits=audit['total_fits'], replayed_checkpoints=audit['total_checkpoints_replayed'],
                         selected_arm=winner['arm'], selected_epoch=int(winner['epoch']),
                         selected_accuracy=winner['accuracy'], selected_rmse=winner['rmse']), indent=2))


if __name__ == '__main__':
    main()

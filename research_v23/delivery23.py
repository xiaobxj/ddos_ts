"""Check and freeze the reviewed round23 research delivery without model execution."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
PROJECT = ROOT.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def csv(name):
    return pd.read_csv(OUT / f'{name}.csv', float_precision='round_trip')


def check_hashes(mapping, root):
    for name, digest in mapping.items():
        assert sha(root / name) == digest, name


def main(freeze):
    prep = read(OUT / 'preparation_manifest.json')
    protocol = read(ROOT / 'protocol.json')
    verification = read(OUT / 'verification.json')
    assert verification['status'] == 'PASS'
    expected = dict(previous_files_preserved=3292, new_primary_fits=30,
        independent_solutions=30, independent_qr_projections=24,
        state_split_checks=12, future_prefix_checks=18, new_hmm_fits=0,
        new_neural_forward_passes=0, neural_training_steps=0,
        inherited_neural_replay_states=18, training_feature_rows=47325,
        new_probability_forecasts=1305, reused_model_records=11223,
        model_records=12528, ensemble_records=7569, metric_groups=1868,
        state_metric_rows=1508, reliability_bins=280, primary_contrasts=26,
        independent_holdout_dates=0)
    for key, value in expected.items():
        assert verification[key] == value, key
    assert prep['protocol_sha256'] == verification['protocol_sha256'] == sha(ROOT / 'protocol.json')
    assert prep['source_sha256'] == verification['source_sha256']
    for key in ['old_evidence', 'source_sha256', 'input_sha256']:
        check_hashes(prep[key], PROJECT)
    assert len(prep['old_evidence']) == 3292
    for record in [prep, verification]:
        check_hashes(record['artifacts'], ROOT)
    contract = read(OUT / 'contract_verification.json')
    assert contract['status'] == 'PASS' and len(contract['tests']) == 6
    for key in ['protocol_sha256', 'source_sha256']:
        assert contract[key] == prep[key]
    check_hashes(contract['artifacts'], ROOT)
    last = contract['completed_utc']
    assert prep['finished_utc'] <= last
    for phase in ['training', 'scoring', 'evaluation']:
        record = read(OUT / f'{phase}_manifest.json')
        assert last <= record['started_utc'] < record['finished_utc']
        last = record['finished_utc']
        for key in ['protocol_sha256', 'source_sha256', 'input_sha256']:
            assert record[key] == prep[key]
        check_hashes(record['artifacts'], ROOT)
    assert last <= verification['completed_utc']
    fit = read(OUT / 'training_manifest.json')
    assert fit['all_converged'] and not fit['validation_scoring_during_fit']
    assert fit['new_projection_fits'] == 24
    heads = read(OUT / 'heads.json')
    assert len(heads) == 30
    assert sum(h['iterations'] for h in heads) == fit['new_newton_iterations'] == 109
    assert sum(len(h['coefficients']) for h in heads) == 762
    for head in heads:
        variant = next(x for x in protocol['variants'] if x['method'] == head['method'])
        assert head['dimensions'] == variant['dimensions'] and head['gate'] == variant['gate']
        assert sha(PROJECT / head['cache_file']) == head['cache_sha256']
        assert head['gradient_inf'] <= 1e-9 and head['hessian_min_eigenvalue'] > 0
        if head['source_job']:
            assert head['objective'] <= head['parent_objective'] + 1e-12
    alternate = csv('independent_solver_verification')
    settings = protocol['independent_solver']
    assert len(alternate) == 30
    assert alternate.objective_absolute_gap.max() <= settings['objective_absolute_tolerance']
    assert alternate.maximum_training_probability_gap.max() <= settings['training_probability_absolute_tolerance']
    assert alternate.alternate_gradient_inf.max() <= settings['gradient_infinity_tolerance']
    checks = csv('state_verification')
    assert len(checks) == 12 and checks.maximum_scalar_gap.max() <= 2e-12
    for split, count in [('training', 9465), ('validation', 261)]:
        assert len(csv(f'{split}_signal_states')) == count
        assert len(csv(f'{split}_stability')) == 6
    forecasts = csv('forecast_verification')
    assert len(forecasts) == 30 and forecasts.training_rows.sum() == 47325
    assert forecasts.heldout_rows.sum() == 1305
    parts = csv('interaction_components')
    assert len(parts) == 1044
    np.testing.assert_allclose(parts.refitted_parent_logit,
        parts.refitted_base_logit + parts.refitted_original_interaction_logit, rtol=0, atol=1e-12)
    np.testing.assert_allclose(parts.logit,
        parts.refitted_parent_logit + parts.interaction_logit, rtol=0, atol=1e-12)
    coefficients = csv('coefficients')
    assert len(coefficients) == 762
    assert coefficients.block.eq('original_interaction').sum() == 24
    assert coefficients.block.eq('interaction').sum() == 24
    models = csv('model_predictions')
    ensemble = csv('ensemble_predictions')
    metrics = csv('ensemble_metrics')
    years = csv('yearly_metrics')
    pairs = read(OUT / 'primary_comparisons.json')
    assessments = read(OUT / 'assessments.json')
    facts = read(OUT / 'report_facts.json')
    assert len(models) == 12528 and len(ensemble) == 7569 and len(metrics) == 87
    assert ensemble.method.nunique() == 29 and ensemble.groupby('method').size().eq(261).all()
    assert len(years) == 174 and len(csv('state_metrics')) == 1508
    assert len(csv('reliability_bins')) == 280
    for row in metrics.itertuples():
        group = ensemble[ensemble.method.eq(row.method)]
        if row.window == 'early_2015_2017':
            group = group[group.year.le(2017)]
        if row.window == 'late_2018_2020':
            group = group[group.year.ge(2018)]
        assert len(group) == row.n and group.date.is_unique
        assert group.direction_up.eq(group.actual_up).sum() == row.correct_directions
        if row.method == 'native_mse':
            assert group.probability.isna().all()
        else:
            assert abs(np.square(group.probability - group.actual_up).mean() - row.brier) < 1e-14
            assert row.clipped_probabilities == 0
    assert len(pairs) == 26
    declared_pairs = [dict(window=w['name'], **q) for w in protocol['windows']
        for q in protocol['primary_comparisons_per_window']]
    for result, declared in zip(pairs, declared_pairs):
        for key, value in declared.items():
            assert result[key] == value
    improvements = [r for r in pairs if r['difference'] < 0 and r['holm_adjusted_p'] < .05]
    worsenings = [r for r in pairs if r['difference'] > 0 and r['holm_adjusted_p'] < .05]
    assert len(improvements) == facts['significant_improvements'] == 0
    assert len(worsenings) == facts['significant_deteriorations'] == 0
    assert len(assessments) == 4
    for item in assessments:
        current = metrics[metrics.window.eq(item['window'])].set_index('method')
        candidate = item['method']
        row = current.loc[candidate]
        window = next(w for w in protocol['windows'] if w['name'] == item['window'])
        yearly = years[years.year.between(int(window['start'][:4]), int(window['end'][:4]))]
        accuracy = yearly.pivot(index='year', columns='method', values='accuracy')
        brier = yearly.pivot(index='year', columns='method', values='brier')
        flags = {f'accuracy_beats_{ref}': bool(row.accuracy > current.loc[ref, 'accuracy'])
            for ref in [item['parent'], item['additive_anchor'], 'native_mse', 'training_frequency']}
        flags.update({f'brier_beats_{ref}': bool(row.brier < current.loc[ref, 'brier'])
            for ref in [item['parent'], item['additive_anchor'], 'training_frequency', item['control']]})
        flags.update(log_loss_beats_frequency=bool(row.log_loss < current.loc['training_frequency', 'log_loss']),
            two_years_beat_native_accuracy=bool((accuracy[candidate] > accuracy.native_mse).sum() >= 2),
            two_years_beat_frequency_brier=bool((brier[candidate] < brier.training_frequency).sum() >= 2))
        assert flags == item['flags'] and len(flags) == 11
        assert item['window_descriptive_pass'] == all(flags.values())
        assert not item['cross_period_descriptive_pass'] and not item['strategy_promotion']
        assert not item['independent_confirmation'] and item['interaction_hypothesis_retained']
    for name, signs in [('example_clustered', np.r_[np.ones(40), -np.ones(20)]),
                         ('example_interleaved', np.tile([1, -1, 1, -1, 1, 1], 10))]:
        observed = float(np.mean(signs[:-1] * signs[1:]))
        expectation = float((signs.sum() ** 2 - np.square(signs).sum()) / (60 * 59))
        assert facts['same_positive_count'] == int((signs > 0).sum()) == 40
        assert facts['same_negative_count'] == int((signs < 0).sum()) == 20
        for key, value in dict(observed=observed, permutation_expectation=expectation,
                               excess_order=observed-expectation).items():
            assert abs(facts[name][key] - value) < 1e-14
    usage = csv('historical_date_usage')
    assert len(usage) == 261 and usage.previously_evaluated.all()
    assert set(usage.date) == set(ensemble.date)
    from report23 import LABELS, MAIN, WINDOWS, FAMILIES, metric_row, pair_row, line, pct, num
    report_path = OUT / '第二十三轮测试报告.md'
    report = report_path.read_text(encoding='utf-8')
    note = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert report.startswith('# 第二十三轮方法测试：') and '\ufffd' not in report
    assert '不能作为独立确认' in report and verification['protocol_sha256'] in report
    assert '保留全历史交互' in report and '58.16%' in note
    for text, folder in [(report, OUT), (note, ROOT)]:
        for link in re.findall(r'\]\(([^)]+)\)', text):
            if not link.startswith(('http://', 'https://')):
                assert (folder / link).resolve().is_file(), link
    for row in metrics[metrics.window.ne('pooled_2015_2020') & metrics.method.isin(MAIN)].itertuples():
        assert line(metric_row(row)) in report
    for result in pairs:
        assert line(pair_row(result)) in report
    for year, group in years.groupby('year'):
        current = group.set_index('method')
        for family in FAMILIES:
            assert line([year, int(group.iloc[0]['n'])] + [int(current.loc[m, 'correct_directions'])
                for m in family + ['order_only', 'native_mse', 'training_frequency']]) in report
    for row in csv('validation_stability').itertuples():
        assert line([int(row.cutoff[:4])+1, row.n, row.missing, pct(row.high_state_share),
            f'{row.lag1_correlation:.4f}', f'{row.switches}/{row.adjacent_pairs}',
            f'{row.median_observed_run:g}', row.max_observed_run,
            f'{row.censored_runs}/{row.runs}']) in report
    correlations = csv('state_correlations')
    for row in correlations[correlations.split.eq('validation')].itertuples():
        assert line([int(row.cutoff[:4])+1, f'{row.correlation_volatility20:.6f}',
            f'{row.correlation_abs_trend60:.6f}', f'{row.correlation_efficiency60:.6f}',
            f'{row.correlation_relativevol:.6f}']) in report
    for method, group in csv('gate_novelty').groupby('method'):
        assert line([LABELS[method], num(group.gate_linear_r2.min()),
            num(group.gate_linear_r2.mean()), num(group.gate_linear_r2.max())]) in report
    for (method, cutoff), group in csv('training_metrics').groupby(['method', 'cutoff']):
        assert line([LABELS[method], cutoff, pct(group.accuracy.mean()),
            num(group.brier.mean()), num(group.objective.mean())]) in report
    for (method, cutoff), group in coefficients[coefficients.method.ne('order_only')].groupby(['method', 'cutoff']):
        assert line([LABELS[method], int(cutoff[:4])+1,
            num(group[group.block.eq('original_interaction')].coefficient.mean()),
            num(group[group.block.eq('interaction')].coefficient.mean())]) in report
    state_metrics = csv('state_metrics')
    displayed_states = state_metrics[state_metrics.partition.eq('order_state') & state_metrics.method.isin(
        ['learned_vol_interaction', 'learned_order_extension', 'raw_order_extension', 'order_only', 'training_frequency'])]
    assert len(displayed_states) == 30
    for row in displayed_states.itertuples():
        assert line([row.state, LABELS[row.method], row.n, pct(row.accuracy), num(row.brier)]) in report
    for item in assessments:
        assert line([WINDOWS[item['window']], LABELS[item['method']],
            f"{sum(item['flags'].values())}/11", '否', '是']) in report
    visual = read(OUT / 'visual_review.json')
    assert visual['status'] == 'PASS' and len(visual['files']) == 2
    assert visual['completed_utc'] >= verification['completed_utc']
    check_hashes(visual['files'], ROOT)
    for name in ['order_extension_comparison', 'order_state_diagnostics']:
        assert (OUT / f'{name}.png').stat().st_size > 50000
        assert (OUT / f'{name}.svg').stat().st_size > 10000
    target = OUT / 'delivery_manifest.json'
    if freeze:
        assert not target.exists(), 'Preserve frozen delivery'
        files = {str(f.relative_to(ROOT)): sha(f) for f in sorted(ROOT.rglob('*'))
            if f.is_file() and '__pycache__' not in f.parts and f.name != 'delivery_manifest.json'}
        result = dict(experiment=23, completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),
            report=str(report_path.relative_to(ROOT)), protocol_sha256=verification['protocol_sha256'],
            previous_files_preserved=3292, new_primary_fits=30, new_newton_updates=109,
            new_projection_fits=24, independent_solutions=30, new_neural_forward_passes=0,
            neural_training_steps=0, new_hmm_fits=0, independent_holdout_dates=0,
            new_probability_forecasts=1305, model_records=12528, ensemble_records=7569,
            figures_visually_reviewed=2, cross_period_descriptive_pass={a['method']: False for a in assessments},
            interaction_hypothesis_retained=True, new_state_interaction_adopted=False,
            order_extension_partial_improvement_retained=True,
            path_efficiency_probability_improvement_retained=True,
            significant_improvements=0, significant_deteriorations=0, files=files)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if target.exists():
        check_hashes(read(target)['files'], ROOT)
    print(json.dumps(dict(status='PASS', frozen=target.exists(),
        hashed_files=len(read(target)['files']) if target.exists() else None,
        old_files_preserved=3292, report_characters=len(report), new_fits=30,
        interaction_hypothesis_retained=True, new_state_interaction_adopted=False), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    main(parser.parse_args().freeze)

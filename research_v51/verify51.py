"""Boundary tests and a real-model rehearsal on explicitly synthetic future bars."""
from contextlib import ExitStack
from datetime import datetime, timedelta
import copy
import tempfile
from unittest.mock import patch

from common51 import *
from cycle51 import Backend, cycle, cycle_lock, describe, due_cutoffs
import common49
import data49
import forecast49
import run50
import build50


def stamp(value):
    return datetime.fromisoformat(value + '+08:00')


class FakeBackend:
    def __init__(self, when='2026-09-18T18:30:00'):
        self.at = stamp(when)
        self.rows = []
        self.calls = []
        self.fetch_end = '2026-09-18'
        self.fetch_error = False
        self.refresh_error = False
        self.fetch_delay = timedelta()
        self.refit_delay = timedelta()
        self.validation_error = False
        self.evaluation = dict(status='DESCRIPTIVE_ONLY')

    def now(self):
        return self.at

    def events(self):
        return self.rows

    def append(self, kind, key, payload=None):
        e = dict(sequence=len(self.rows)+1, kind=kind, key=key,
                 recorded_utc=iso(self.at), payload=payload or {})
        self.rows.append(e)
        return e

    def validate(self):
        if self.validation_error:
            raise ValueError('fixture provenance failure')

    def fetch(self):
        self.calls.append('fetch')
        if self.fetch_error:
            raise RuntimeError('fixture fetch failure')
        self.at += self.fetch_delay
        return self.append('snapshot', str(len(self.rows)), dict(end=self.fetch_end))

    def refresh(self, cutoff):
        self.calls.append('refresh:' + cutoff)
        if self.refresh_error:
            raise RuntimeError('fixture refresh failure')
        self.at += self.refit_delay
        return self.append('parameter_package', cutoff)

    def record(self):
        self.calls.append('record')
        return self.append('prediction', self.at.astimezone(TZ).strftime('%Y-%m-%d'))

    def settle(self):
        self.calls.append('settle')
        return []

    def evaluate(self):
        self.calls.append('evaluate')
        return self.evaluation


def boundary_tests(folder):
    cases = []
    def run(b, name):
        return cycle(b, folder / name)
    def passed(name):
        cases.append(name)

    b = FakeBackend('2026-09-18T17:59:59')
    assert run(b, 'preclose')['status'] == 'PRE_CLOSE_WAIT' and b.calls == [] and b.rows == []
    passed('preclose_has_no_fetch_or_scientific_writes')
    b = FakeBackend(); b.fetch_error = True
    r = run(b, 'fetch_failure')
    assert r['status'] == 'FAILED' and b.calls == ['fetch'] and not b.rows
    passed('fetch_failure_stops_dependent_steps_and_keeps_receipt')
    b = FakeBackend(); b.fetch_end = '2026-09-17'
    r = run(b, 'stale')
    assert r['status'] == 'DATA_PENDING' and 'record' not in b.calls
    passed('missing_friday_never_uses_thursday')
    b = FakeBackend('2026-09-18T22:59:59'); b.fetch_delay = timedelta(seconds=2)
    r = run(b, 'late_fetch')
    assert 'record' not in b.calls and any(s.get('status') == 'OUTSIDE_RECORDING_WINDOW' for s in r['steps'])
    passed('fetch_crossing_deadline_cannot_record')
    b = FakeBackend(); r = run(b, 'once'); original = copy.deepcopy(b.rows)
    b.at += timedelta(minutes=1); r2 = run(b, 'twice')
    assert b.rows == original and b.calls.count('fetch') == b.calls.count('record') == 1
    assert r2['steps'][0]['status'] == 'ALREADY_RECORDED'
    passed('repeated_cycle_reuses_snapshot_and_prediction')
    b = FakeBackend('2026-09-19T18:30:00')
    run(b, 'weekend'); assert b.calls == ['evaluate']
    passed('weekend_has_no_same_day_fetch_or_prediction')
    b = FakeBackend(); b.validation_error = True
    assert run(b, 'provenance')['status'] == 'FAILED' and not b.calls
    passed('provenance_failure_prevents_network_and_writes')
    b = FakeBackend()
    with cycle_lock(folder / 'concurrent'):
        try:
            run(b, 'concurrent')
        except FileExistsError:
            pass
        else:
            raise AssertionError('Concurrent cycle was accepted')
    assert not b.calls and not (folder / 'concurrent/cycle.lock').exists()
    passed('whole_cycle_exclusive_lock_and_release')
    b = FakeBackend('2026-12-31T18:30:00'); b.fetch_end = '2026-12-31'
    run(b, 'quarter_order')
    assert [x for x in b.calls if x.startswith('refresh:')] == ['refresh:2026-09-30', 'refresh:2026-12-31']
    passed('missed_quarters_dispatched_in_order_including_annual_cutoff_mock_only')
    b = FakeBackend('2026-09-30T18:30:00'); b.fetch_end = '2026-09-29'
    assert run(b, 'cutoff_pending')['status'] == 'DATA_PENDING'
    assert not any(x.startswith('refresh:') for x in b.calls)
    passed('cutoff_day_requires_covering_data')
    b = FakeBackend('2026-09-30T18:30:00'); b.fetch_end = '2026-09-30'
    run(b, 'cutoff_ready')
    assert 'refresh:2026-09-30' in b.calls and 'record' not in b.calls
    passed('quarter_end_after18_refreshes_next_quarter')
    assert due_cutoffs([], stamp('2026-09-30T17:59:59')) == []
    assert due_cutoffs([], stamp('2026-12-31T17:59:59')) == ['2026-09-30']
    passed('no_early_quarter_or_annual_dispatch')
    b = FakeBackend('2026-10-02T18:30:00'); b.fetch_end = '2026-10-02'; b.refresh_error = True
    r = run(b, 'refresh_failure')
    assert r['status'] == 'FAILED' and len(b.rows) == 1 and b.rows[0]['kind'] == 'snapshot'
    assert 'record' not in b.calls and 'settle' not in b.calls
    passed('refresh_failure_keeps_snapshot_but_stops_prediction')
    b = FakeBackend('2026-10-02T22:59:59'); b.fetch_end = '2026-10-02'; b.refit_delay = timedelta(seconds=2)
    run(b, 'late_refit'); assert 'record' not in b.calls
    passed('refresh_crossing_deadline_cannot_record')
    # Exercise the production evaluation wrapper, separately from mock dispatch.
    class EvaluationJournal:
        def __init__(self, saved=None):
            self.rows = [] if saved is None else [dict(kind='primary_evaluation', payload=saved)]
        def events(self):
            return self.rows
        def append(self, kind, key, payload):
            self.rows.append(dict(kind=kind, key=key, payload=payload))
    sentinel = dict(status='PRIMARY_READY', fixture='test_only')
    j = EvaluationJournal(); backend = Backend(j)
    with patch('cycle51.summarize', return_value=sentinel) as summarize_mock:
        assert backend.evaluate() == backend.evaluate() == sentinel
        assert summarize_mock.call_count == 1 and len(j.rows) == 1
    passed('primary_evaluation_saved_once_and_reused')
    return cases


def real_model_rehearsal(folder):
    at = [stamp('2026-09-18T17:59:59')]
    j = Journal(folder / 'SYNTHETIC_TEST_ONLY_journal')
    original_runtime = {relative(p): sha(p) for p in previous.base.RUNTIME.rglob('*') if p.is_file()}
    actual = Journal(); original = data49.load_snapshot(actual, [e for e in actual.events() if e['kind'] == 'snapshot'][-1])
    pd = previous.pd
    dates = pd.bdate_range('2026-09-15', '2026-09-28').strftime('%Y-%m-%d').tolist()
    last = float(original.close.iloc[-1])
    added = []
    for i, date in enumerate(dates):
        price = last * (1 + .001 * (i+1))
        added.append(dict(date=date, open=price, high=price*1.004, low=price*.996,
                          close=price*1.001, volume=float(original.volume.iloc[-1])))
    synthetic = pd.concat([original[data49.COLS], pd.DataFrame(added)], ignore_index=True)
    labels = ['common49.utc', 'data49.utc', 'forecast49.utc', 'run50.utc', 'build50.utc', 'cycle51.utc']
    with ExitStack() as stack:
        for label in labels:
            stack.enter_context(patch(label, side_effect=lambda: at[0]))
        # Explicit test marker is created before hashing the first event.
        save(j.root / 'genesis.json', dict(scope='prospective', synthetic_test_only=True,
             warning='ISOLATED ENGINEERING FIXTURE. SIMULATED FUTURE DATES AND PRICES. NOT RESEARCH PERFORMANCE.',
             created_utc=iso(at[0]), protocol_sha256=sha(PROJECT/'research_v49/protocol.json'),
             source_freeze_sha256=sha(PROJECT/'research_v49/results/freeze.json')))
        (j.root/'events').mkdir(); (j.root/'blobs').mkdir()
        build50.activate(j, PROJECT/'research_v49/results/bootstrap_package.json',
                         PROJECT/'research_v50/results/bootstrap_verification.json')
        backend = Backend(j)
        def synthetic_fetch(target):
            assert target.root == j.root and target.root != previous.base.RUNTIME
            end = at[0].astimezone(TZ).strftime('%Y-%m-%d')
            return data49.snapshot(target, synthetic[synthetic.date.le(end)].copy(),
                                   dict(provider='SYNTHETIC_TEST_ONLY', fixture_clock=True))
        stack.enter_context(patch('data49.fetch', side_effect=synthetic_fetch))
        # Any accidental real future fitting is a test failure, not additional research.
        refit = stack.enter_context(patch('build50.refresh', side_effect=AssertionError('Unexpected actual future fitting')))
        operation = folder / 'SYNTHETIC_TEST_ONLY_runs'
        stages = []
        def run_stage(name):
            result = cycle(backend, operation)
            assert result['status'] != 'FAILED', result
            stages.append(dict(stage=name, status=result['status'], counts=result['after'], steps=result['steps']))
            return result
        run_stage('before_friday_close')
        assert len(j.events()) == 1
        at[0] = stamp('2026-09-18T18:30:00')
        first = run_stage('friday_timely_prediction')
        event = next(e for e in j.events() if e['kind']=='prediction')
        forecast = event['payload']['forecast']
        assert len(forecast['seed_predictions']) == 60 and len(forecast['ensemble_predictions']) == 20
        assert first['evaluation']['comparisons'] == [] and first['evaluation']['complete_mature_signals'] == 0
        committed = copy.deepcopy(j.events())
        at[0] += timedelta(minutes=1); run_stage('same_friday_rerun')
        assert j.events() == committed
        at[0] = stamp('2026-09-25T23:05:00'); pending = run_stage('next_friday_after_deadline')
        assert pending['after']['recorded_predictions'] == 1 and pending['after']['labels'] == 0
        assert pending['evaluation']['comparisons'] == []
        at[0] = stamp('2026-09-28T18:30:00'); mature = run_stage('following_monday_mature_label')
        label = next(e['payload'] for e in j.events() if e['kind']=='label')
        assert label['status']=='MATURE' and label['entry_date']=='2026-09-21' and label['exit_date']==label['joint_completed']=='2026-09-28'
        expected_return = float(synthetic.set_index('date').loc['2026-09-28','open']/synthetic.set_index('date').loc['2026-09-21','open']-1)
        assert label['exec_return'] == expected_return and label['actual_up'] == 1
        evaluation = mature['evaluation']
        assert evaluation['status']=='DESCRIPTIVE_ONLY' and evaluation['comparisons']==[]
        assert evaluation['complete_mature_signals']==1 and len(evaluation['metrics'])==20
        assert all(m['n']==1 for m in evaluation['metrics'])
        assert next(c for c in evaluation['coverage'] if c['date']=='2026-09-25')['status']=='NO_TIMELY_PREDICTION'
        committed = copy.deepcopy(j.events())
        at[0] += timedelta(minutes=1); run_stage('mature_label_rerun')
        assert j.events()==committed and refit.call_count==0
        # Original writer independently protects a deadline crossed during inference.
        at[0] = stamp('2026-09-28T23:00:00')
        try:
            j.append('prediction', '2026-09-25', {})
        except ValueError:
            pass
        else:
            raise AssertionError('Historical backfill accepted')
        assert j.events()==committed
        # Changed and regressive post-base prefixes are rejected before dependent work.
        at[0] = stamp('2026-09-29T18:30:00')
        changed = synthetic.copy(); changed.loc[changed.date.eq('2026-09-18'),'volume'] += 1
        data49.snapshot(j, changed, dict(provider='SYNTHETIC_TEST_ONLY_CORRUPTION_CASE'))
        bad = cycle(backend, operation)
        assert bad['status']=='FAILED' and 'revised or omitted' in bad['error']
        evidence = dict(status='PASS', synthetic_test_only=True, stages=stages,
             model_package_sha256=sha(PROJECT/'research_v49/results/bootstrap_package.json'),
             seed_predictions=60, ensemble_predictions=20, synthetic_prediction_dates=['2026-09-18'],
             label={k:label[k] for k in ['status','entry_date','exit_date','joint_completed','exec_return','actual_up']},
             evaluation_status=evaluation['status'], statistical_comparisons=0, future_parameter_fits=0,
             no_september25_backfill=True, changed_prefix_blocked=True,
             fixture_files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()})
    assert original_runtime == {relative(p):sha(p) for p in previous.base.RUNTIME.rglob('*') if p.is_file()}
    return evidence


def main():
    freeze = check_freeze()
    assert old_evidence() == freeze['old_evidence']
    work = (PROJECT/'_work').resolve()
    # Keep synthetic evidence in a clearly named isolated folder for inspection.
    fixture = Path(tempfile.mkdtemp(prefix='r51_SYNTHETIC_TEST_ONLY_', dir=work)).resolve()
    guard(fixture.is_relative_to(work), 'Fixture path outside intended workspace')
    try:
        cases = boundary_tests(fixture / 'boundaries')
        print(encoded(dict(status='BOUNDARIES_PASS', cases=len(cases))).decode('utf-8'), flush=True)
        replay = real_model_rehearsal(fixture / 'end_to_end')
        save(OUT/'synthetic_rehearsal.json', replay)
        cases.extend(['real_models_snapshot_prediction_pending_mature_chain',
                      'synthetic_events_never_enter_real_journal', 'real_prediction_and_label_reruns_are_idempotent',
                      'actual_label_matches_hand_computed_open_return', 'one_mature_fixture_has_no_primary_tests',
                      'missed_friday_and_late_append_are_not_backfilled', 'changed_post_base_prefix_blocks_cycle'])
        check_freeze(); assert old_evidence()==freeze['old_evidence']
        save(OUT/'verification.json', dict(status='PASS', completed_utc=iso(utc()),
             freeze_sha256=sha(OUT/'freeze.json'), cases=cases, old_files_preserved=len(freeze['old_evidence']),
             synthetic_fixture_directory=relative(fixture), synthetic_rehearsal_sha256=sha(OUT/'synthetic_rehearsal.json'),
             actual_prospective_predictions=0, actual_future_parameter_fits=0,
             real_runtime_files={relative(p):sha(p) for p in previous.base.RUNTIME.rglob('*') if p.is_file()}))
        print(encoded(dict(status='PASS', cases=len(cases), actual_predictions=0, fixture=relative(fixture))).decode('utf-8'))
    except Exception as exc:
        save(fixture/'failure.json',dict(status='FAILED',error_type=type(exc).__name__,error=str(exc)))
        raise


if __name__ == '__main__':
    main()

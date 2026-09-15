"""One serialized manual cycle; no CLI clock injection or scheduler."""
from contextlib import contextmanager
from datetime import datetime
import os
import uuid

from common51 import *
import build50
import data49
import run50
from evaluate49 import summarize


def latest_snapshot(events):
    rows = [e for e in events if e['kind'] == 'snapshot']
    return rows[-1] if rows else None


def due_cutoffs(events, at):
    existing = {e['key'] for e in events if e['kind'] == 'parameter_package'}
    allowed = {quarter_for(d) for d in cohort()['signal_slots']}
    return sorted(c for c in allowed if c >= '2026-09-30' and c not in existing
                  and at >= datetime.fromisoformat(c + 'T18:00:00+08:00'))


class Backend:
    def __init__(self, journal):
        self.journal = journal

    def now(self):
        return utc()

    def events(self):
        return self.journal.events()

    def validate(self):
        check_freeze()
        build50.registered(self.journal)
        # Reject a regressive/revised latest snapshot before any dependent work.
        snaps = [e for e in self.events() if e['kind'] == 'snapshot']
        if len(snaps) >= 2:
            before, after = snaps[-2:]
            data49.unchanged_prefix(data49.load_snapshot(self.journal, before),
                                    data49.load_snapshot(self.journal, after), before['payload']['end'])

    def fetch(self):
        return data49.fetch(self.journal)

    def refresh(self, cutoff):
        return build50.refresh(self.journal, cutoff)

    def record(self):
        return run50.record(self.journal)

    def settle(self):
        return run50.settle(self.journal)

    def evaluate(self):
        events = self.events()
        old = [e for e in events if e['kind'] == 'primary_evaluation']
        if old:
            return old[0]['payload']
        result = summarize(events, self.now())
        if result['status'] == 'PRIMARY_READY':
            self.journal.append('primary_evaluation', 'fixed_52_slots', result)
        return result


def describe(backend):
    at = backend.now()
    local = at.astimezone(TZ)
    date = local.strftime('%Y-%m-%d')
    events = backend.events()
    snap = latest_snapshot(events)
    pred = {e['key'] for e in events if e['kind'] == 'prediction'}
    labels = {e['key'] for e in events if e['kind'] == 'label'}
    future = [d for d in cohort()['signal_slots'] if d not in pred
              and at < datetime.fromisoformat(d + 'T23:00:00+08:00')]
    cutoffs = sorted({quarter_for(d) for d in cohort()['signal_slots']}
                     - {e['key'] for e in events if e['kind'] == 'parameter_package'})
    window = date in cohort()['signal_slots'] and record_window(date, at)
    return dict(now_utc=iso(at),date_cst=date,
                status='PRE_CLOSE_WAIT' if local.hour < 18 else 'READY_FOR_MANUAL_CYCLE',
                recording_window_open=window, latest_snapshot_end=snap['payload']['end'] if snap else None,
                fetch_needed=local.hour >= 18 and local.weekday() < 5
                    and (snap is None or snap['payload']['end'] < date),
                due_parameter_cutoffs=due_cutoffs(events, at),
                next_parameter_cutoff=cutoffs[0] if cutoffs else None,
                next_unrecorded_signal=future[0] if future else None,
                recorded_predictions=len(pred), labels=len(labels), pending_labels=len(pred - labels),
                journal_events=len(events), scheduler_installed=False)


@contextmanager
def cycle_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / 'cycle.lock'
    exclusive(lock, encoded(dict(pid=os.getpid(), started_utc=iso(utc()))))
    try:
        yield
    finally:
        lock.unlink()


def cycle(backend, directory):
    # Internal backend injection is solely a test seam; run51 exposes no overrides.
    with cycle_lock(directory):
        run_id = uuid.uuid4().hex
        result = dict(run_id=run_id, started_utc=iso(backend.now()), steps=[],
                      source_freeze_sha256=sha(OUT / 'freeze.json'))
        # Write-ahead marker survives a process interruption; final receipt is separate.
        save(directory / 'runs' / (run_id + '.started.json'), result)
        try:
            backend.validate()
            before = describe(backend)
            result['before'] = before
            if before['status'] == 'PRE_CLOSE_WAIT':
                result['status'] = 'PRE_CLOSE_WAIT'
            else:
                if before['fetch_needed']:
                    event = backend.fetch()
                    result['steps'].append(dict(action='fetch', sequence=event['sequence'],
                                                end=event['payload']['end']))
                    backend.validate()
                for cutoff in due_cutoffs(backend.events(), backend.now()):
                    snap = latest_snapshot(backend.events())
                    if snap is None or snap['payload']['end'] < cutoff or datetime.fromisoformat(snap['recorded_utc']) < datetime.fromisoformat(cutoff + 'T18:00:00+08:00'):
                        result['steps'].append(dict(action='refresh', cutoff=cutoff, status='DATA_PENDING'))
                        break
                    event = backend.refresh(cutoff)
                    result['steps'].append(dict(action='refresh', cutoff=cutoff, sequence=event['sequence']))
                # Fetch/refit may be slow: compute date and recording eligibility again.
                current = describe(backend)
                date = current['date_cst']
                if current['recording_window_open']:
                    events = backend.events()
                    snap = latest_snapshot(events)
                    if any(e['kind'] == 'prediction' and e['key'] == date for e in events):
                        result['steps'].append(dict(action='record', status='ALREADY_RECORDED'))
                    elif snap is None or snap['payload']['end'] != date or datetime.fromisoformat(snap['recorded_utc']).astimezone(TZ).strftime('%Y-%m-%d') != date:
                        result['steps'].append(dict(action='record', status='DATA_PENDING'))
                    else:
                        event = backend.record()
                        result['steps'].append(dict(action='record', sequence=event['sequence'], date=date))
                else:
                    result['steps'].append(dict(action='record', status='OUTSIDE_RECORDING_WINDOW'))
                if latest_snapshot(backend.events()) is not None:
                    settlements = backend.settle()
                    result['steps'].append(dict(action='settle', committed=sum(e.get('kind') == 'label' for e in settlements),
                                                pending=sum(e.get('status') == 'PENDING' for e in settlements)))
                result['evaluation'] = backend.evaluate()
                current = describe(backend)
                result['status'] = 'DATA_PENDING' if current['fetch_needed'] or any(s.get('status') == 'DATA_PENDING' for s in result['steps']) else 'COMPLETED'
            result['after'] = describe(backend)
        except Exception as exc:
            # Already committed evidence is retained, never rolled back or replaced.
            result.update(status='FAILED', error_type=type(exc).__name__, error=str(exc))
        result['completed_utc'] = iso(backend.now())
        save(directory / 'runs' / (run_id + '.json'), result)
        return result

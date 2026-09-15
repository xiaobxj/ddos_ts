"""Run the unchanged research engine in a separate process for the local UI."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'research_v51'))
from common51 import Journal, TZ, RUNTIME, check_freeze, cohort, iso, utc
from cycle51 import Backend, cycle, describe
from build50 import registered


def emit(kind, **payload):
    print(json.dumps(dict(event=kind, **payload), ensure_ascii=False, allow_nan=False), flush=True)


def dashboard(events, plan, packages, at):
    """Read-only descriptive presentation; never computes a new primary test."""
    predictions = {e['key']: e for e in events if e['kind'] == 'prediction'}
    labels = {e['key']: e for e in events if e['kind'] == 'label'}
    weeks, forecasts, metrics = [], {}, {}
    for date in cohort()['signal_slots']:
        pred, label = predictions.get(date), labels.get(date)
        value = label['payload'] if label else None
        if value:
            state = 'COMPLETE' if value['status'] == 'MATURE' else 'INVALID_LABEL'
        elif pred:
            state = 'PENDING_LABEL'
        elif at < datetime.fromisoformat(date + 'T23:00:00+08:00'):
            state = 'NOT_DUE'
        else:
            state = 'NO_TIMELY_PREDICTION'
        weeks.append(dict(date=date, status=state, recorded_utc=pred['recorded_utc'] if pred else None,
                          label=value, package=pred['payload'].get('package_file') if pred else None))
        if pred:
            f = pred['payload']['forecast']
            forecasts[date] = {k: f[k] for k in ['signal_date', 'state', 'ensemble_predictions', 'controls']}
        if state == 'COMPLETE':
            y = value['actual_up']
            for row in forecasts[date]['ensemble_predictions']:
                key = (row['history'], row['method'])
                m = metrics.setdefault(key, dict(history=key[0], method=key[1], n=0, correct=0, squared_error=0.0))
                m['n'] += 1
                m['correct'] += int((row['probability'] > .5) == y)
                m['squared_error'] += (row['probability'] - y) ** 2
    for m in metrics.values():
        m['accuracy'] = m['correct'] / m['n']
        m['brier'] = m.pop('squared_error') / m['n']
    saved = [e['payload'] for e in events if e['kind'] == 'primary_evaluation']
    return dict(plan=plan, weeks=weeks, forecasts=forecasts, metrics=list(metrics.values()),
                packages=packages, latest_prediction=max(predictions, default=None),
                mature_signals=sum(w['status'] == 'COMPLETE' for w in weeks),
                evaluation_status='PRIMARY_SAVED' if saved else 'DESCRIPTIVE_ONLY',
                primary_review_not_before=cohort()['cohort']['primary_review_not_before'],
                updated_utc=iso(at))


def inspect():
    check_freeze(True)
    backend = Backend(Journal())
    backend.validate()
    packages = [{k: p[k] for k in ['package_id', 'annual_cutoff', 'quarter_cutoff', 'valid_from', 'valid_until']}
                for _, p in registered(backend.journal)]
    return dashboard(backend.events(), describe(backend), packages, utc())


class ProgressBackend(Backend):
    def stage(self, name, text, action):
        emit('progress', stage=name, state='running', message=text)
        result = action()
        emit('progress', stage=name, state='done', message=text + ' · 完成')
        return result

    def validate(self):
        return self.stage('validate', '检查数据、参数与记录', super().validate)

    def fetch(self):
        return self.stage('fetch', '采集并存档最新日线', super().fetch)

    def refresh(self, cutoff):
        label = '年度滚动重训' if cutoff.endswith('12-31') else '季度参数更新'
        return self.stage('refresh', f'{label}：{cutoff}，请保持程序运行', lambda: super(ProgressBackend, self).refresh(cutoff))

    def record(self):
        return self.stage('record', '计算各方案并记录本周预测', super().record)

    def settle(self):
        return self.stage('settle', '结算已成熟的历史标签', super().settle)

    def evaluate(self):
        return self.stage('evaluate', '汇总固定样本结果', super().evaluate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inspect', 'cycle'])
    args = parser.parse_args()
    result = None
    try:
        if args.action == 'cycle':
            check_freeze(True)
            result = cycle(ProgressBackend(Journal()), RUNTIME)
        else:
            emit('progress', stage='inspect', state='running', message='读取当前状态')
        data = inspect()
        emit('result', ok=result is None or result['status'] != 'FAILED', result=result, data=data)
    except Exception as exc:
        emit('result', ok=False, result=result, error_type=type(exc).__name__, error=str(exc))
        raise SystemExit(1)


if __name__ == '__main__':
    main()

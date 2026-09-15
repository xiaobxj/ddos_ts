"""Operational regression tests; all scored examples are isolated old fixtures."""
import copy
from datetime import datetime
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import app
import worker

PROJECT=app.PROJECT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v=json.loads((PROJECT/'research_v51/results/verification.json').read_text(encoding='utf-8'))
        cls.fixture=PROJECT/v['synthetic_fixture_directory']/'end_to_end/SYNTHETIC_TEST_ONLY_journal'
        cls.events=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((cls.fixture/'events').glob('*.json'))]

    def render(self,events):
        return worker.dashboard(events,{},[],datetime.fromisoformat('2026-09-28T18:30:00+08:00'))

    def test_pending_is_not_counted_as_accuracy(self):
        events=[e for e in self.events if e['kind']!='label']
        data=self.render(events)
        self.assertEqual(data['metrics'],[])
        self.assertEqual(data['mature_signals'],0)
        self.assertEqual(data['weeks'][0]['status'],'PENDING_LABEL')

    def test_mature_labels_have_one_common_sample_for_all20(self):
        data=self.render(self.events)
        self.assertEqual(data['mature_signals'],1)
        self.assertEqual(len(data['metrics']),20)
        self.assertEqual({m['n'] for m in data['metrics']},{1})
        for m in data['metrics']:
            row=next(r for r in data['forecasts']['2026-09-18']['ensemble_predictions'] if (r['history'],r['method'])==(m['history'],m['method']))
            self.assertEqual(m['brier'],(row['probability']-1)**2)
            self.assertEqual(m['correct'],int(row['probability']>.5))

    def test_invalid_label_is_explicit_and_excluded(self):
        events=copy.deepcopy(self.events)
        next(e for e in events if e['kind']=='label')['payload']['status']='INVALID'
        data=self.render(events)
        self.assertEqual(data['weeks'][0]['status'],'INVALID_LABEL')
        self.assertEqual(data['metrics'],[])

    def test_missing_friday_retained_among_all52(self):
        data=self.render(self.events)
        self.assertEqual(len(data['weeks']),52)
        self.assertEqual(data['weeks'][1]['status'],'NO_TIMELY_PREDICTION')
        self.assertEqual(data['evaluation_status'],'DESCRIPTIVE_ONLY')
        self.assertNotIn('comparisons',data)

    def test_saved_primary_status_only_no_recalculation(self):
        events=self.events+[dict(kind='primary_evaluation',payload={'status':'PRIMARY_READY'})]
        self.assertEqual(self.render(events)['evaluation_status'],'PRIMARY_SAVED')


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.jobs=app.Jobs()
        self.server=app.Server(self.jobs,'test-token')
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.headers={'X-App-Token':'test-token','Origin':self.server.origin}

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)

    def request(self,path,method='GET',headers=None,body=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        conn.request(method,path,body=body,headers=self.headers if headers is None else headers)
        response=conn.getresponse();value=response.read();status=response.status;meta=dict(response.getheaders());conn.close()
        return status,value,meta

    def test_only_loopback_and_authorized_local_requests(self):
        self.assertEqual(self.server.server_address[0],'127.0.0.1')
        self.assertEqual(self.request('/api/state',headers={})[0],403)
        self.assertEqual(self.request('/api/state',headers={**self.headers,'Origin':'http://example.test'})[0],403)
        self.assertEqual(self.request('/api/state',headers={**self.headers,'Host':'example.test'})[0],403)
        self.assertEqual(self.request('/api/state')[0],200)

    def test_no_arbitrary_paths_or_time_override(self):
        self.assertEqual(self.request('/../prospective_r49/genesis.json')[0],404)
        self.assertEqual(self.request('/api/cycle?at=2026-09-18','POST')[0],404)
        self.assertEqual(self.request('/api/cycle','POST',body='{"at":"2026-09-18"}')[0],400)

    def test_html_assets_are_utf8_and_self_contained(self):
        for path,needle in [('/','每周预测'),('/app.js','运行本周流程'),('/style.css','@media')]:
            status,raw,headers=self.request(path)
            self.assertEqual(status,200)
            self.assertIn(needle,raw.decode('utf-8'))
            self.assertIn("frame-ancestors 'none'",headers['Content-Security-Policy'])

    def test_double_click_cannot_create_second_job(self):
        with patch.object(self.jobs,'run') as run:
            self.assertEqual(self.request('/api/cycle','POST')[0],202)
            self.assertEqual(self.request('/api/cycle','POST')[0],409)
            self.assertEqual(self.request('/api/inspect','POST')[0],409)
            self.assertEqual(self.request('/api/quit','POST')[0],409)
        self.assertTrue(self.jobs.snapshot()['busy'])

    def test_unauthorized_post_never_starts_job(self):
        with patch.object(self.jobs,'run') as run:
            self.assertEqual(self.request('/api/cycle','POST',headers={})[0],403)
            run.assert_not_called()

    def test_shutdown_reservation_prevents_racing_new_job(self):
        self.assertTrue(self.jobs.request_stop())
        self.assertFalse(self.jobs.start('cycle'))


class WorkerTests(unittest.TestCase):
    def test_real_worker_and_server_job_preclose_preserve_scientific_journal(self):
        self.assertLess(worker.utc().astimezone(worker.TZ).hour,18,'Run this acceptance test only before 18:00 CST')
        before={str(p):digest(p) for p in (PROJECT/'prospective_r49').rglob('*') if p.is_file()}
        with tempfile.TemporaryDirectory(prefix='weekly_app_test_',dir=PROJECT/'_work') as temp:
            self.assertTrue(Path(temp).resolve().is_relative_to((PROJECT/'_work').resolve()))
            with patch.object(app,'STATE',Path(temp)):
                jobs=app.Jobs()
                self.assertTrue(jobs.start('inspect'))
                limit=time.monotonic()+45
                while jobs.snapshot()['busy'] and time.monotonic()<limit:time.sleep(.1)
                state=jobs.snapshot();self.assertFalse(state['busy']);self.assertIsNone(state['error'])
                self.assertEqual(state['data']['plan']['recorded_predictions'],0)
                self.assertTrue(jobs.start('cycle'))
                limit=time.monotonic()+45
                while jobs.snapshot()['busy'] and time.monotonic()<limit:time.sleep(.1)
                state=jobs.snapshot();self.assertFalse(state['busy']);self.assertIsNone(state['error'])
                self.assertEqual(state['result']['status'],'PRE_CLOSE_WAIT')
                self.assertEqual(state['result']['steps'],[])
                self.assertTrue(any('检查数据' in l['message'] for l in state['log']))
                self.assertEqual(state['data']['metrics'],[])
        after={str(p):digest(p) for p in (PROJECT/'prospective_r49').rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_worker_exit_failure_releases_busy_state(self):
        with tempfile.TemporaryDirectory(prefix='weekly_app_failure_',dir=PROJECT/'_work') as temp:
            self.assertTrue(Path(temp).resolve().is_relative_to((PROJECT/'_work').resolve()))
            with patch.object(app,'STATE',Path(temp)),patch.object(app.subprocess,'Popen',side_effect=OSError('fixture launch failure')):
                jobs=app.Jobs();jobs.state['busy']=True;jobs.run('inspect')
                self.assertFalse(jobs.snapshot()['busy'])
                self.assertIn('fixture launch failure',jobs.snapshot()['error'])


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() else 'FAILED',tests_run=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),
                scope='HTTP, concurrency, read-only metrics and real preclose worker; no browser automation available')
    output=PROJECT/'_work/weekly_app_backend_verification.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)

from pathlib import Path
import json,sys
from datetime import datetime
PROJECT=Path('D:/ddos_v3');sys.path.insert(0,str(PROJECT/'weekly_app'))
import worker
from cycle51 import describe,Backend
v=json.loads((PROJECT/'research_v51/results/verification.json').read_text(encoding='utf-8'))
root=PROJECT/v['synthetic_fixture_directory']/'end_to_end/SYNTHETIC_TEST_ONLY_journal'
events=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'events').glob('*.json'))]
plan=describe(Backend(worker.Journal(root)))
packages=[dict(package_id='fixture',valid_from='2026-07-01',valid_until='2026-09-30')]
data=worker.dashboard(events,plan,packages,datetime.fromisoformat('2026-09-28T18:30:00+08:00'))
data['synthetic_test_only']=True
(PROJECT/'_work/weekly_app_ui_fixture.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print('SYNTHETIC_UI_FIXTURE_READY')

"""Check actual hidden-server lifecycle and preservation of the frozen research."""
from pathlib import Path
import json,os,subprocess,sys,time,urllib.request
PROJECT=Path('D:/ddos_v3');sys.path.insert(0,str(PROJECT/'weekly_app'));import app
sys.path.insert(0,str(PROJECT/'research_v51'));from common51 import check_freeze,old_evidence,sha,relative

def read(p):return json.loads(p.read_text(encoding='utf-8'))
preserved=dict(read(PROJECT/'research_v51/results/freeze.json')['old_evidence'])
prior=PROJECT/'research_v51';delivery=read(prior/'results/delivery_manifest.json')
preserved.update({relative(prior/n):h for n,h in delivery['files'].items()})
preserved[relative(prior/'results/delivery_manifest.json')]=sha(prior/'results/delivery_manifest.json')
assert len(preserved)==6296
before={relative(p):sha(p) for p in (PROJECT/'prospective_r49').rglob('*') if p.is_file()}
assert len(before)==5

process=subprocess.Popen([str(app.PYTHON),'-B',str(app.ROOT/'app.py'),'--no-browser'],cwd=PROJECT/'_work',
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
    env=dict(os.environ,PYTHONUTF8='1'))
def state_request(info,path='state',method='GET'):
    req=urllib.request.Request(info['origin']+'/api/'+path,method=method,headers={'X-App-Token':info['token']})
    with urllib.request.urlopen(req,timeout=5) as response:return json.load(response)
deadline=time.monotonic()+20;info=None
while time.monotonic()<deadline:
    try:
        info=read(app.STATE/'server.json');state=state_request(info)
        if not state['busy'] and state['data'] is not None:break
    except (OSError,ValueError):pass
    time.sleep(.15)
else:raise AssertionError('Actual app did not become ready')
assert state['app_id']==app.APP_ID and state['error'] is None and state['data']['plan']['recorded_predictions']==0
pid=info['pid'];assert app.process_alive(pid)
again=subprocess.run([str(app.PYTHON),'-B',str(app.ROOT/'app.py'),'--no-browser'],cwd=PROJECT/'_work',capture_output=True,timeout=15)
assert again.returncode==0 and read(app.STATE/'server.json')['pid']==pid
assert state_request(info)['data']['plan']['recorded_predictions']==0
assert state_request(info,'quit','POST')['stopped'] is True
deadline=time.monotonic()+10
while (app.STATE/'app.lock').exists() and time.monotonic()<deadline:time.sleep(.1)
assert not (app.STATE/'app.lock').exists()
process.communicate(timeout=10)
assert process.returncode==0
assert before=={relative(p):sha(p) for p in (PROJECT/'prospective_r49').rglob('*') if p.is_file()}
check_freeze(True)
for name,h in preserved.items():assert sha(PROJECT/name)==h,name
for filename in ['打开每周预测.cmd','打开每周预测.vbs']:
    assert (PROJECT/filename).is_file()
assert (PROJECT/'research_v4/.venv_gpu/Scripts/pythonw.exe').is_file()
backend=read(PROJECT/'_work/weekly_app_backend_verification.json')
frontend=read(PROJECT/'_work/weekly_app_frontend_verification.json')
assert backend['status']==frontend['status']=='PASS'
receipt=dict(status='PASS',backend_tests=backend['tests_run'],frontend_logic_checks=len(frontend['checks']),
    lifecycle_checks=['hidden_loopback_server_starts','open_only_inspects','repeat_launch_reuses_server','quit_cleans_lock_and_exits'],
    frozen_files_preserved=len(preserved),real_journal_files_unchanged=before,
    real_predictions=0,real_labels=0,
    browser_visual_verification='NOT_AVAILABLE: CUA returned no browser or native app surfaces. HTTP and Node VM tests do not replace visual QA.',
    files={relative(p):sha(p) for p in app.ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
(app.ROOT/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k not in ['files','real_journal_files_unchanged']},ensure_ascii=False))

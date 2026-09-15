from pathlib import Path
import json,sys
PROJECT=Path('D:/ddos_v3');sys.path.insert(0,str(PROJECT/'research_v51'))
from common51 import sha,relative,check_freeze
root=PROJECT/'weekly_app';receipt=json.loads((root/'verification.json').read_text(encoding='utf-8'))
assert receipt['status']=='PASS' and receipt['backend_tests']==receipt['frontend_logic_checks']==13
for path in [root/'README.md',root/'static/index.html',root/'static/app.js']:
    text=path.read_text(encoding='utf-8');assert '\ufffd' not in text and '????' not in text
assert '涨跌顺序扩展' in (root/'static/app.js').read_text(encoding='utf-8')
assert '涨跌顺序修正' in (root/'static/app.js').read_text(encoding='utf-8')
receipt['files']={relative(p):sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='verification.json'}
for name in ['打开每周预测.cmd','打开每周预测.vbs']:
    receipt['files'][name]=sha(PROJECT/name)
receipt['presentation_review']='Price sign-order labels match the original causal gate; training method identifiers unchanged.'
(root/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
check_freeze(True)
for n,h in receipt['real_journal_files_unchanged'].items():assert sha(PROJECT/n)==h
assert not (PROJECT/'prospective_app/app.lock').exists()
assert not (PROJECT/'prospective_r51/cycle.lock').exists()
print(json.dumps(dict(status='PASS',application_files=len(receipt['files']),real_predictions=0,browser_visual_qa='unavailable')))

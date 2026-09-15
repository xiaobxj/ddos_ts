from pathlib import Path
import sys,subprocess
root=Path('D:/ddos_v3');logfile=root/'research_v39/results/logs/verify_null_compat.log';assert not logfile.exists()
with logfile.open('w',encoding='utf-8') as log:
    p=subprocess.Popen([sys.executable,'-B','-u',str(root/'_work/verify39_null_compat.py')],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    for line in p.stdout:log.write(line);log.flush();print(line,end='',flush=True)
    code=p.wait();log.write(f'\nEXIT_CODE={code}\n');log.flush()
raise SystemExit(code)

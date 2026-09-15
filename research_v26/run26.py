"""Run sequential frozen phases; retain failure logs and never overwrite outputs."""
from pathlib import Path
import sys,subprocess,time
ROOT=Path(__file__).resolve().parent
def main():
    logs=ROOT/'results/logs';logs.mkdir(parents=True,exist_ok=True)
    for phase in sys.argv[1:]:
        assert phase in ['prepare','contract','train','score','evaluate','verify','report','delivery']
        path=logs/f'{phase}.log';assert not path.exists()
        with path.open('w',encoding='utf-8') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/f'{phase}26.py')],cwd=ROOT.parent,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in p.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=p.wait();log.write(f'\nEXIT_CODE={code}\n');log.flush()
        if code:raise SystemExit(code)
        print(f'Phase {phase} finished.',flush=True)
if __name__=='__main__':main()

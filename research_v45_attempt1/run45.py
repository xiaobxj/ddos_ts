from pathlib import Path
import sys,subprocess,json
ROOT=Path(__file__).resolve().parent
def main():
    logs=ROOT/'results/logs';logs.mkdir(parents=True,exist_ok=True);cfg=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    for phase in sys.argv[1:]:
        assert phase in ['prepare','contract','fit','validate','score','evaluate','verify','report'];path=logs/f'{phase}.log';assert not path.exists();exe=cfg['report_python'] if phase=='report' else sys.executable
        with path.open('w',encoding='utf-8') as log:
            p=subprocess.Popen([exe,'-B','-u',str(ROOT/f'{phase}45.py')],cwd=ROOT.parent,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in p.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=p.wait();log.write(f'\nEXIT_CODE={code}\n');log.flush()
        if code:raise SystemExit(code)
        print(f'Phase {phase} complete.',flush=True)
if __name__=='__main__':main()

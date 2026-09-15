"""Create a complete, immutable round8 delivery inventory after report QA."""
from pathlib import Path
import json
import hashlib

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'


def main():
    verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    assert verification['status']=='PASS'
    report=OUT/'第八轮测试报告.md';assert report.exists()
    excluded={'delivery_manifest.json'}
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in excluded}
    data=dict(experiment=8,report=str(report.relative_to(ROOT)),formal_new_neural_fits=45,reused_neural_checkpoints=9,
              linear_models=18,withdrawn_preflight_neural_fits=4,validation_weeks=141,files=files)
    (OUT/'delivery_manifest.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(files=len(files),report=str(report)),ensure_ascii=False))


if __name__=='__main__':main()

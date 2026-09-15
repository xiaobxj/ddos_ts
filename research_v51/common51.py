"""Manual orchestration of the frozen R49/R50 prospective experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'research_v50'))
import common50 as previous
from common50 import (Journal, TZ, cohort, encoded, exclusive, guard, iso,
                      quarter_for, read, record_window, relative, save, sha, utc)

OUT = ROOT / 'results'
RUNTIME = PROJECT / 'prospective_r51'


def old_evidence():
    files = dict(read(PROJECT / 'research_v50/results/freeze.json')['old_evidence'])
    prior = PROJECT / 'research_v50'
    manifest = prior / 'results/delivery_manifest.json'
    files.update({relative(prior / n): h for n, h in read(manifest)['files'].items()})
    files[relative(manifest)] = sha(manifest)
    for p in (PROJECT / 'research_v50_attempt1').rglob('*'):
        if p.is_file():
            files[relative(p)] = sha(p)
    files.update(read(prior / 'results/runtime_handoff_manifest.json')['files'])
    guard(len(files) == 6279, 'Unexpected R50 handoff file count')
    for name, digest in files.items():
        guard(sha(PROJECT / name) == digest, f'Prior evidence changed: {name}')
    return files


def check_freeze(verified=False):
    previous.check_freeze(True)
    freeze = read(OUT / 'freeze.json')
    for name, digest in freeze['immutable_files'].items():
        guard(sha(PROJECT / name) == digest, f'R51 frozen file changed: {name}')
    if verified:
        receipt = read(OUT / 'verification.json')
        guard(receipt['status'] == 'PASS' and receipt['freeze_sha256'] == sha(OUT / 'freeze.json'),
              'R51 has not passed verification')
    return freeze

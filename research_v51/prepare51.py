"""Freeze the bounded operational implementation, not a new model experiment."""
from common51 import *


def main():
    prior = old_evidence()
    save(ROOT / 'protocol.json', dict(
        version=51, scope='manual operational orchestration and isolated engineering verification',
        scientific_protocol_sha256=sha(PROJECT / 'research_v49/protocol.json'),
        parameter_implementation_freeze_sha256=sha(PROJECT / 'research_v50/results/freeze.json'),
        no_recipe_or_selection_change=True, original_histories=previous.HISTORIES,
        original_methods=previous.METHODS, original_seeds=previous.SEEDS,
        clock='Production uses actual system UTC, interpreted in UTC+8; CLI has no timestamp override.',
        before_18='Read-only journal inspection, operational receipt only; no fetch, refit, prediction or label append.',
        sequence=['validate provenance and journal', 'fetch weekday final bars if latest snapshot ends before today',
                  'validate unchanged observed prefix', 'refresh missing due cohort quarters in chronological order',
                  'recheck clock and record only current Friday 18:00 inclusive to 23:00 exclusive',
                  'settle recorded predictions using original joint maturity rule', 'reuse or generate original fixed-cohort evaluation'],
        data_pending='Missing current Friday is never replaced by a Thursday or a historical backfill. No exchange holiday calendar is inferred.',
        cutoff='Use original R50 request checks; no fitting before cutoff at 18:00 and covering observed snapshot. December delegates to unchanged annual refit.',
        locking='Exclusive entire-cycle lock for run51, plus original per-append journal lock. Do not run older write commands concurrently.',
        failure='Stop dependent steps, retain append-only scientific evidence and per-run write-ahead/final receipts; no automatic retry or lock removal.',
        evaluation='Unchanged R49 descriptive-only checks until fixed final review; save primary event once.',
        synthetic='Test-only clock/feed substitutions, original production record/settle functions and models, isolated directory with synthetic_test_only genesis marker; never counted as real predictions.',
        scheduler_installed=False))
    sources = sorted(ROOT.glob('*.py')) + [ROOT / 'run.ps1', ROOT / 'protocol.json']
    save(OUT / 'freeze.json', dict(status='FROZEN', created_utc=iso(utc()),
         immutable_files={relative(p): sha(p) for p in sources}, old_evidence=prior,
         real_runtime_files=read(PROJECT / 'research_v50/results/runtime_handoff_manifest.json')['files']))
    print(encoded(dict(status='FROZEN', sources=len(sources), old_files=len(prior))).decode('utf-8'))


if __name__ == '__main__':
    main()

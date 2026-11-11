"""Verify this maintenance pass without retraining or changing frozen evidence."""
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'verification.json'
STUDIES = ('state_recipe_development', 'state_reset_exposure', 'state_momentum_training',
           'copy_marker_training', 'copy_marker_alignment')
CHANGED = ('AGENTS.md', 'README.md', 'docs/research-plan.md',
           'docs/assessments/research-checkpoint.md', 'docs/README.md',
           'docs/assessments/README.md', '.gitignore', 'tests/run_tests.py',
           'tests/test_runner_safety.py', 'tests/README.md',
           'docs/assessments/research-history.md')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = json.loads((HERE / 'before.json').read_text())
    altered = [p for p, h in before['protected'].items() if sha(ROOT / p) != h]
    assert not altered, altered
    for row in before['snapshots']:
        assert sha(HERE / 'snapshots' / row['path']) == row['sha256']
    history = (ROOT / 'docs/assessments/research-history.md').read_text(encoding='utf-8')
    assert history.endswith((HERE / 'snapshots/docs/assessments/research-checkpoint.md').read_text(encoding='utf-8'))
    old_rules = (HERE / 'snapshots/AGENTS.md').read_text(encoding='utf-8')
    new_rules = (ROOT / 'AGENTS.md').read_text(encoding='utf-8')
    permitted = lambda s: [line for line in s.splitlines() if line.startswith('| High training')
                          or line.startswith('| Low training') or line.startswith('| Short held-out')
                          or line.startswith('| Weak groups')]
    assert permitted(old_rules) == permitted(new_rules)
    review = json.loads((ROOT / 'results/project_review_20261004/review_verification.json').read_text())
    assert sha(ROOT / 'docs/assessments/project-quality-review.md') == review['review_sha256']
    provenance = []
    for name in STUDIES:
        folder = ROOT / 'results' / name
        report = json.loads((folder / 'report.json').read_text())
        audit = json.loads((folder / 'audit.json').read_text())
        assert audit['passed'] and audit['report_sha256'] == sha(folder / 'report.json')
        dependencies = report['source_dependencies']
        assert all(sha(ROOT / p) == h for p, h in dependencies.items()), name
        provenance.append(dict(study=name,dependencies_checked=len(dependencies),passed=True))

    runs = []
    for name in ('maintenance_first', 'maintenance_second'):
        path = ROOT / 'results/test_runs' / name / 'summary.json'
        summary = json.loads(path.read_text())
        assert summary['passed'] and summary['methods'] == 15
        assert sum(summary['initialized_equation_gradient_cases'].values()) == 63
        runs.append(dict(path=str(path.relative_to(ROOT)).replace('\\', '/'),
                         sha256=sha(path),methods=15,equation_gradient_cases=63,
                         frozen_files_checked=summary['frozen_files_checked']))
    # The first run preceded adding protocol preservation and grouping safety
    # fixtures. The second run checks the final launcher sources.
    second = json.loads((ROOT / runs[-1]['path']).read_text())
    assert all(sha(ROOT / p) == h for p, h in second['source_sha256'].items())
    cli = []
    for name in ('maintenance_second', '../escape'):
        summary_path = ROOT / runs[-1]['path']
        saved_hash = sha(summary_path)
        result = subprocess.run([sys.executable, 'tests/run_tests.py', '--run-name', name],
                                cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 2 and sha(summary_path) == saved_hash
        assert 'Ran 15 tests' not in result.stderr
        cli.append(dict(run_name=name,exit_code=2,test_execution=False,
                        existing_summary_unchanged=True,error=result.stderr.strip()))

    documents = [ROOT / 'README.md', ROOT / 'AGENTS.md', ROOT / 'tests/README.md',
                 *sorted((ROOT / 'docs').rglob('*.md'))]
    checked = 0
    missing = []
    for source in documents:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', source.read_text(encoding='utf-8')):
            target = target.strip().strip('<>')
            if target.startswith(('#', 'http:', 'https:', 'mailto:', 'app:', 'data:')):
                continue
            target = unquote(target.split('#', 1)[0])
            if not target:
                continue
            path = (source.parent / target).resolve()
            checked += 1
            if not path.exists() and path != OUTPUT:
                missing.append(dict(source=str(source.relative_to(ROOT)),target=target))
    assert not missing, missing
    for name in ('tests/run_tests.py', 'tests/test_runner_safety.py'):
        ast.parse((ROOT / name).read_text(encoding='utf-8'), filename=name)
    sizes = [dict(path=row['path'],before_bytes=row['bytes'],
                  after_bytes=(ROOT / row['path']).stat().st_size)
             for row in before['snapshots']]
    out = dict(passed=True,date='2026-10-04',training_updates=0,
               frozen_files_unchanged=len(before['protected']),
               snapshot_hashes_match=True,historical_checkpoint_text_retained=True,
               permitted_conclusion_table_unchanged=True,original_quality_review_hash_matches=True,
               recent_manifests=provenance,dependency_entries_checked=sum(p['dependencies_checked'] for p in provenance),
               safe_test_runs=runs,cli_rejection_checks=cli,
               markdown_documents_checked=len(documents),local_file_links_checked=checked,
               missing_local_file_targets=missing,
               link_scope='File targets only; excludes raw snapshots, anchors and external URLs. This output is created after checks.',
               active_document_sizes=sizes,changed_file_sha256={p:sha(ROOT / p) for p in CHANGED},
               verification_source_sha256=sha(Path(__file__)),
               limits='No new training or exhaustive archived-code audit. Historical standalone test entries remain unsafe; use the current launcher.')
    # This is a maintenance artifact, not a frozen empirical study result.
    OUTPUT.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:out[k] for k in ('passed','frozen_files_unchanged','dependency_entries_checked',
                                       'markdown_documents_checked','local_file_links_checked','active_document_sizes')}))


if __name__ == '__main__':
    main()

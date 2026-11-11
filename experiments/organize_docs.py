"""One bounded documentation migration; preserve executed sources and frozen plans."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
LINK = re.compile(r'\]\(([^)]+)\)')
GUIDES = {'architecture.md', 'interpretability.md', 'test-clarifications.md', 'ffn-investigation.md'}
REPORTS = {'decoder-verification.md', 'first-token-interventions.md', 'results.md',
           'query-position-effects-on-attention-versus-the-residual-path.md',
           'residual-effects-through-ffn-normalization-and-logits.md'}
REFERENCE = {'sources.md', 'claims.md', 'foundation-audit.md'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def bounded(path):
    assert path.resolve().is_relative_to(ROOT), str(path)
    return path


def target(path, href):
    if href.startswith(('http://', 'https://', 'mailto:', '#')):
        return None
    return (path.parent / href.split('#')[0]).resolve()


def main():
    evidence = list((ROOT / 'experiments').rglob('*.py')) + list((ROOT / 'results').rglob('*.json'))
    frozen = set()
    for path in evidence:
        for name in re.findall(r'docs[/\\][^\x27"\s<>]+', path.read_text(encoding='utf-8')):
            candidate = ROOT / name.replace('\\\\', '/').replace('\\', '/')
            if candidate.is_file():
                frozen.add(candidate.resolve())
    frozen.update(p.resolve() for p in DOCS.glob('*protocol.md'))
    frozen.update(p.resolve() for p in (DOCS / 'decisions').glob('*.md'))
    frozen.update(p.resolve() for p in (DOCS / 'figures').iterdir() if p.is_file())
    pinned = set()
    for path in frozen:
        if path.suffix != '.md':
            continue
        for href in LINK.findall(path.read_text(encoding='utf-8')):
            candidate = target(path, href)
            if candidate and candidate.is_relative_to(DOCS) and candidate.is_file():
                pinned.add(candidate)
    moves = {}
    for path in DOCS.glob('*.md'):
        path = path.resolve()
        if path in frozen or path in pinned:
            continue
        if path.name in REFERENCE:
            folder = 'reference'
        elif path.name.endswith(('-guide.md', '-walkthrough.md')) or path.name in GUIDES:
            folder = 'guides'
        elif path.name.endswith('-results.md') or path.name in REPORTS:
            folder = 'results'
        else:
            continue
        destination = bounded(DOCS / folder / path.name)
        assert destination.is_relative_to(DOCS) and not destination.exists()
        moves[path] = destination
    assert moves, 'Migration is already applied or no documents qualify.'
    bodies = {p.resolve(): p.read_text(encoding='utf-8') for p in [ROOT / 'README.md', *DOCS.rglob('*.md')]}
    before = {p: digest(p) for p in bodies}
    frozen_hashes = {p: digest(p) for p in frozen}
    evidence_hashes = {p: digest(p) for p in evidence}

    def future_exists(path):
        return path not in moves and (path in moves.values() or path.exists())

    negative = []
    for path, body in bodies.items():
        for href in LINK.findall(body):
            old = target(path, href)
            if old is None:
                continue
            assert old.exists(), (relative(path), href)
            if not future_exists(target(moves.get(path, path), href)):
                negative.append((relative(moves.get(path, path)), href))
    assert negative, 'Bare-move negative control must detect broken links.'

    def rewrite(body, old, new):
        def replace(match):
            href = match.group(1)
            original = target(old, href)
            if original is None:
                return match.group(0)
            destination = moves.get(original, original)
            fragment = '#' + href.split('#', 1)[1] if '#' in href else ''
            return '](' + os.path.relpath(destination, new.parent).replace('\\', '/') + fragment + ')'
        body = LINK.sub(replace, body)
        for original, destination in moves.items():
            body = body.replace(relative(original), relative(destination))
        return body

    updated = {moves.get(p, p): body if p in frozen else rewrite(body, p, moves.get(p, p)) for p, body in bodies.items()}
    for path, body in updated.items():
        for href in LINK.findall(body):
            local = target(path, href)
            if local is not None:
                assert future_exists(local), (relative(path), href)
    plan = dict(moves=len(moves), frozen_files=len(frozen),
                negative_control_broken_links=len(negative),
                original_link_targets=sorted(relative(p) for p in pinned if p.parent == DOCS),
                moves_by_folder={folder: sum(p.parent.name == folder for p in moves.values())
                                 for folder in ('guides', 'results', 'reference')})
    if '--apply' not in sys.argv:
        print(json.dumps(plan, indent=2))
        return
    manifest_path = DOCS / 'organization.json'
    assert not manifest_path.exists()
    for path, expected in before.items():
        assert digest(path) == expected
    for path in moves.values():
        bounded(path.parent).mkdir(exist_ok=True)
    bounded(DOCS / 'protocols').mkdir(exist_ok=True)
    indexes = [DOCS / 'README.md', DOCS / 'reading-order.md',
               *(DOCS / f / 'README.md' for f in ('guides', 'results', 'reference', 'protocols', 'decisions'))]
    assert all(not p.exists() for p in indexes)
    readme = bodies[ROOT / 'README.md']
    sequence = readme.split('## Read in this order\n', 1)[1].split('\n## Project tree', 1)[0]
    updated[DOCS / 'reading-order.md'] = '# Full research reading order\n\n' + rewrite(sequence, ROOT / 'README.md', DOCS / 'reading-order.md')
    tree = '\n'.join([
        'Transformers_RL/', '|-- README.md', '|-- docs/',
        '|   |-- README.md            # documentation index',
        '|   |-- guides/              # explanations and walkthroughs',
        '|   |-- results/             # study reports, methods and limits',
        '|   |-- reference/           # papers, claims and foundational audit',
        '|   |-- protocols/           # frozen protocol index; future protocols',
        '|   |-- assessments/         # cross-study conclusions and failures',
        '|   |-- decisions/           # frozen research choices',
        '|   |-- figures/             # diagrams and measured plots',
        '|   |-- reading-order.md     # complete earlier reading sequence',
        '|   |-- research-plan.md     # current status and stopping rules',
        '|   |-- *protocol.md         # original paths retained for provenance',
        '|   `-- *.md                 # a few targets linked by frozen protocols',
        '|-- experiments/             # producers and independent audits',
        '|-- results/                 # raw measurements and checkpoints',
        '|-- data/                    # pinned corpus and encoding',
        '|-- requirements-cpu.txt', '|-- requirements-gpu.txt', '`-- requirements-data.txt'])
    intro = readme.split('## Read in this order\n', 1)[0]
    navigation = '## Read in this order\n\nStart with the [architecture assessment](docs/assessments/architecture-assessment.md), then the [evidence summary](docs/assessments/evidence-summary.md).\n\nUse the [documentation index](docs/README.md) for guides, study reports, protocols and sources. The [full research reading order](docs/reading-order.md) preserves the earlier sequence.\n\nConsult the [failure register](docs/assessments/failure-register.md) before repeating a study.\n\n'
    updated[ROOT / 'README.md'] = intro + navigation + '## Project tree\n\n```text\n' + tree + '\n```\n\n## Reproduce the first experiments\n' + readme.split('## Reproduce the first experiments\n', 1)[1]
    updated[DOCS / 'README.md'] = '# Documentation\n\nStart with the [architecture assessment](assessments/architecture-assessment.md) for current conclusions and limits.\n\n| Folder | Contains |\n|---|---|\n| [Guides](guides/README.md) | Short explanations, equations and walkthroughs |\n| [Study reports](results/README.md) | Methods, findings, audits and limits |\n| [Reference](reference/README.md) | Primary sources, claims and paper audit |\n| [Protocols](protocols/README.md) | Frozen plans and evaluation rules |\n| [Assessments](assessments/README.md) | Cross-study conclusions and failures |\n| [Decisions](decisions/README.md) | Research choices frozen before execution |\n| [Figures](figures/) | Standalone diagrams and plots |\n\n[Research plan](research-plan.md) records status and stopping rules. [Full reading order](reading-order.md) preserves the earlier sequence.\n\nExisting protocols retain their bytes and fixed paths for experiment provenance checks. A few guides/reports linked by those protocols also retain their original paths; the indexes include them. New protocols belong in `protocols/`.\n\nMarkdown reports live under `docs/results/`. Raw measurements and checkpoints remain in the workspace `results/` directory.\n'
    assessment = DOCS / 'assessments/README.md'
    updated[assessment] = updated[assessment].replace('Individual methods and results remain in the parent `docs/` folder; raw evidence remains in `results/`.', 'Study reports are indexed in [results](../results/README.md), explanations in [guides](../guides/README.md), and frozen plans in [protocols](../protocols/README.md). Raw evidence remains in the workspace `results/` folder.')
    catalog = {f: [] for f in ('guides', 'results', 'reference', 'protocols', 'decisions')}
    for original, body in bodies.items():
        current = moves.get(original, original)
        if current.parent.name in catalog and current.parent.parent == DOCS:
            folder = current.parent.name
        elif original.parent == DOCS and original.name.endswith('protocol.md'):
            folder = 'protocols'
        elif original.parent == DOCS and (original.name.endswith(('-guide.md', '-walkthrough.md')) or original.name in GUIDES):
            folder = 'guides'
        elif original.parent == DOCS and (original.name.endswith('-results.md') or original.name in REPORTS):
            folder = 'results'
        elif original == DOCS / 'length-exposure-decision.md':
            folder = 'decisions'
        else:
            continue
        title = next((line[2:].strip() for line in body.splitlines() if line.startswith('# ')), original.name)
        catalog[folder].append((title, current))
    titles = {'guides': 'Learning guides', 'results': 'Study reports', 'reference': 'Sources and claims',
              'protocols': 'Frozen experiment protocols', 'decisions': 'Research decisions'}
    for folder, entries in catalog.items():
        index = DOCS / folder / 'README.md'
        intro = '# ' + titles[folder] + '\n\n'
        if folder == 'protocols':
            intro += 'Existing protocols retain their original paths and bytes for provenance checks. Add future protocols here.\n\n'
        elif folder == 'results':
            intro += 'Individual findings, methods and limits. Raw JSON, score tensors and checkpoints remain in the workspace `results/` directory.\n\n'
        else:
            intro += 'Browse by subject. See the [documentation index](../README.md) for the full structure.\n\n'
        updated[index] = intro + '\n'.join('- [' + title + '](' + os.path.relpath(p, index.parent).replace('\\', '/') + ')' for title, p in sorted(entries, key=lambda entry: entry[1].name)) + '\n'
    manifest = dict(status='planned', date='2026-10-01', plan=plan,
                    moved_documents=[dict(original=relative(p), current=relative(q), before_sha256=before[p]) for p, q in sorted(moves.items())],
                    frozen_files={relative(p): h for p, h in sorted(frozen_hashes.items())})
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    for original, destination in sorted(moves.items()):
        bounded(original); bounded(destination)
        assert original.is_file() and not destination.exists()
        original.rename(destination)
    for path, body in updated.items():
        bounded(path)
        if path in frozen:
            continue
        if not path.exists() or path.read_text(encoding='utf-8') != body:
            path.write_text(body, encoding='utf-8')
    for path, expected in frozen_hashes.items():
        assert digest(path) == expected, relative(path)
    for path, expected in evidence_hashes.items():
        assert digest(path) == expected, relative(path)
    for original, destination in moves.items():
        assert not original.exists() and destination.is_file()
    checked = 0
    for path in [ROOT / 'README.md', *DOCS.rglob('*.md')]:
        for href in LINK.findall(path.read_text(encoding='utf-8')):
            local = target(path, href)
            if local is not None:
                assert local.exists(), (relative(path), href)
                checked += 1
    for entry in manifest['moved_documents']:
        entry['after_sha256'] = digest(ROOT / entry['current'])
    manifest.update(status='complete', verification=dict(local_links=checked, frozen_files_unchanged=len(frozen_hashes),
                    experiment_sources_and_result_json_unchanged=len(evidence_hashes), bare_move_negative_control=len(negative)))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(**plan, verification=manifest['verification'], status='complete'), indent=2))


if __name__ == '__main__':
    main()

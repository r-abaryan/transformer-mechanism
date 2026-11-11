"""Group existing guides/reports by subject without altering frozen evidence."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
TOPICS = DOCS / 'topics'
LINK = re.compile(r'\]\(([^)]+)\)')
SUBJECTS = {
    'ffn': ('Feed-forward networks', 'Activations, gates, width and branch contributions.'),
    'attention': ('Attention', 'Q/K/V, heads, routing capacity, kernels and K/V caching.'),
    'normalization': ('Normalization and residuals', 'Gradient flow, placement, LayerNorm and RMSNorm.'),
    'positions': ('Positional representations', 'Offsets, layouts, sinusoidal encodings and RoPE.'),
    'retrieval': ('Retrieval mechanisms', 'Binding, counterfactuals, interventions and retrieval failures.'),
    'training': ('Training and transfer', 'Optimization, depth, exposure and held-out generalization.'),
    'benchmarks': ('Benchmark comparisons', 'External word prediction and the separate majority task.'),
    'foundations': ('Transformer foundations', 'Architecture, interpretability and numerical verification.'),
}
FILES = {
    'ffn': 'ffn-activation-guide ffn-investigation ffn-width-guide swiglu-guide ffn-branch-results ffn-normalization-results ffn-width-results gelu-comparison-results swiglu-results',
    'attention': 'mqa-training-guide attention-competition-results attention-cost-results cache-precision-results causal-width-results gpu-attention-results head-ablation-results head-row-results head-token-results head-width-results kv-cache-results mqa-decoding-results mqa-training-results native-attention-results native-backward-results output-capacity-results qkv-results query-key-results shared-kv-results target-distractor-results trained-competition-results block1-key-results block1-source-results donor-slot-results',
    'normalization': 'gradient-flow-results norm-extension-results norm-placement-results rmsnorm-math-results rmsnorm-training-results residual-effects-through-ffn-normalization-and-logits',
    'positions': 'positional-comparison-guide canonical-position-results layout-results position-distractor-results position-routing-results position-training-results positional-comparison-results query-position-effects-on-attention-versus-the-residual-path shift-results',
    'retrieval': 'trained-recall-walkthrough activation-results associative-recall-results direct-pair-results donor-controls-results error-strata-results first-token-interventions first-token-results first-value-results joint-routing-confirmation-results key-content-results pair-confirmation-results skip-confirmation-results strict-retrieval-results subset-confirmation-results value-counterfactual-results',
    'training': 'query-exposure-guide depth-comparison-results depth-confirmation-results final-length-results length-exposure-results optimizer-comparison-results positive-controls-results query-exposure-results query-switch-results repeated-key-results reserved-cycle-results',
    'benchmarks': 'wikitext2-comparison-guide majority-transfer-results wikitext2-results',
    'foundations': 'architecture interpretability numerical-walkthrough test-clarifications decoder-verification results',
}
CATEGORY = {}
for folder, names in FILES.items():
    for name in names.split():
        assert name not in CATEGORY, name
        CATEGORY[name] = folder


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def href_target(path, href):
    if href.startswith(('https://', 'http://', 'mailto:', '#')):
        return None
    return (path.parent / href.split('#')[0]).resolve()


def bounded(path):
    assert path.resolve().is_relative_to(ROOT), str(path)
    return path


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    original = json.loads((DOCS / 'organization.json').read_text())
    assert original['status'] == 'complete'
    frozen = {ROOT / name: h for name, h in original['frozen_files'].items()}
    for path, expected in frozen.items():
        assert sha(path) == expected, relative(path)
    bodies = {p.resolve(): p.read_text(encoding='utf-8') for p in [ROOT / 'README.md', *DOCS.rglob('*.md')]}
    before = {p: sha(p) for p in bodies}
    evidence = [DOCS / 'organization.json', *list((ROOT / 'experiments').rglob('*.py')),
                *list((ROOT / 'results').rglob('*.json'))]
    evidence_hashes = {p: sha(p) for p in evidence}
    catalog = {folder: {'guides': [], 'results': []} for folder in SUBJECTS}
    # The existing type indexes provide complete, explicit document membership.
    for kind in ('guides', 'results'):
        index = DOCS / kind / 'README.md'
        for line in bodies[index].splitlines():
            match = re.fullmatch(r'- \[(.+)\]\(([^)]+)\)', line)
            if match is None:
                continue
            title, href = match.groups()
            path = href_target(index, href)
            assert path is not None and path.exists()
            assert path.stem in CATEGORY, path.name
            folder = CATEGORY[path.stem]
            catalog[folder][kind].append((title, path))
    moves = {}
    for folder, groups in catalog.items():
        for title, path in groups['guides'] + groups['results']:
            if path.parent in (DOCS / 'guides', DOCS / 'results'):
                destination = bounded(TOPICS / folder / path.name)
                assert not destination.exists() and path not in frozen
                moves[path] = destination
    assert moves and len(moves) == 78

    def future_exists(path):
        return path not in moves and (path in moves.values() or path.exists())

    negative = []
    for path, body in bodies.items():
        for href in LINK.findall(body):
            target = href_target(path, href)
            if target is None:
                continue
            assert target.exists(), (relative(path), href)
            if not future_exists(href_target(moves.get(path, path), href)):
                negative.append((relative(path), href))
    assert negative

    def rewrite(body, original_path, new_path):
        def change(match):
            href = match.group(1)
            target = href_target(original_path, href)
            if target is None:
                return match.group(0)
            target = moves.get(target, target)
            suffix = '#' + href.split('#', 1)[1] if '#' in href else ''
            return '](' + os.path.relpath(target, new_path.parent).replace('\\', '/') + suffix + ')'
        body = LINK.sub(change, body)
        for old, new in moves.items():
            body = body.replace(relative(old), relative(new))
        return body

    updated = {moves.get(p, p): body if p in frozen else rewrite(body, p, moves.get(p, p))
               for p, body in bodies.items()}
    for path, body in updated.items():
        for href in LINK.findall(body):
            target = href_target(path, href)
            if target is not None:
                assert future_exists(target), (relative(path), href)
    plan = dict(moves=len(moves), sections={folder: sum(len(items) for items in groups.values()) for folder, groups in catalog.items()},
                bare_move_broken_links=len(negative), frozen_files=len(frozen))
    if '--apply' not in sys.argv:
        print(json.dumps(plan, indent=2))
        return
    assert not TOPICS.exists() and not (DOCS / 'topic-organization.json').exists()
    for path, expected in before.items():
        assert sha(path) == expected, relative(path)
    for folder in SUBJECTS:
        bounded(TOPICS / folder).mkdir(parents=True)
    updated[TOPICS / 'README.md'] = '# Research by subject\n\n| Section | Includes |\n|---|---|\n' + '\n'.join(
        '| [' + title + '](' + folder + '/README.md) | ' + description + ' |'
        for folder, (title, description) in SUBJECTS.items()) + '\n\n[Documentation index](../README.md). Each document has one primary section; cross-topic studies remain linked from related reports. Frozen protocols retain their original paths and are listed in the [protocol index](../protocols/README.md).\n'
    for folder, groups in catalog.items():
        index = TOPICS / folder / 'README.md'
        title, description = SUBJECTS[folder]
        body = '# ' + title + '\n\n' + description + '\n\n'
        for kind, heading in (('guides', 'Learn'), ('results', 'Findings')):
            if not groups[kind]:
                continue
            body += '## ' + heading + '\n\n'
            for label, path in sorted(groups[kind], key=lambda entry: entry[1].name):
                destination = moves.get(path, path)
                body += '- [' + label + '](' + os.path.relpath(destination, index.parent).replace('\\', '/') + ')\n'
            body += '\n'
        body += '[All subjects](../README.md) · [Frozen protocols](../../protocols/README.md)\n'
        updated[index] = body
    updated[DOCS / 'README.md'] = '# Documentation\n\nStart with the [architecture assessment](assessments/architecture-assessment.md) for current conclusions and limits.\n\n## Browse by subject\n\n| Section | Includes |\n|---|---|\n' + '\n'.join(
        '| [' + title + '](topics/' + folder + '/README.md) | ' + description + ' |'
        for folder, (title, description) in SUBJECTS.items()) + '\n\n## Other indexes\n\n[Guides](guides/README.md) · [Study reports](results/README.md) · [Primary sources and claims](reference/README.md) · [Protocols](protocols/README.md) · [Assessments](assessments/README.md) · [Decisions](decisions/README.md) · [Figures](figures/)\n\n[Research plan](research-plan.md) records status and stopping rules. [Full reading order](reading-order.md) preserves the earlier sequence.\n\nExisting protocols retain their bytes and fixed paths for provenance checks. A few guides/reports linked by those protocols also retain their original paths; the subject indexes include them. Future documents belong in their subject folders; future frozen protocols belong in `protocols/`.\n\nSubject folders contain Markdown explanations and reports. Raw measurements and checkpoints remain in the workspace `results/` directory.\n'
    readme = ROOT / 'README.md'
    updated[readme] = updated[readme].replace('Use the [documentation index](docs/README.md) for guides, study reports, protocols and sources.', 'Browse [FFN](docs/topics/ffn/README.md), [attention](docs/topics/attention/README.md), or the [full subject index](docs/topics/README.md). The [documentation index](docs/README.md) also links protocols and primary sources.')
    updated[readme] = updated[readme].replace('|   |-- guides/              # explanations and walkthroughs\n|   |-- results/             # study reports, methods and limits', '|   |-- topics/              # FFN, attention, normalization and other subjects\n|   |-- guides/              # index of learning documents\n|   |-- results/             # index of study reports')
    for kind in ('guides', 'results'):
        index = DOCS / kind / 'README.md'
        updated[index] = updated[index].replace('Browse by subject. See the [documentation index](../README.md) for the full structure.', 'This index lists learning documents across the [subject folders](../topics/README.md).').replace('Individual findings, methods and limits. Raw JSON, score tensors and checkpoints remain in the workspace `results/` directory.', 'This index lists reports across the [subject folders](../topics/README.md). Raw JSON, score tensors and checkpoints remain in the workspace `results/` directory.')
    manifest = dict(status='planned', date='2026-10-01', previous_manifest_sha256=sha(DOCS / 'organization.json'), plan=plan,
                    moved_documents=[dict(original=relative(p), current=relative(q), before_sha256=before[p]) for p, q in sorted(moves.items())],
                    frozen_files={relative(p): h for p, h in sorted(frozen.items())})
    manifest_path = DOCS / 'topic-organization.json'
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
    for path, expected in frozen.items():
        assert sha(path) == expected, relative(path)
    for path, expected in evidence_hashes.items():
        assert sha(path) == expected, relative(path)
    checked = 0
    for path in [ROOT / 'README.md', *DOCS.rglob('*.md')]:
        for href in LINK.findall(path.read_text(encoding='utf-8')):
            target = href_target(path, href)
            if target is not None:
                assert target.exists(), (relative(path), href)
                checked += 1
    for entry in manifest['moved_documents']:
        assert not (ROOT / entry['original']).exists()
        entry['after_sha256'] = sha(ROOT / entry['current'])
    manifest.update(status='complete', verification=dict(local_links=checked, frozen_files_unchanged=len(frozen),
                    experiment_sources_result_json_and_previous_manifest_unchanged=len(evidence_hashes)))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(**plan, verification=manifest['verification'], status='complete'), indent=2))


if __name__ == '__main__':
    main()

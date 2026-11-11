"""Read-only evidence/table/link checks; writes a separate verification record."""
import csv
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

report = json.loads((OUT / 'report.json').read_text())
audit = json.loads((OUT / 'audit.json').read_text())
lock = json.loads((OUT / 'lock.json').read_text())
assert audit['passed'] and audit['report_sha256'] == digest(OUT / 'report.json')
for name, expected in lock['files'].items():
    assert digest(ROOT / name) == expected, name
assert report['lock_sha256'] == digest(OUT / 'lock.json')
assert report['scores_sha256'] == digest(OUT / 'scores.pt')
for field, name in [('pools_sha256', 'pools.pt'), ('source_scores_sha256', 'source_scores.pt')]:
    assert lock[field] == digest(OUT / name)

cells = {(c['seed'], c['routing'], c['variant'], c['pool']): c for c in report['cells']}
assert len(cells) == len(report['cells']) == 288
seeds = [3473, 3479, 3481]
routes = ['lbfgs', 'negative_gradient']
variants = ['base', 'scale', 'shape', 'full']
splits = ['train', 'validation', 'test']
loads = [2, 8, 16, 32]
expected = {(s, r, v, f'{p}_{m}') for s, r, v, p, m in itertools.product(seeds, routes, variants, splits, loads)}
assert set(cells) == expected
assert sum(c['metrics']['groups']['all']['count'] for c in cells.values()) == report['evaluation_targets'] == 1870848
rows = list(csv.DictReader((OUT / 'evaluation.csv').open(newline='', encoding='utf-8')))
assert len(rows) == 864
seen = set()
for row in rows:
    key = (int(row['seed']), row['routing'], row['variant'], row['pool'])
    group = row['group']
    assert (key, group) not in seen
    seen.add((key, group))
    metric = cells[key]['metrics']['groups'][group]
    maps = cells[key]['metrics']['complete_maps']
    assert int(row['count']) == metric['count']
    for field in ['accuracy', 'ce']:
        actual = None if row[field] == '' else float(row[field])
        assert actual == metric[field], (key, group, field)
    for field, source in [('maps_count', 'count'), ('maps_correct', 'correct'), ('maps_accuracy', 'accuracy')]:
        assert float(row[field]) == maps[source]
assert seen == {(key, g) for key in cells for g in ['all', 'unique', 'repeat']}

contrasts = {(c['seed'], c['routing'], c['pool']): c for c in report['summary']['contrasts']}
assert len(contrasts) == 72
primary = {}
ratios = []
for s, r, p, m in itertools.product(seeds, routes, splits, loads):
    pool = f'{p}_{m}'
    a, b, c, d = [cells[s, r, v, pool]['metrics']['groups']['all']['ce'] for v in variants]
    total = d-a
    sf, sl, qf, ql = b-a, d-c, c-a, d-b
    wanted = dict(base=a, scale=b, shape=c, full=d, total_penalty=total,
                  scale_at_base_shape=sf, scale_at_lbfgs_shape=sl,
                  shape_at_base_scale=qf, shape_at_lbfgs_scale=ql,
                  interaction=d-b-c+a,
                  scale_first_ratio=sf/total if abs(total)>1e-8 else None,
                  scale_last_ratio=sl/total if abs(total)>1e-8 else None,
                  scale_dominates=total>1e-8 and all(x>abs(y) for x in [sf, sl] for y in [qf, ql]))
    saved = contrasts[s, r, pool]
    for field, value in wanted.items():
        assert saved[field] == value, (s, r, pool, field)
    if p in ['validation', 'test'] and m == 32:
        primary[s, r, pool] = saved
        ratios += [sf/total, sl/total]
assert len(primary) == 12 and all(c['scale_dominates'] for c in primary.values())
assert {(c['seed'], c['routing'], c['pool']): c for c in report['summary']['primary']} == primary
assert report['summary']['all_primary_scale_dominates'] is True

path = ROOT / 'docs/topics/attention/decoder-scale-shape-results.md'
text = path.read_text(encoding='utf-8')
tables = [b for b in text.split('\n\n') if b.startswith('| Seed |')]
assert len(tables) == 4
parsed = [[list(map(str.strip, line.strip('|').split('|'))) for line in table.splitlines()[2:]] for table in tables]
assert parsed[0] == [[str(p['seed']), f"{p['scale']:.4f}", f"{100*p['relative_residual']:.2f}%"] for p in report['projections']]
for table, split in zip(parsed[1:3], ['test', 'validation']):
    expected = []
    for seed, routing in itertools.product(seeds, routes):
        ce = [cells[seed, routing, v, f'{split}_32']['metrics']['groups']['all']['ce'] for v in variants]
        expected.append([str(seed), 'L' if routing == 'lbfgs' else 'N'] + [f'{x:.4f}' for x in ce])
    assert table == expected
train = []
for seed in seeds:
    cs = {v: [cells[seed, 'lbfgs', v, f'train_{m}']['metrics'] for m in loads] for v in ['shape', 'full']}
    ce = [math.fsum(c['groups']['all']['ce'] for c in cs[v])/4 for v in ['shape', 'full']]
    errors = [sum(round(c['groups']['all']['count']*(1-c['groups']['all']['accuracy'])) for c in cs[v]) for v in ['shape', 'full']]
    maps = [cs[v][-1]['complete_maps']['correct'] for v in ['shape', 'full']]
    train.append([str(seed), *[f'{x:.7f}' for x in ce], f'{errors[0]} / {errors[1]}', f'{maps[0]} / {maps[1]}'])
assert parsed[3] == train
assert f'{100*min(ratios):.2f}% to {100*max(ratios):.2f}%' in text
assert all(report[k] == 0 for k in ['optimizer_calls', 'gradient_calls', 'solver_calls', 'producer_model_forwards'])
assert audit['protected_files'] == len(lock['files']) == 52
assert f"{report['seconds']:.2f} seconds" in text
for seed in seeds:
    for pool in [f'{p}_{m}' for p, m in itertools.product(splits, loads)]:
        for a, b in [('base', 'scale'), ('shape', 'full')]:
            x, y = [cells[seed, 'lbfgs', v, pool]['metrics'] for v in [a,b]]
            assert x['complete_maps'] == y['complete_maps']
            for group in ['all', 'unique', 'repeat']:
                assert x['groups'][group]['accuracy'] == y['groups'][group]['accuracy']

docs = [
    'docs/topics/attention/decoder-scale-shape-guide.md',
    'docs/topics/attention/decoder-scale-shape-results.md',
    'docs/protocols/decoder-scale-shape.md',
    'docs/assessments/research-checkpoint.md',
    'docs/assessments/learning-mechanism-agenda.md',
    'docs/assessments/retention-learning-strategy.md',
    'docs/assessments/failure-register.md',
    'docs/assessments/research-history.md',
    'docs/assessments/README.md',
    'docs/topics/attention/README.md',
    'docs/guides/README.md', 'docs/results/README.md',
    'docs/protocols/README.md', 'tests/README.md',
    'docs/reference/sources.md', 'README.md',
]
link_count = 0
for name in docs:
    doc = ROOT / name
    for target in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
        target = target.strip('<>')
        if re.match(r'^[a-zA-Z]+:', target) or target.startswith('#'):
            continue
        local = unquote(target.split('#')[0])
        assert (doc.parent/local).resolve().exists(), (name, target)
        link_count += 1
verification = dict(passed=True, csv_rows=len(rows), score_cells=len(cells),
    contrasts_checked=len(contrasts), primary_contrasts=len(primary), measured_table_rows=sum(map(len, parsed)),
    protected_files=len(lock['files']), local_links=link_count,
    additional_training=0, additional_model_evaluations=0,
    scope='CSV/report concordance, independently reconstructed scalar effects, displayed tables, local file targets and frozen evidence hashes; no new forwards or gradients',
    files={name:digest(ROOT/name) for name in docs},
    report_sha256=digest(OUT/'report.json'), audit_sha256=digest(OUT/'audit.json'),
    csv_sha256=digest(OUT/'evaluation.csv'), verifier_sha256=digest(Path(__file__)))
destination = OUT / 'document_verification.json'
assert not destination.exists(), 'Preserve verification'
destination.write_text(json.dumps(verification, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in verification.items() if k != 'files'}))

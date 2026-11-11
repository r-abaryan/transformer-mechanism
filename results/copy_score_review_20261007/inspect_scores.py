"""One saved-score review. No model forward, intervention, or training."""
import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
SOURCE = ROOT / 'results/copy_source_visibility'
TOL = 1e-10


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def category(gaps):
    if any(abs(g) <= TOL for g in gaps):
        return 'near_tie'
    return ('neither_outranked', 'one_outranked', 'both_outranked')[sum(g < -TOL for g in gaps)]


def main():
    assert not (OUT/'plan.json').exists() and not (OUT/'report.json').exists()
    assert category([2., 1.]) == 'neither_outranked'
    assert category([-2., 1.]) == 'one_outranked'
    assert category([-2., -1.]) == 'both_outranked'
    assert category([0., -1.]) == 'near_tie'
    assert category([2., 1.]) != category([-2., -1.])
    paths = [SOURCE/p for p in ('lock.json', 'report.json', 'audit.json', 'evidence.pt')]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    write(OUT/'plan.json', dict(question='Is source position zero outranked in either head on native first-token failures?',
          scope='All eight frozen pools, all three seeds, native E1M1 only; all/distinct/repeated; correct and wrong cohorts',
          gap='score(position 0) minus maximum score at any other source position, per head',
          categories=['neither_outranked', 'one_outranked', 'both_outranked', 'near_tie'],
          tolerance=TOL, interpretation='Association only; attention rank is not output rank. No head attribution or sharpening intervention.',
          stop='One reduction; no selected-example reruns or temperature search', input_hashes=hashes,
          script_sha256=sha(Path(__file__))))
    report = json.loads((SOURCE/'report.json').read_text())
    audit = json.loads((SOURCE/'audit.json').read_text())
    assert audit['passed'] and audit['report_sha256'] == sha(SOURCE/'report.json')
    assert report['evidence_sha256'] == sha(SOURCE/'evidence.pt')
    lock = json.loads((SOURCE/'lock.json').read_text())
    assert report['lock_sha256'] == sha(SOURCE/'lock.json')
    evidence = torch.load(SOURCE/'evidence.pt', weights_only=True)
    records = []
    max_error = 0.
    total = 0
    for run in lock['runs']:
        seed = run['seed']
        for pool, rows in lock['pools'].items():
            v = evidence[f'{seed}/{pool}/E1M1']
            scores = v['scores'][:, :, 0]
            assert scores.shape[1] == 2
            reconstructed = (v['q'] @ v['k'].transpose(-1, -2)/math.sqrt(v['q'].shape[-1]))[:, :, 0]
            error = float((reconstructed-scores).abs().max())
            assert error <= 1e-9
            max_error = max(max_error, error)
            gaps = scores[:, :, 0]-scores[:, :, 1:].max(-1).values
            winners = scores.argmax(-1)
            good = v['logits'].argmax(-1) == torch.tensor([r[0] for r in rows])
            cats = []
            for i, row in enumerate(rows):
                scalar_gaps = [s[0]-max(s[1:]) for s in scores[i].tolist()]
                assert max(abs(a-b) for a,b in zip(scalar_gaps,gaps[i].tolist())) <= 1e-9
                cat = category(scalar_gaps)
                vector_cat = 'near_tie' if bool((gaps[i].abs() <= TOL).any()) else ('neither_outranked','one_outranked','both_outranked')[int((gaps[i] < -TOL).sum())]
                assert cat == vector_cat
                cats.append(cat)
                for head in range(2):
                    s = scores[i,head].tolist(); mx=max(s)
                    weights = [math.exp(x-mx) for x in s];den=math.fsum(weights)
                    error=max(abs(w/den-float(v['weights'][i,head,0,j])) for j,w in enumerate(weights))
                    assert error <= 1e-9
                    max_error=max(max_error,error)
            total += len(rows)
            for group in ('all','distinct','repeated'):
                for correct in (True,False):
                    ids=[i for i,r in enumerate(rows) if bool(good[i]) == correct and (group=='all' or (len(set(r))==len(r)) == (group=='distinct'))]
                    counts={c:sum(cats[i]==c for i in ids) for c in ('neither_outranked','one_outranked','both_outranked','near_tie')}
                    assert sum(counts.values()) == len(ids)
                    records.append(dict(seed=seed,pool=pool,group=group,correct=correct,count=len(ids),categories=counts,
                       added_winners=[sum(6 <= int(winners[i,h]) < len(rows[i]) for i in ids) for h in range(2)],
                       mean_target_mass=float(v['weights'][ids,:,0,0].mean()) if ids else None))
    assert total == 2184 and len(records) == 144
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    write(OUT/'report.json',dict(passed=True,records=records,unique_seed_example_evaluations=total,
          forward_calls=0,training_updates=0,max_arithmetic_error=max_error,input_hashes=hashes,
          plan_sha256=sha(OUT/'plan.json'),script_sha256=sha(Path(__file__))))
    for r in records:
        if r['pool'] in ('test_8','test_10') and r['group']=='all': print(json.dumps(r))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()

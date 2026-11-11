"""Reload both budgets through explicit forward and check corpus/streams."""
import collections
import itertools
import json
import torch
from norm_placement import ROOT, model_for, explicit
from extend_norm_placement import excluded_maps, identity
from position_distractors import digest
from position_training import metrics


def main():
    torch.set_num_threads(1)
    report=json.loads((ROOT/'results/norm_extension.json').read_text())
    assert report['phase']=='complete' and len(report['runs'])==20
    assert {(r['seed'],r['placement'],r['schedule']) for r in report['runs']} == set(itertools.product((601,631,659,683,719),('pre','post'),('constant','warmup')))
    old=json.loads((ROOT/'results/norm_placement.json').read_text())
    assert digest(ROOT/'results/norm_placement.json')==report['provenance']['original_sha256']
    assert digest(ROOT/'experiments/extend_norm_placement.py')==report['provenance']['source_sha256']
    assert digest(ROOT/'docs/norm-extension-protocol.md')==report['provenance']['protocol_sha256']
    inp=ROOT/'results/norm_extension_inputs.json'; assert digest(inp)==report['input_sha256']
    rows=json.loads(inp.read_text()); groups=collections.defaultdict(list)
    for x,y in rows:
        m=identity(x); assert dict(m)[x[-1]]==y and x[-2]==18
        groups[m].append((tuple(x),y))
    assert len(groups)==400 and not set(groups)&excluded_maps()
    for m,variants in groups.items():
        expected={(tuple(t for pair in order for t in pair)+(18,q),v) for order in itertools.permutations(m) for q,v in m}
        assert len(variants)==18 and set(variants)==expected
    assert set(collections.Counter(y for x,y in rows).values())=={1200}
    assert set(collections.Counter(x[:6:2].index(x[-1]) for x,y in rows).values())=={2400}
    dp=ROOT/'results/norm_extension_scores.json'; assert digest(dp)==report['scores_sha256']
    details=json.loads(dp.read_text()); x=torch.tensor([x for x,y in rows]); y=torch.tensor([y for x,y in rows])
    checks=[]
    for run in report['runs']:
        assert run['replay_exact']
        peers=[r for r in report['runs'] if r['seed']==run['seed'] and r['status']=='trained']
        assert len({r['minibatch_sha256'] for r in peers})==1 and len({r['offset_sha256'] for r in peers})==1
        original=next(r for r in old['runs'] if all(r[n]==run[n] for n in ('seed','placement','schedule')))
        assert run['curve'][0]==original['curve'][-1]
        if run['status']=='trained':
            assert [v['step'] for v in run['curve']]==list(range(3000,6001,500))
        for step in (3000,6000):
            if step==6000 and run['status']!='trained': continue
            cp=ROOT/(original['checkpoint'] if step==3000 else run['checkpoint'])
            assert digest(cp)==(original['checkpoint_sha256'] if step==3000 else run['checkpoint_sha256'])
            state=torch.load(cp,weights_only=True)
            if step==6000:
                assert state['step']==6000
                assert all(v['step'].item()==6000 for v in state['optimizer']['state'].values())
            model=model_for(run['placement']).eval(); model.load_state_dict(state if step==3000 else state['model'])
            with torch.no_grad(): s=torch.cat([explicit(model,c) for c in x.split(256)])
            saved=next(d for d in details if d['step']==step and all(d[n]==run[n] for n in ('seed','placement','schedule')))
            err=(s-torch.tensor(saved['scores'],dtype=torch.float64)).abs().max().item()
            assert err<1e-10 and metrics(s,y)==run[f'test_{step}']
            checks.append(dict(seed=run['seed'],placement=run['placement'],schedule=run['schedule'],step=step,max_logit_error=err))
    result=dict(report_sha256=digest(ROOT/'results/norm_extension.json'),checks=checks,mappings=400,examples=len(rows),passed=True)
    (ROOT/'results/norm_extension_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,evaluations=len(checks),maximum_error=max(c['max_logit_error'] for c in checks))))


if __name__=='__main__': main()

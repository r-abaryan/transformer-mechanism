"""Audit factorial training provenance and reload all finite final models."""
import itertools
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from norm_placement import model_for,explicit
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    path=ROOT/'results/norm_placement.json'; r=json.loads(path.read_text()); assert r['phase']=='complete' and len(r['runs'])==20
    for n,h in r['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    assert digest(ROOT/'docs/norm-placement-protocol.md')==r['protocol_sha256']
    if 'resume' in r:
        assert digest(ROOT/'experiments/resume_norm_placement.py')==r['resume']['runner_sha256']
        archive=ROOT/'results/norm_placement_interrupted.json'; assert digest(archive)==r['resume']['archive_sha256']
        previous=json.loads(archive.read_text())
        for old in previous['runs']:
            now=next(v for v in r['runs'] if (v['seed'],v['placement'],v['schedule'])==(old['seed'],old['placement'],old['schedule']))
            if old['status']=='trained': assert old['checkpoint_sha256']==now['checkpoint_sha256']
            if old['status']=='running':
                assert old['curve']==now['curve'][:len(old['curve'])] and old['probes']==now['probes'][:len(old['probes'])]
    sp=ROOT/'results/norm_placement_splits.json'; dp=ROOT/'results/norm_placement_test_scores.json'
    assert digest(sp)==r['split_sha256'] and digest(dp)==r['scores_sha256']
    data=json.loads(sp.read_text()); details=json.loads(dp.read_text())
    def key(ids): return tuple(sorted(zip(ids[:-2:2],ids[1:-2:2])))
    excluded=set()
    for name,h in r['exclusion_hashes'].items():
        p=ROOT/'results'/name; assert digest(p)==h; d=json.loads(p.read_text())
        for rows in (d.values() if isinstance(d,dict) else [d]): excluded.update(key(x) for x,y in rows if len(x)==8)
    assert len(excluded)==6020
    sets={n:{key(x) for x,y in rows} for n,rows in data.items()}
    assert {n:len(s) for n,s in sets.items()}==dict(train=1600,validation=400,test=400)
    combined=set.union(*sets.values()); assert len(combined)==2400 and not combined&excluded
    for rows in data.values():
        assert all(sum(y==v for x,y in rows)==len(rows)//6 for v in range(12,18))
        assert all(sum(x[:6:2].index(x[-1])==i for x,y in rows)==len(rows)//3 for i in range(3))
        for start in range(0,len(rows),18):
            m=key(rows[start][0]); expected={(tuple(t for pair in order for t in pair)+(18,q),v) for order in itertools.permutations(m) for q,v in m}
            assert {(tuple(x),y) for x,y in rows[start:start+18]}==expected
    for seed in (601,631,659,683,719):
        a=[run for run in r['runs'] if run['seed']==seed]
        assert {(v['placement'],v['schedule']) for v in a}==set(itertools.product(('pre','post'),('constant','warmup')))
        assert len({v['initial_sha256'] for v in a})==1
        finished=[v for v in a if v['status']=='trained']
        for k in ('minibatch_sha256','offset_sha256'): assert len({v[k] for v in finished})<=1
        for v in finished:
            assert v['steps']==3000
            expected=sum(.001*min(t/300,1) if v['schedule']=='warmup' else .001 for t in range(1,3001))
            assert abs(v['learning_rate_sum']-expected)<1e-12
            assert [c['step'] for c in v['curve']]==list(range(0,3001,500))
            assert v['first_observed_validation_99']==next((c['step'] for c in v['curve'] if c['accuracy']>=.99),None)
    x=torch.tensor([x for x,y in data['test']]); y=torch.tensor([y for x,y in data['test']]); error=0.; logit_error=0.; count=0
    with torch.no_grad():
        for run in r['runs']:
            if run['status']!='trained': continue
            d=next(v for v in details if (v['seed'],v['placement'],v['schedule'])==(run['seed'],run['placement'],run['schedule']))
            s=torch.tensor(d['scores'],dtype=torch.float64); assert s.shape==(7200,19) and torch.isfinite(s).all()
            cp=ROOT/run['checkpoint']; assert digest(cp)==run['checkpoint_sha256']
            model=model_for(run['placement']).eval(); model.load_state_dict(torch.load(cp,weights_only=True))
            reloaded=torch.cat([explicit(model,c) for c in x.split(256)]); logit_error=max(logit_error,(s-reloaded).abs().max().item())
            correct=reloaded.argmax(-1)==y
            measured=dict(accuracy=correct.double().mean().item(),loss=F.cross_entropy(reloaded,y).item(),every_order_query_correct=correct.reshape(400,18).all(1).double().mean().item())
            for k,v in measured.items(): error=max(error,abs(v-run['test'][k]))
            count+=1; print(run['seed'],run['placement'],run['schedule'],'verified',flush=True)
    assert error<1e-10 and logit_error<1e-10 and len(details)==count
    for name,passed in r['condition_all_99'].items():
        p,w=name.split('_'); selected=[v for v in r['runs'] if v['placement']==p and v['schedule']==w]
        assert len(selected)==5 and passed==all(v['status']=='trained' and v['test']['accuracy']>=.99 for v in selected)
    result=dict(passed=True,checkpoints=count,maximum_metric_error=error,maximum_logit_error=logit_error,report_sha256=digest(path),audit_source_sha256=digest(Path(__file__)),scope='Mapping and paired-recipe checks; final checkpoint reload with explicit forward, shared PyTorch.')
    out=ROOT/'results/norm_placement_audit.json'; assert not out.exists(); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()

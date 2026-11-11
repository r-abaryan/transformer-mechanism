"""Separate labels, metrics, split checks and previously independent forward."""
import hashlib
import itertools
import json
from pathlib import Path

import torch
from audit_mqa_training import independent
from majority_transfer import ROOT, SEEDS, LENGTHS, model_new, hash_state, tensors, provenance
from position_distractors import digest


def variants(x):
    complement=tuple(int(not b) for b in x)
    return sorted(set((tuple(x), tuple(reversed(x)), complement, tuple(reversed(complement)))))


def label(x): return int(sum(1 if b else -1 for b in x)>0)


def audit_inputs(p):
    expected={'train_9':80,'validation_9':20,'test_9':36,'train_11':350,'validation_11':50,'test_11':128,'test_15':200,'test_21':200}
    assert set(p['datasets'])==set(p['families'])==set(expected); seen=set(); counts={}
    for key,count in expected.items():
        n=int(key.split('_')[1]); families=p['families'][key]; rows=p['datasets'][key]
        assert len(families)==count and len({tuple(x) for x in families})==count
        expanded=[]
        for r in families:
            assert len(r)==n and set(r)<={0,1} and tuple(r)==variants(r)[0]
            for x in variants(r):
                assert x not in seen; seen.add(x); expanded.append((list(x)+[2],label(x)))
        assert rows==[[x,y] for x,y in expanded]
        assert sum(y for x,y in rows)*2==len(rows)
        assert any(abs(sum(1 if b else -1 for b in x[:-1]))==1 for x,y in rows)
        counts[key]=dict(families=count,rows=len(rows),boundary=sum(abs(2*sum(x[:-1])-n)==1 for x,y in rows))
    for n in (9,11):
        allrows={tuple(x[:-1]) for key,rows in p['datasets'].items() if key.endswith(f'_{n}') for x,y in rows}
        assert allrows==set(itertools.product((0,1),repeat=n))
    return counts


def metrics(scores, rows, representatives):
    predictions=scores.argmax(-1).tolist(); bits=[x[:-1] for x,y in rows]; truth=[label(x) for x in bits]
    correct=[int(a==b) for a,b in zip(predictions,truth)]; margin=[abs(sum(2*b-1 for b in x)) for x in bits]
    strata={str(m):dict(count=margin.count(m),accuracy=sum(c for c,v in zip(correct,margin) if v==m)/margin.count(m)) for m in sorted(set(margin))}
    lengths=[len(variants(x)) for x in representatives]; cursor=0; families_correct=0
    for size in lengths: families_correct+=int(all(correct[cursor:cursor+size])); cursor+=size
    assert cursor==len(rows)
    losses=scores.logsumexp(-1)-scores[torch.arange(len(rows)),torch.tensor(truth)]
    return dict(count=len(rows),correct=sum(correct),accuracy=sum(correct)/len(rows),loss=losses.mean().item(),
                boundary_accuracy=strata['1']['accuracy'], class_accuracy={str(c):sum(a for a,b in zip(correct,truth) if b==c)/truth.count(c) for c in (0,1)},
                family_count=len(lengths),family_all_correct=families_correct/len(lengths),strata=strata,
                baselines=dict(constant=truth.count(0)/len(rows),first=sum(x[0]==y for x,y in zip(bits,truth))/len(rows),
                               last=sum(x[-1]==y for x,y in zip(bits,truth))/len(rows),exact_count=1.0))


def compare(a,b):
    assert set(a)==set(b)
    for key,value in a.items():
        if isinstance(value,dict): compare(value,b[key])
        else: assert abs(value-b[key])<=1e-10,(key,value,b[key])


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    rp=ROOT/'results/majority_transfer.json'; report=json.loads(rp.read_text()); assert report['phase']=='complete'
    assert report['provenance']==provenance(); ip=ROOT/'results/majority_transfer_data.json'
    assert digest(ip)==report['input_sha256']; payload=json.loads(ip.read_text()); inputs=audit_inputs(payload)
    assert {r['seed'] for r in report['runs']}==set(SEEDS)
    dp=ROOT/'results/majority_transfer_scores.pt'; assert digest(dp)==report['scores_sha256']; saved=torch.load(dp,weights_only=True)
    errors=[]; predictions_checked=0; checks={}; finished=True
    for r in report['runs']:
        torch.manual_seed(r['seed']); model=model_new(); assert hash_state(model.state_dict())==r['initial_sha256']
        assert sum(p.numel() for p in model.parameters())==r['parameters']==4016
        cp=ROOT/r['checkpoint']; assert digest(cp)==r['checkpoint_sha256']; state=torch.load(cp,weights_only=True)
        assert state['provenance']==report['provenance']; assert state['run']['step']==r['step']
        for n,offset in ((9,1000),(11,2000)):
            g=torch.Generator().manual_seed(r['seed']+offset); h=hashlib.sha256()
            for _ in range(r['step']): h.update(json.dumps(torch.randint(len(payload['datasets'][f'train_{n}']),(64,),generator=g).tolist()).encode())
            assert h.hexdigest()==r[f'stream{n}_sha256']; assert torch.equal(g.get_state(),state[f'g{n}'])
        if r['status']!='trained': finished=False; continue
        assert r['step']==6000 and r['tokens']==4224000 and r['attention_entries']==46848000
        assert all(s['step'].item()==6000 for s in state['optimizer']['state'].values())
        model.load_state_dict(state['model']); model.eval(); detail=next(s for s in saved if s['seed']==r['seed'])
        for n in LENGTHS:
            key=f'test_{n}'; rows=payload['datasets'][key]; x,y=tensors(rows)
            with torch.no_grad(): logits=torch.cat([independent(model,c)[:,-1] for c in x.split(256)])
            error=(logits-detail['scores'][str(n)]).abs().max().item(); assert error<=1e-10
            assert torch.equal(logits.argmax(-1),detail['scores'][str(n)].argmax(-1)); errors.append(error)
            actual=metrics(logits,rows,payload['families'][key]); compare(actual,r['test'][str(n)]); predictions_checked+=len(rows)
    checks['completed']=finished and len(report['runs'])==5
    for n in LENGTHS:
        values=[r.get('test',{}).get(str(n),{}) for r in report['runs']]
        checks[f'{n}_overall']=all(v.get('accuracy',0)>=(.99 if n in (9,11) else .90) for v in values)
        if n>=15: checks[f'{n}_boundary']=all(v.get('boundary_accuracy',0)>=.90 for v in values)
    assert report['decision']==dict(pass_transfer=all(checks.values()),checks=checks)
    verification=json.loads((ROOT/'results/majority_transfer_verification.json').read_text())
    assert verification['passed'] and verification['provenance']==report['provenance']
    result=dict(passed=True,evaluations=len(errors),predictions_checked=predictions_checked,max_logit_error=max(errors,default=0),
                input_checks=inputs,report_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    path=ROOT/'results/majority_transfer_audit.json'; assert not path.exists(); path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()

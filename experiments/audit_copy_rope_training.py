"""Final independent copy equations, decoding, scalar metrics and stream checks."""
import copy
import hashlib
import json
import math
import struct
from pathlib import Path

import torch

from copy_rope_training import ROOT, DIRECTORY, DATA, SEEDS, ARMS, STEPS, decision
from copy_rope_model import make_model, reference
from copy_length_exposure import COUNTS, audit_data
from audit_copy_length_exposure import scalar_metrics as ordinary_metrics
from audit_copy_eos_training import scalar_metrics as oracle_metrics
from mqa_training import hash_state
from position_distractors import digest
from recurrent_retrieval import write


def decode(model, source, arm, oracle=False):
    outputs=[[] for _ in source];finished=[False]*len(source)
    prefix=torch.ones(len(source),1,dtype=torch.long)
    limit=source.shape[1] if oracle else source.shape[1]+3
    for _ in range(limit):
        z=reference(model,source,prefix,arm)[:,-1].tolist()
        candidates=[k for k in range(16) if not oracle or k!=2]
        choices=[max(candidates,key=row.__getitem__) for row in z]
        for i,c in enumerate(choices):
            if not finished[i]:outputs[i].append(c)
            finished[i]=finished[i] or (not oracle and c==2)
        prefix=torch.cat((prefix,torch.tensor(choices)[:,None]),1)
        if all(finished):break
    return outputs


def scalar_metrics(z, rows, generated, oracles):
    a=ordinary_metrics(z,rows,generated);b=oracle_metrics(z,rows,oracles,1.)
    return {g:dict(a[g],**b[g]) for g in a}


def audit_cell(model, arm, rows, expected, stored):
    zs=[];generated=[];oracles=[]
    with torch.no_grad():
        for source in torch.tensor(rows).split(64):
            prefix=torch.cat((torch.ones(len(source),1,dtype=torch.long),source),1)
            zs.append(reference(model,source,prefix,arm))
            generated.extend(decode(model,source,arm))
            oracles.extend(decode(model,source,arm,oracle=True))
    z=torch.cat(zs);error=float((z-stored['logits']).abs().max())
    assert error<=1e-9 and torch.equal(z.argmax(-1),stored['logits'].argmax(-1))
    assert generated==stored['generated'] and oracles==stored['oracle_outputs']
    actual=scalar_metrics(z.tolist(),rows,generated,oracles);metric_error=0.
    assert actual.keys()==expected.keys()
    for g, m in actual.items():
        assert m.keys()==expected[g].keys()
        metric_error=max(metric_error,max(abs(v-expected[g][k]) for k,v in m.items()))
    assert metric_error<=1e-9
    return actual,dict(logit_error=error,metric_error=metric_error,teacher_argmax_changes=0,
        ordinary_sequence_changes=0,oracle_sequence_changes=0,sequences=len(rows),
        teacher_rows=len(rows)*(len(rows[0])+1),oracle_content_tokens=len(rows)*len(rows[0]),
        ordinary_generated_tokens=sum(map(len,generated)))


def own_decision(runs):
    fits={};seen={};pairs=[]
    for arm in ARMS:
        rs=[r for r in runs if r['arm']==arm];assert len(rs)==3
        fits[arm]=all(r['status']=='trained' and min(r['metrics'][f'train_{n}'][g]['exact_accuracy']
            for n in range(3,7) for g in ('distinct','repeated'))>=.99 for r in rs)
        seen[arm]=fits[arm] and min(r['metrics'][f'test_{n}'][g]['exact_accuracy']
            for r in rs for n in range(3,7) for g in ('distinct','repeated'))>=.95
    for seed in SEEDS:
        s=next(r for r in runs if r['seed']==seed and r['arm']=='sinusoidal')
        r=next(r for r in runs if r['seed']==seed and r['arm']=='rope')
        gains={str(n):r['metrics'][f'test_{n}']['all']['exact_accuracy']-s['metrics'][f'test_{n}']['all']['exact_accuracy'] for n in (8,10)}
        worst=min(r['metrics'][f'test_{n}'][g]['exact_accuracy']-s['metrics'][f'test_{n}'][g]['exact_accuracy']
            for n in range(3,7) for g in ('distinct','repeated'))
        pairs.append(dict(seed=seed,long_gains=gains,seen_length_preserved=worst>=-.05))
    transfer={}
    for n in (8,10):
        gs=('distinct','repeated') if n==8 else ('repeated',)
        transfer[str(n)]=seen['rope'] and min(r['metrics'][f'test_{n}'][g]['exact_accuracy']
            for r in runs if r['arm']=='rope' for g in gs)>=.9
    return dict(training_fit=fits,seen_length_reliability=seen,paired=pairs,
        rope_benefit=all(fits.values()) and all(p['seen_length_preserved'] and min(p['long_gains'].values())>=.1 for p in pairs),
        rope_extrapolation=transfer)


def main():
    report=json.loads((DIRECTORY/'report.json').read_text(encoding='utf-8'))
    assert report['status']=='complete' and not (DIRECTORY/'audit.json').exists()
    assert all(digest(ROOT/p)==h for p,h in report['source_dependencies'].items())
    payload=json.loads(DATA.read_text(encoding='utf-8'));assert audit_data(payload)==report['input_checks']
    assert digest(DIRECTORY/'metrics.csv')==report['metrics_sha256']
    pre=json.loads((DIRECTORY/'pretest.json').read_text(encoding='utf-8'))
    assert pre['status']=='all_training_finished_before_test' and len(pre['runs'])==6
    assert all(not any(p.startswith('test_') for p in r['metrics']) for r in pre['runs'])
    assert [r['checkpoint_sha256'] for r in pre['runs']]==[r['checkpoint_sha256'] for r in report['runs']]
    checked=[];cells=[];streams={}
    for run in report['runs']:
        for kind in ('checkpoint','scores'):assert digest(ROOT/run[kind])==run[kind+'_sha256']
        cp=torch.load(ROOT/run['checkpoint'],weights_only=True);saved=torch.load(ROOT/run['scores'],weights_only=True)
        torch.manual_seed(run['seed']);model=make_model(run['arm']);assert hash_state(model.state_dict())==run['initial_sha256']
        model.load_state_dict(cp['model']);assert cp['step']==run['step']==STEPS
        assert all(float(v['step'])==STEPS for v in cp['optimizer']['state'].values())
        pg=cp['optimizer']['param_groups'][0];assert pg['lr']==.001 and pg['betas']==(.9,.999) and pg['eps']==1e-8 and pg['weight_decay']==0
        gen=torch.Generator().manual_seed(run['seed']+1000);ranks=hashlib.sha256();stream=hashlib.sha256()
        counts=dict(sequences=0,supervised_targets=0,content_targets=0,attention_pairs=0)
        for step in range(STEPS):
            n=(3,4,5,6)[step%4];batch=840//(n+1);half=COUNTS[f'train_{n}']//2
            values=torch.rand(2,105,generator=gen,dtype=torch.float64).tolist()
            ranks.update(struct.pack('210d',*(values[0]+values[1])))
            ids=[math.floor(v*half)+g*half for g in range(2) for v in values[g][:batch//2]]
            stream.update(json.dumps([n,ids],separators=(',',':')).encode())
            counts['sequences']+=batch;counts['supervised_targets']+=batch*(n+1);counts['content_targets']+=batch*n
            counts['attention_pairs']+=batch*(n*n+sum(range(1,n+2))+n*(n+1))
        assert counts=={k:run[k] for k in counts}
        assert ranks.hexdigest()==run['rank_stream_sha256'] and stream.hexdigest()==run['sample_stream_sha256']
        assert torch.equal(gen.get_state(),cp['generator']);streams[run['seed'],run['arm']]=stream.hexdigest()
        assert saved.keys()==payload['pools'].keys()==run['metrics'].keys()
        other={**run,'metrics':{}}
        for pool,ev in saved.items():
            actual,cell=audit_cell(model,run['arm'],payload['pools'][pool],run['metrics'][pool],ev)
            other['metrics'][pool]=actual;cells.append(dict(seed=run['seed'],arm=run['arm'],pool=pool,**cell))
        checked.append(other);print(json.dumps(dict(seed=run['seed'],arm=run['arm'],pools=len(saved))),flush=True)
    assert all(streams[s,'sinusoidal']==streams[s,'rope'] for s in SEEDS)
    result=own_decision(checked);assert result==decision(report['runs'])==report['decision']
    wrong=copy.deepcopy(report['runs'])
    for r in wrong:
        for n in range(3,7):
            for g in ('distinct','repeated'):r['metrics'][f'train_{n}'][g]['exact_accuracy']=0 if result['training_fit'][r['arm']] else 1
    assert own_decision(wrong)['training_fit']!=result['training_fit']
    assert all(digest(ROOT/p)==h for p,h in report['source_dependencies'].items())
    totals={k:sum(c[k] for c in cells) for k in ('sequences','teacher_rows','oracle_content_tokens','ordinary_generated_tokens')}
    audit=dict(passed=True,report_sha256=digest(DIRECTORY/'report.json'),audit_source_sha256=digest(Path(__file__)),
        cells_checked=len(cells),**totals,max_logit_error=max(c['logit_error'] for c in cells),
        max_metric_error=max(c['metric_error'] for c in cells),teacher_argmax_changes=0,ordinary_sequence_changes=0,oracle_sequence_changes=0,
        sampling_and_counts_verified=True,criterion_error_detected=True,prior_hashes_unchanged=True,decision=result,cells=cells)
    write(DIRECTORY/'audit.json',audit);print(json.dumps({k:v for k,v in audit.items() if k!='cells'}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()

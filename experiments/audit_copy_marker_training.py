"""Independent equations, decoding, metrics and marked-copy exposure audit."""
import copy
import hashlib
import json
import math
import struct
from pathlib import Path

import torch

from copy_marker_training import ROOT,DIRECTORY,DATA,SEEDS,ARMS,STEPS,LENGTHS,pools_and_checks,decision
from copy_marker_model import make_model,reference
from copy_length_exposure import COUNTS
from audit_copy_rope_training import scalar_metrics
from mqa_training import hash_state
from position_distractors import digest
from recurrent_retrieval import write


def decode(model,content,arm,oracle=False):
    outputs=[[] for _ in content];finished=[False]*len(content)
    prefix=torch.ones(len(content),1,dtype=torch.long)
    limit=content.shape[1] if oracle else content.shape[1]+3
    for _ in range(limit):
        rows=reference(model,content,prefix,arm)[:,-1].tolist()
        candidates=[k for k in range(16) if not oracle or k!=2]
        choices=[max(candidates,key=row.__getitem__) for row in rows]
        for i,c in enumerate(choices):
            if not finished[i]:outputs[i].append(c)
            finished[i]=finished[i] or (not oracle and c==2)
        prefix=torch.cat((prefix,torch.tensor(choices)[:,None]),1)
        if all(finished):break
    return outputs


def audit_cell(model,arm,rows,expected,stored):
    zs=[];ordinary=[];oracles=[]
    with torch.no_grad():
        for content in torch.tensor(rows).split(64):
            prefix=torch.tensor([[1]+row for row in content.tolist()])
            zs.append(reference(model,content,prefix,arm))
            ordinary.extend(decode(model,content,arm));oracles.extend(decode(model,content,arm,oracle=True))
    z=torch.cat(zs);error=float((z-stored['logits']).abs().max());assert error<=1e-9
    assert torch.equal(z.argmax(-1),stored['logits'].argmax(-1))
    assert ordinary==stored['generated'] and oracles==stored['oracle_outputs']
    actual=scalar_metrics(z.tolist(),rows,ordinary,oracles);assert actual.keys()==expected.keys();metric_error=0.
    for g,m in actual.items():
        assert m.keys()==expected[g].keys()
        metric_error=max(metric_error,max(abs(v-expected[g][k]) for k,v in m.items()))
    assert metric_error<=1e-9
    return actual,dict(logit_error=error,metric_error=metric_error,sequences=len(rows),
        teacher_rows=len(rows)*(len(rows[0])+1),oracle_content_tokens=len(rows)*len(rows[0]),
        ordinary_generated_tokens=sum(map(len,ordinary)))


def own_decision(runs):
    fits={};seen={};short={};transfer={};pairs=[]
    for arm in ARMS:
        selected=[r for r in runs if r['arm']==arm];assert len(selected)==len(SEEDS)
        fits[arm]=all(r['status']=='trained' and min(r['metrics'][f'train_{n}'][g]['exact_accuracy']
             for n in range(3,7) for g in ('distinct','repeated'))>=.99 for r in selected)
        seen[arm]=fits[arm] and min(r['metrics'][f'test_{n}'][g]['exact_accuracy']
             for r in selected for n in range(3,7) for g in ('distinct','repeated'))>=.95
        short[arm]=min(r['metrics'][f'uniform_{n}']['all']['exact_accuracy']
             for r in selected for n in range(3,7))>=.9
        transfer[arm]={}
        for n in (8,10):
            groups=('distinct','repeated') if n==8 else ('repeated',)
            transfer[arm][str(n)]=seen[arm] and min(r['metrics'][f'test_{n}'][g]['exact_accuracy']
                for r in selected for g in groups)>=.9
    for seed in SEEDS:
        s=next(r for r in runs if r['seed']==seed and r['arm']=='sinusoidal')
        r=next(r for r in runs if r['seed']==seed and r['arm']=='rope')
        gains={str(n):r['metrics'][f'test_{n}']['all']['exact_accuracy']-s['metrics'][f'test_{n}']['all']['exact_accuracy'] for n in (8,10)}
        worst=min(r['metrics'][f'test_{n}'][g]['exact_accuracy']-s['metrics'][f'test_{n}'][g]['exact_accuracy']
            for n in range(3,7) for g in ('distinct','repeated'))
        pairs.append(dict(seed=seed,long_gains=gains,seen_length_preserved=worst>=-.05))
    return dict(training_fit=fits,seen_length_reliability=seen,uniform_short_copy=short,extrapolation=transfer,
        paired=pairs,rope_benefit=all(fits.values()) and all(p['seen_length_preserved'] and min(p['long_gains'].values())>=.1 for p in pairs))


def main():
    report=json.loads((DIRECTORY/'report.json').read_text(encoding='utf-8'))
    assert report['status']=='complete' and not (DIRECTORY/'audit.json').exists()
    assert all(digest(ROOT/p)==h for p,h in report['source_dependencies'].items())
    pools,checks=pools_and_checks(json.loads(DATA.read_text(encoding='utf-8')))
    assert pools==report['evaluation_pools'] and checks==report['input_checks']
    assert digest(DIRECTORY/'metrics.csv')==report['metrics_sha256']
    pre=json.loads((DIRECTORY/'pretest.json').read_text(encoding='utf-8'))
    assert pre['status']=='all_training_finished_before_test' and len(pre['runs'])==len(SEEDS)*len(ARMS)
    assert all(not any(p.startswith('test_') or p in ('uniform_8','uniform_10') for p in r['metrics']) for r in pre['runs'])
    assert [r['checkpoint_sha256'] for r in pre['runs']]==[r['checkpoint_sha256'] for r in report['runs']]
    checked=[];cells=[];streams={}
    for run in report['runs']:
        for kind in ('checkpoint','scores'):assert digest(ROOT/run[kind])==run[kind+'_sha256']
        cp=torch.load(ROOT/run['checkpoint'],weights_only=True);saved=torch.load(ROOT/run['scores'],weights_only=True)
        torch.manual_seed(run['seed']);model=make_model(run['arm']);assert hash_state(model.state_dict())==run['initial_sha256']
        model.load_state_dict(cp['model']);assert cp['step']==run['step']==STEPS
        assert all(float(s['step'])==STEPS for s in cp['optimizer']['state'].values())
        pg=cp['optimizer']['param_groups'][0];assert pg['lr']==.001 and pg['betas']==(.9,.999) and pg['eps']==1e-8 and pg['weight_decay']==0
        gen=torch.Generator().manual_seed(run['seed']+1000);ranks=hashlib.sha256();stream=hashlib.sha256()
        counts=dict(sequences=0,content_targets=0,supervised_targets=0,source_tokens=0,source_end_tokens=0,attention_pairs=0)
        for step in range(STEPS):
            n=(3,4,5,6)[step%4];batch=840//(n+1);half=COUNTS[f'train_{n}']//2
            values=torch.rand(2,105,generator=gen,dtype=torch.float64).tolist();ranks.update(struct.pack('210d',*(values[0]+values[1])))
            ids=[math.floor(v*half)+g*half for g in range(2) for v in values[g][:batch//2]]
            stream.update(json.dumps([n,ids],separators=(',',':')).encode())
            counts['sequences']+=batch;counts['content_targets']+=batch*n;counts['supervised_targets']+=batch*(n+1)
            counts['source_tokens']+=batch*(n+1);counts['source_end_tokens']+=batch
            source_pairs=sum(1 for q in range(n+1) for k in range(n+1))
            decoder_pairs=sum(q+1 for q in range(n+1));cross_pairs=(n+1)*(n+1)
            counts['attention_pairs']+=batch*(source_pairs+decoder_pairs+cross_pairs)
        assert counts=={k:run[k] for k in counts}
        assert ranks.hexdigest()==run['rank_stream_sha256'] and stream.hexdigest()==run['sample_stream_sha256']
        assert torch.equal(gen.get_state(),cp['generator']);streams[run['seed'],run['arm']]=stream.hexdigest()
        assert saved.keys()==pools.keys()==run['metrics'].keys();other={**run,'metrics':{}}
        for pool,ev in saved.items():
            actual,cell=audit_cell(model,run['arm'],pools[pool],run['metrics'][pool],ev)
            other['metrics'][pool]=actual;cells.append(dict(seed=run['seed'],arm=run['arm'],pool=pool,**cell))
        checked.append(other);print(json.dumps(dict(seed=run['seed'],arm=run['arm'],pools=len(saved))),flush=True)
    assert all(streams[s,'sinusoidal']==streams[s,'rope'] for s in SEEDS)
    assert len(cells)==len(SEEDS)*len(ARMS)*len(pools)
    result=own_decision(checked);assert result==decision(report['runs'])==report['decision']
    wrong=copy.deepcopy(report['runs'])
    for r in wrong:
        for n in range(3,7):
            for g in ('distinct','repeated'):r['metrics'][f'train_{n}'][g]['exact_accuracy']=0 if result['training_fit'][r['arm']] else 1
    assert own_decision(wrong)['training_fit']!=result['training_fit']
    assert all(digest(ROOT/p)==h for p,h in report['source_dependencies'].items())
    totals={k:sum(c[k] for c in cells) for k in ('sequences','teacher_rows','oracle_content_tokens','ordinary_generated_tokens')}
    audit=dict(passed=True,report_sha256=digest(DIRECTORY/'report.json'),audit_source_sha256=digest(Path(__file__)),
        cells_checked=len(cells),**totals,max_logit_error=max(c['logit_error'] for c in cells),max_metric_error=max(c['metric_error'] for c in cells),
        teacher_argmax_changes=0,ordinary_sequence_changes=0,oracle_sequence_changes=0,
        sampling_and_counts_verified=True,criterion_error_detected=True,prior_hashes_unchanged=True,decision=result,cells=cells)
    write(DIRECTORY/'audit.json',audit);print(json.dumps({k:v for k,v in audit.items() if k!='cells'}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()

"""Recompute factorial scores, streams and decisions from frozen checkpoints."""
import hashlib
import json
from pathlib import Path

import torch

from copy_position_sites import ROOT, OUT, ARMS, make_model, forward, digest, write, hash_state
from audit_copy_position_exposure import close, own_measures, validate_data


def audit_cell(model,rows,pos,arm,stored,expected):
    # Explicit arm wiring, independent of the producer's SITES lookup.
    self_pos=pos if arm in ('self_only','both') else None
    cross_pos=pos if arm in ('cross_only','both') else None
    zs=[];ss=[];outputs=[]
    with torch.no_grad():
        for start in range(0,len(rows),64):
            part=rows[start:start+64];x=torch.tensor(part);prefix=torch.tensor([[1]+r for r in part])
            z,s=forward(model,x,prefix,self_pos,cross_pos,True);zs.append(z);ss.append(s)
            prefix=[[1] for _ in part];done=[False]*len(part);answers=[[] for _ in part]
            for _ in range(len(part[0])+3):
                zz=forward(model,x,torch.tensor(prefix),self_pos,cross_pos,True)[0][:,-1].tolist()
                for i,row in enumerate(zz):
                    token=max(range(16),key=row.__getitem__);prefix[i].append(token)
                    if not done[i]:answers[i].append(token)
                    done[i]=done[i] or token==2
                if all(done):break
            outputs.extend(answers)
    z,s=torch.cat(zs),torch.cat(ss)
    error=max(float((z-stored['logits']).abs().max()),float((s-stored['scores']).abs().max()))
    assert error<=1e-9 and torch.equal(z.argmax(-1),stored['logits'].argmax(-1))
    assert outputs==stored['generated']
    return error,close(own_measures(rows,z,s,outputs),expected)


def own_fit(metrics,arm):
    cells=[v for k,v in metrics.items() if k.startswith('train_') or (arm!='ordinary' and k.startswith('layout'))]
    assert len(cells)==(4 if arm=='ordinary' else 20)
    return min(m[g]['exact_accuracy'] for m in cells for g in ('distinct','repeated'))>=.99


def own_decision(runs):
    seeds=[]
    for seed in sorted({r['seed'] for r in runs}):
        rr={r['arm']:r for r in runs if r['seed']==seed};assert set(rr)==set(ARMS)
        fit_ok=all(own_fit(r['metrics'],a) for a,r in rr.items());short={};cells=[]
        for a in ARMS:
            short[a]=min(rr[a]['metrics'][f'test_{n}'][g]['exact_accuracy']-rr['ordinary']['metrics'][f'test_{n}'][g]['exact_accuracy'] for n in range(3,7) for g in ('distinct','repeated'))
        for n in (8,10):
            for g in ('distinct','repeated') if n==8 else ('repeated',):
                acc={a:r['metrics'][f'test_{n}'][g]['exact_accuracy'] for a,r in rr.items()}
                rank={a:r['metrics'][f'test_{n}'][g]['outranked_fraction'] for a,r in rr.items()}
                gains={a:dict(copy=acc[a]-acc['ordinary'],ranking=rank['ordinary']-rank[a]) for a in ARMS}
                enough={}
                for a in ('self_only','cross_only'):
                    enough[a]=acc[a]>=.99 and acc['both']-acc[a]<=.05 and gains[a]['copy']>=.1 and gains[a]['ranking']>=.1
                cells.append(dict(length=n,group=g,accuracy=acc,outranked=rank,gains=gains,sufficient=enough,
                    copy_interaction=(acc['both']-acc['cross_only'])-(acc['self_only']-acc['ordinary'])))
        replicated=fit_ok and all(v>=-.05 for v in short.values())
        for c in cells:replicated &= c['accuracy']['both']>=.99 and c['gains']['both']['copy']>=.1 and c['gains']['both']['ranking']>=.1
        seeds.append(dict(seed=seed,fit=fit_ok,short_worst_change=short,cells=cells,combined_replicated=bool(replicated)))
    valid=bool(seeds) and all(s['combined_replicated'] for s in seeds)
    return dict(seeds=seeds,combined_replicated=valid,
        self_only_sufficient=valid and all(c['sufficient']['self_only'] for s in seeds for c in s['cells']),
        cross_only_sufficient=valid and all(c['sufficient']['cross_only'] for s in seeds for c in s['cells']))


def audit(directory=OUT,save=True,formal=True):
    directory=Path(directory)
    if save:assert not (directory/'audit.json').exists()
    report=json.loads((directory/'report.json').read_text());lock=json.loads((directory/'lock.json').read_text())
    assert report['status'] in ('complete','development_fit_failed')
    assert report['lock_sha256']==digest(directory/'lock.json')
    assert all(digest(ROOT/p)==sha for p,sha in lock['protected'].items())
    data=json.loads((directory/'data.json').read_text());validate_data(data,formal)
    if formal:
        previous=json.loads((ROOT/'results/copy_position_exposure/data.json').read_text())['pools']
        seen={tuple(r) for rows in previous.values() for r in rows}
        assert not any(tuple(r) in seen for n in (8,10) for r in data['pools'][f'test_{n}'])
    maxerror=maxmetric=0.;cells=answers=0;checked=[]
    for run in report['development']+report['confirmation']:
        assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256'] and digest(ROOT/run['scores'])==run['scores_sha256']
        torch.manual_seed(run['seed']);model=make_model('rope');assert hash_state(model.state_dict())==run['initial_sha256']
        cp=torch.load(ROOT/run['checkpoint'],weights_only=True);assert cp['step']==run['step']
        assert all(int(s['step'])==run['step'] for s in cp['optimizer']['state'].values())
        if run['begin']:
            prior=next(r for r in report['development'] if r['arm']==run['arm'] and r['step']==run['begin'])
            assert run['previous_sha256']==prior['checkpoint_sha256']
        model.load_state_dict(cp['model']);model.eval();stored=torch.load(ROOT/run['scores'],weights_only=True)
        gen=torch.Generator().manual_seed(run['seed']+1000);pg=torch.Generator().manual_seed(run['seed']+2000)
        stream=hashlib.sha256();pstream=hashlib.sha256();sequences=targets=0
        for step in range(1,run['step']+1):
            n=3+(step-1)%4;batch=840//(n+1);half=len(data['pools'][f'train_{n}'])//2
            ranks=torch.rand(2,105,generator=gen,dtype=torch.float64)
            idx=torch.floor(ranks[:,:batch//2]*half).long();idx[1]+=half
            p=torch.sort(torch.randperm(11,generator=pg)[:n+1]).values.tolist()
            if step>run['begin']:
                stream.update(json.dumps([n,idx.flatten().tolist()]).encode());pstream.update(json.dumps(p).encode())
                sequences+=batch;targets+=batch*(n+1)
        assert stream.hexdigest()==run['stream_sha256'] and pstream.hexdigest()==run['position_stream_sha256']
        assert torch.equal(gen.get_state(),cp['generator']) and torch.equal(pg.get_state(),cp['position_generator'])
        expected={p for p in data['pools'] if run in report['confirmation'] or p.startswith(('train_','validation_'))}
        expected|={f'layout{j}_{n}' for n in range(3,7) for j in range(4)}
        assert set(stored)==set(run['metrics'])==expected
        for name in sorted(expected):
            if name.startswith('layout'):
                layout,n=name.split('_');rows=data['pools'][f'train_{n}'];pos=torch.tensor(data['layouts'][n][int(layout[6:])])
            else:rows=data['pools'][name];pos=None
            error,me=audit_cell(model,rows,pos,run['arm'],stored[name],run['metrics'][name]);cells+=1;answers+=len(rows)
            maxerror=max(maxerror,error);maxmetric=max(maxmetric,me)
        assert own_fit(run['metrics'],run['arm'])==run['fit']
        checked.append(dict(seed=run['seed'],arm=run['arm'],step=run['step'],begin=run['begin'],segment_sequences=sequences,segment_targets=targets))
        print(json.dumps(dict(audited_seed=run['seed'],arm=run['arm'],step=run['step'],cells=cells)),flush=True)
    chosen=None
    for stage in lock['stages']:
        group=[r for r in report['development'] if r['step']==stage]
        if not group:break
        assert len(group)==4 and {r['arm'] for r in group}==set(ARMS)
        assert all(r['seed']==lock['dev'] for r in group)
        for key in ('initial_sha256','stream_sha256','position_stream_sha256'):assert len({r[key] for r in group})==1
        if all(own_fit(r['metrics'],r['arm']) for r in group):chosen=stage;break
    assert chosen==report['selected_steps']
    selection=json.loads((directory/'development-selection.json').read_text())
    close(selection,dict(selected_steps=chosen,runs=report['development']))
    if chosen is None:
        assert report['status']=='development_fit_failed' and not report['confirmation']
        assert {r['step'] for r in report['development']}==set(lock['stages'])
    else:
        assert report['status']=='complete'
        assert {(r['seed'],r['arm']) for r in report['confirmation']}=={(s,a) for s in lock['confirm'] for a in ARMS}
        assert len(report['confirmation'])==len(lock['confirm'])*4
        for seed in lock['confirm']:
            quartet=[r for r in report['confirmation'] if r['seed']==seed]
            assert all(r['step']==chosen and r['begin']==0 for r in quartet)
            for key in ('initial_sha256','stream_sha256','position_stream_sha256'):assert len({r[key] for r in quartet})==1
        pre=json.loads((directory/'pretest.json').read_text())
        assert [r['checkpoint_sha256'] for r in pre['confirmation']]==[r['checkpoint_sha256'] for r in report['confirmation']]
        assert all(not any(p.startswith('test_') for p in r['metrics']) for r in pre['confirmation'])
        close(own_decision(report['confirmation']),report['decision'])
    assert all(digest(ROOT/p)==sha for p,sha in lock['protected'].items())
    result=dict(passed=True,cells=cells,sequence_evaluations=answers,max_logit_score_error=maxerror,max_metric_error=maxmetric,
        prediction_differences=0,runs=checked,report_sha256=digest(directory/'report.json'),protected_files=len(lock['protected']),
        limitation='Reference equations, metrics, decoding, streams and counters checked; shared forward structure/projection helpers; full training gradients not replayed.')
    if save:write(directory/'audit.json',result)
    return result


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    print(json.dumps(audit(),indent=2))

"""Saved-checkpoint equation/decoding audit and independent stream/fit checks."""
import hashlib
import json
import math
from pathlib import Path

import torch

from copy_position_exposure import ROOT, OUT, ARMS, make_model, forward, digest, write
from audit_copy_length_exposure import scalar_metrics
from mqa_training import hash_state


def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return max([close(a[k],b[k]) for k in a]+[0.])
    if isinstance(a,list):
        assert len(a)==len(b)
        return max([close(x,y) for x,y in zip(a,b)]+[0.])
    if isinstance(a,float):
        e=abs(a-b);assert math.isfinite(e) and e<=1e-9,(a,b);return e
    assert a==b,(a,b)
    return 0.


def own_measures(rows,z,scores,outputs):
    met=scalar_metrics(z.tolist(),rows,outputs)
    scores=scores.tolist();first=z[:,0].tolist()
    for g in ('all','distinct','repeated'):
        ids=[i for i,r in enumerate(rows) if g=='all' or (len(set(r))==len(r))==(g=='distinct')]
        if not ids:met[g]=None;continue
        gaps=[[head[0]-max(head[1:]) for head in scores[i]] for i in ids]
        met[g].update(first_accuracy=sum(max(range(16),key=first[i].__getitem__)==rows[i][0] for i in ids)/len(ids),
                      outranked_fraction=sum(any(v<-1e-10 for v in row) for row in gaps)/len(ids),
                      near_tie_fraction=sum(any(abs(v)<=1e-10 for v in row) for row in gaps)/len(ids))
    return met


def audit_cell(model,rows,pos,stored,expected):
    zs=[];ss=[];outputs=[]
    with torch.no_grad():
        for start in range(0,len(rows),64):
            part=rows[start:start+64];x=torch.tensor(part)
            prefix=torch.tensor([[1]+r for r in part])
            z,s=forward(model,x,prefix,pos,True);zs.append(z);ss.append(s)
            prefix=[[1] for _ in part];done=[False]*len(part);answers=[[] for _ in part]
            for _ in range(len(part[0])+3):
                zz=forward(model,x,torch.tensor(prefix),pos,True)[0][:,-1].tolist()
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
    actual=own_measures(rows,z,s,outputs)
    metric_error=close(actual,expected)
    return actual,error,metric_error


def own_fit(metrics,arm):
    cells=[m for name,m in metrics.items() if name.startswith('train_') or (arm=='spaced' and name.startswith('layout'))]
    assert len(cells)==(20 if arm=='spaced' else 4)
    return min(m[g]['exact_accuracy'] for m in cells for g in ('distinct','repeated'))>=.99


def own_decision(runs):
    pairs=[]
    for seed in sorted({r['seed'] for r in runs}):
        a=next(r for r in runs if r['seed']==seed and r['arm']=='contiguous')
        b=next(r for r in runs if r['seed']==seed and r['arm']=='spaced')
        groups=[]
        for n in (8,10):
            for g in ('distinct','repeated') if n==8 else ('repeated',):
                am,bm=a['metrics'][f'test_{n}'][g],b['metrics'][f'test_{n}'][g]
                groups.append(dict(length=n,group=g,copy_gain=bm['exact_accuracy']-am['exact_accuracy'],ranking_gain=am['outranked_fraction']-bm['outranked_fraction']))
        preserve=min(b['metrics'][f'test_{n}'][g]['exact_accuracy']-a['metrics'][f'test_{n}'][g]['exact_accuracy'] for n in range(3,7) for g in ('distinct','repeated'))
        pairs.append(dict(seed=seed,fit=own_fit(a['metrics'],a['arm']) and own_fit(b['metrics'],b['arm']),short_worst_change=preserve,groups=groups))
    supported=bool(pairs)
    for p in pairs:
        supported &= p['fit'] and p['short_worst_change']>=-.05
        for g in p['groups']:supported &= g['copy_gain']>=.1 and g['ranking_gain']>=.1
    return dict(pairs=pairs,supported=bool(supported))


def validate_data(data,formal=True):
    for name,rows in data['pools'].items():
        n=int(name.rsplit('_',1)[1]);assert all(len(r)==n and all(3<=v<=10 for v in r) for r in rows)
        if name.startswith('train_'):
            assert all(len(set(r))==n for r in rows[:len(rows)//2])
            assert all(len(set(r))<n for r in rows[len(rows)//2:])
    for n,layouts in data['layouts'].items():
        assert len(layouts)==4
        for p in layouts:assert len(p)==int(n)+1 and p==sorted(set(p)) and min(p)>=0 and max(p)<=10
    if formal:
        old=json.loads((ROOT/'results/copy_marker_training/report.json').read_text())['evaluation_pools']
        seen={tuple(r) for rows in old.values() for r in rows}
        for name,rows in data['pools'].items():
            if name in ('test_8','test_10'):
                assert len(rows)==128
                for i,r in enumerate(rows):
                    assert tuple(r) not in seen;seen.add(tuple(r))
                    assert (len(set(r))==len(r)) == (name=='test_8' and i<64)
            else:assert rows==old[name]


def audit(directory=OUT,save=True,formal=True):
    directory=Path(directory)
    if save:assert not (directory/'audit.json').exists()
    report=json.loads((directory/'report.json').read_text());lock=json.loads((directory/'lock.json').read_text())
    assert report['status'] in ('complete','development_fit_failed')
    assert report['lock_sha256']==digest(directory/'lock.json')
    assert all(digest(ROOT/p)==h for p,h in lock['protected'].items())
    data=json.loads((directory/'data.json').read_text());validate_data(data,formal)
    cells=answers=0;maxerror=maxmetric=0.;checked=[]
    for run in report['development']+report['confirmation']:
        assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256'] and digest(ROOT/run['scores'])==run['scores_sha256']
        torch.manual_seed(run['seed']);model=make_model('rope');assert hash_state(model.state_dict())==run['initial_sha256']
        cp=torch.load(ROOT/run['checkpoint'],weights_only=True);assert cp['step']==run['step']
        assert all(int(s['step'])==run['step'] for s in cp['optimizer']['state'].values())
        model.load_state_dict(cp['model']);model.eval();stored=torch.load(ROOT/run['scores'],weights_only=True)
        gen=torch.Generator().manual_seed(run['seed']+1000);pg=torch.Generator().manual_seed(run['seed']+2000)
        stream=hashlib.sha256();pstream=hashlib.sha256();sequences=targets=0
        for step in range(1,run['step']+1):
            n=3+(step-1)%4;batch=840//(n+1);half=len(data['pools'][f'train_{n}'])//2
            ranks=torch.rand(2,105,generator=gen,dtype=torch.float64)
            idx=torch.floor(ranks[:,:batch//2]*half).long();idx[1]+=half
            p=torch.sort(torch.randperm(11,generator=pg)[:n+1]).values.tolist() if run['arm']=='spaced' else None
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
            _,error,me=audit_cell(model,rows,pos,stored[name],run['metrics'][name]);cells+=1;answers+=len(rows)
            maxerror=max(maxerror,error);maxmetric=max(maxmetric,me)
        assert own_fit(run['metrics'],run['arm'])==run['fit']
        checked.append(dict(seed=run['seed'],arm=run['arm'],step=run['step'],segment_sequences=sequences,segment_targets=targets))
        print(json.dumps(dict(audited_seed=run['seed'],arm=run['arm'],step=run['step'],cells=cells)),flush=True)
    chosen=None
    for stage in lock['stages']:
        pair=[r for r in report['development'] if r['step']==stage]
        if len(pair)!=2:break
        assert {r['arm'] for r in pair}==set(ARMS)
        assert pair[0]['stream_sha256']==pair[1]['stream_sha256']
        if all(own_fit(r['metrics'],r['arm']) for r in pair):chosen=stage;break
    assert chosen==report['selected_steps']
    if chosen is None:
        assert report['status']=='development_fit_failed' and not report['confirmation']
    else:
        assert {(r['seed'],r['arm']) for r in report['confirmation']}=={(s,a) for s in lock['confirm'] for a in ARMS}
        pre=json.loads((directory/'pretest.json').read_text())
        assert [r['checkpoint_sha256'] for r in pre['confirmation']]==[r['checkpoint_sha256'] for r in report['confirmation']]
        assert all(not any(p.startswith('test_') for p in r['metrics']) for r in pre['confirmation'])
        close(own_decision(report['confirmation']),report['decision'])
    assert all(digest(ROOT/p)==h for p,h in lock['protected'].items())
    result=dict(passed=True,cells=cells,sequence_evaluations=answers,max_logit_score_error=maxerror,max_metric_error=maxmetric,
         prediction_differences=0,runs=checked,report_sha256=digest(directory/'report.json'),protected_files=len(lock['protected']),
         limitation='Outputs, streams, initialization, optimizer counters and decisions checked; full training gradients/updates not replayed. Projection/attention utilities shared.')
    if save:write(directory/'audit.json',result)
    return result


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    print(json.dumps(audit(),indent=2))

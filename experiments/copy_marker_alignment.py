"""Fixed saved-checkpoint alignment probe; learned source values are retained."""
import csv
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from encoder_decoder_contract import ROOT
from encoder_decoder_copy import inputs, logits, generate
from copy_marker_model import make_model, marked_source
from copy_rope_model import attention, rotate
from copy_length_exposure import groups, metrics as ordinary_metrics
from copy_eos_training import oracle, measures as oracle_metrics
from position_distractors import digest
from recurrent_retrieval import write

DIRECTORY=ROOT/'results/copy_marker_alignment'
PROTOCOL=ROOT/'docs/protocols/copy-marker-alignment.md'
OLD=ROOT/'results/copy_marker_training/report.json'
SEEDS=(3163,3167,3169)
POOLS=('train_6','validation_6','test_6','test_8','test_10','uniform_6','uniform_8','uniform_10')
MODES=('native','aligned','shifted')


def route(weights, mode):
    if mode=='native':return weights
    if mode not in MODES:raise ValueError(mode)
    result=weights.clone();end=min(weights.shape[-2],weights.shape[-1])
    j=torch.arange(end);k=j if mode=='aligned' else (j+1)%weights.shape[-1]
    result[:,:,:end]=0;result[:,:,j,k]=1
    return result


def patched(model, content, prefix, mode):
    base=model.base;enc,dec=base.encoder,base.decoder
    s,t=base.features(marked_source(content),prefix)
    ep,dp=torch.arange(s.shape[1]),torch.arange(t.shape[1])
    sp=torch.zeros(s.shape[:2],dtype=torch.bool);tp=torch.zeros_like(prefix,dtype=torch.bool)
    memory=enc.norm1(s+attention(enc.self_attn,s,s,s,sp,ep,ep))
    memory=enc.norm2(memory+enc.linear2(F.relu(enc.linear1(memory))))
    x=dec.norm1(t+attention(dec.self_attn,t,t,t,tp,dp,dp,True))
    m=dec.multihead_attn;heads=m.num_heads;d=m.embed_dim//heads
    w=m.in_proj_weight.chunk(3);b=m.in_proj_bias.chunk(3)
    q,k,v=[F.linear(a,p,c).reshape(len(content),a.shape[1],heads,d).transpose(1,2)
           for a,p,c in zip((x,memory,memory),w,b)]
    q,k=rotate(q,dp),rotate(k,ep)
    raw=(q@k.transpose(-1,-2)/math.sqrt(d)).softmax(-1)
    cross=(route(raw,mode)@v).transpose(1,2).reshape_as(x)
    x=dec.norm2(x+m.out_proj(cross))
    return base.output(dec.norm3(x+dec.linear2(F.relu(dec.linear1(x))))),raw


def details(z, rows, raw):
    source=torch.tensor(rows);_,labels=inputs(source);n=source.shape[1];out={}
    loss=F.cross_entropy(z.flatten(0,1),labels.flatten(),reduction='none').reshape_as(labels)
    top=z.argmax(-1);good=top==labels
    other=z.clone();other.scatter_(-1,labels[:,:,None],-torch.inf)
    margin=z.gather(-1,labels[:,:,None]).squeeze(-1)-other.max(-1).values
    for g,mask in groups(rows).items():
        first=[next((j for j,v in enumerate(row) if not v),n+1) for row in good[mask].tolist()]
        out[g]=dict(first_error_counts=[first.count(j) for j in range(n+2)],positions=[])
        for j in range(n+1):
            out[g]['positions'].append(dict(row=j,kind='EOS' if j==n else 'content',
                count=int(mask.sum()),correct=int(good[mask,j].sum()),eos_top=int((top[mask,j]==2).sum()),
                mean_ce=float(loss[mask,j].mean()),mean_correct_margin=float(margin[mask,j].mean()),
                native_matching_mass=float(raw[mask,:,j,j].mean())))
    return out


def evaluate(model, rows, mode):
    zs=[];weights=[];normal=[];oracles=[]
    with torch.no_grad():
        for x in torch.tensor(rows).split(64):
            prefix,_=inputs(x);z,w=patched(model,x,prefix,mode);zs.append(z);weights.append(w)
            fn=lambda a,b:patched(model,a,b,mode)[0][:,-1]
            normal.extend(generate(fn,x,x.shape[1]+3));oracles.extend(oracle(fn,x))
    z=torch.cat(zs);raw=torch.cat(weights)
    a=ordinary_metrics(z,rows,normal);b=oracle_metrics(z,rows,oracles,1.)
    met={g:dict(a[g],**b[g]) for g in a}
    return dict(metrics=met,diagnostics=details(z,rows,raw)),dict(logits=z,native_weights=raw,generated=normal,oracle_outputs=oracles)


def decision(cells):
    index={(c['seed'],c['pool'],c['mode']):c for c in cells};pairs=[]
    for seed in SEEDS:
        gains=[];changes=[]
        for pool in ('test_8','test_10'):
            for g,m in index[seed,pool,'native']['metrics'].items():
                if g=='all':continue
                gains.append(dict(pool=pool,group=g,oracle_gain=index[seed,pool,'aligned']['metrics'][g]['oracle_content_exact']-m['oracle_content_exact']))
        for pool in ('train_6','validation_6','test_6'):
            for g,m in index[seed,pool,'native']['metrics'].items():
                if g!='all':changes.append(index[seed,pool,'aligned']['metrics'][g]['exact_accuracy']-m['exact_accuracy'])
        pairs.append(dict(seed=seed,long_groups=gains,short_worst_change=min(changes),
            gain_rule=all(v['oracle_gain']>=.1 for v in gains),short_preserved=min(changes)>=-.05))
    return dict(paired=pairs,conditional_recovery=all(p['gain_rule'] and p['short_preserved'] for p in pairs),
        full_long_recovery=all(c['metrics']['all']['exact_accuracy']==1 for c in cells if c['mode']=='aligned' and c['pool'] in ('test_8','test_10')))


def main():
    assert PROTOCOL.exists() and (DIRECTORY/'preflight.json').exists() and not (DIRECTORY/'report.json').exists()
    old=json.loads(OLD.read_text(encoding='utf-8'));audit=json.loads((OLD.parent/'audit.json').read_text(encoding='utf-8'))
    assert old['status']=='complete' and audit['passed'] and audit['report_sha256']==digest(OLD)
    runs=[r for r in old['runs'] if r['arm']=='rope'];assert tuple(r['seed'] for r in runs)==SEEDS
    assert all(digest(ROOT/r[k])==r[k+'_sha256'] for r in runs for k in ('checkpoint','scores'))
    pools={p:old['evaluation_pools'][p] for p in POOLS}
    paths=[PROTOCOL,Path(__file__),ROOT/'experiments/audit_copy_marker_alignment.py',ROOT/'tests/test_copy_marker_alignment.py',
           DIRECTORY/'preflight.json',OLD,OLD.parent/'audit.json',ROOT/'results/copy_length_exposure/data.json',
           *[ROOT/'experiments'/p for p in ('copy_rope_model.py','copy_marker_model.py','encoder_decoder_copy.py','encoder_decoder_contract.py',
             'copy_rope_training.py','copy_eos_training.py','copy_length_exposure.py','audit_copy_rope_training.py',
             'audit_copy_eos_training.py','audit_copy_length_exposure.py','audit_kv_cache_equivalence.py','position_distractors.py','recurrent_retrieval.py')],
           *[ROOT/r[k] for r in runs for k in ('checkpoint','scores')]]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    records=[];evidence={};report=dict(status='running',training_updates=0,device='cpu',dtype='float64',threads=1,
        seeds=list(SEEDS),pools=pools,modes=list(MODES),source_dependencies=hashes,
        expected_cells=len(SEEDS)*len(POOLS)*len(MODES),cells=records)
    write(DIRECTORY/'report.json',report)
    for run in runs:
        model=make_model('rope').eval();model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        archived=torch.load(ROOT/run['scores'],weights_only=True)
        for pool,rows in pools.items():
            for mode in MODES:
                result,stored=evaluate(model,rows,mode)
                if mode=='native':
                    assert float((stored['logits']-archived[pool]['logits']).abs().max())<=1e-9
                    assert stored['generated']==archived[pool]['generated'] and stored['oracle_outputs']==archived[pool]['oracle_outputs']
                    for g,m in result['metrics'].items():
                        assert max(abs(v-run['metrics'][pool][g][k]) for k,v in m.items())<=1e-9
                cell=dict(seed=run['seed'],pool=pool,mode=mode,**result);records.append(cell)
                evidence[run['seed'],pool,mode]=stored
            print(json.dumps(dict(seed=run['seed'],pool=pool,ordinary=[c['metrics']['all']['exact_accuracy'] for c in records[-3:]],
                oracle=[c['metrics']['all']['oracle_content_exact'] for c in records[-3:]])),flush=True)
            write(DIRECTORY/'report.json',report)
    assert len(records)==report['expected_cells'] and all(digest(ROOT/p)==h for p,h in hashes.items())
    torch.save(evidence,DIRECTORY/'evidence.pt')
    rows=[dict(seed=c['seed'],pool=c['pool'],mode=c['mode'],group=g,**m) for c in records for g,m in c['metrics'].items()]
    with (DIRECTORY/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report.update(status='complete',prior_hashes_unchanged=True,evidence_sha256=digest(DIRECTORY/'evidence.pt'),
        metrics_sha256=digest(DIRECTORY/'metrics.csv'),decision=decision(records))
    write(DIRECTORY/'report.json',report);print(json.dumps(report['decision']),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()

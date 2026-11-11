"""Explicit layer equations and direct value indexing for the alignment probe."""
import json
import math
from pathlib import Path

import torch

from copy_marker_alignment import ROOT,DIRECTORY,SEEDS,POOLS,MODES,OLD,decision
from copy_marker_model import make_model
from copy_rope_model import attention,matrix_rotation
from audit_kv_cache_equivalence import norm
from audit_copy_rope_training import scalar_metrics
from position_distractors import digest
from recurrent_retrieval import write


def reference(model, rows, prefix, mode):
    base=model.base;source=torch.tensor([r+[2] for r in rows.tolist()]);enc,dec=base.encoder,base.decoder
    s=base.source(source)*math.sqrt(8);t=base.target(prefix)*math.sqrt(8)
    ep,dp=torch.arange(source.shape[1]),torch.arange(prefix.shape[1]);sp=torch.zeros_like(source,dtype=torch.bool);tp=torch.zeros_like(prefix,dtype=torch.bool)
    def ff(layer,x):return (x@layer.linear1.weight.T+layer.linear1.bias).clamp_min(0)@layer.linear2.weight.T+layer.linear2.bias
    memory=norm(s+attention(enc.self_attn,s,s,s,sp,ep,ep,reference=True),enc.norm1)
    memory=norm(memory+ff(enc,memory),enc.norm2)
    x=norm(t+attention(dec.self_attn,t,t,t,tp,dp,dp,True,True),dec.norm1)
    m=dec.multihead_attn;wq,wk,wv=m.in_proj_weight.chunk(3);bq,bk,bv=m.in_proj_bias.chunk(3);h=m.num_heads;d=m.embed_dim//h
    q=matrix_rotation((x@wq.T+bq).reshape(len(rows),len(dp),h,d).transpose(1,2),dp)
    k=matrix_rotation((memory@wk.T+bk).reshape(len(rows),len(ep),h,d).transpose(1,2),ep)
    v=(memory@wv.T+bv).reshape(len(rows),len(ep),h,d).transpose(1,2)
    scores=q@k.transpose(-1,-2)/math.sqrt(d);ex=(scores-scores.max(-1,keepdim=True).values).exp();raw=ex/ex.sum(-1,keepdim=True)
    # No producer selector/one-hot matrix is used here.
    parts=[]
    for j in range(len(dp)):
        if mode=='native' or j>=len(ep):parts.append(raw[:,:,j:j+1]@v)
        else:
            col=j if mode=='aligned' else (j+1)%len(ep)
            parts.append(v[:,:,col:col+1])
    cross=torch.cat(parts,2).transpose(1,2).reshape_as(x)@m.out_proj.weight.T+m.out_proj.bias
    x=norm(x+cross,dec.norm2);x=norm(x+ff(dec,x),dec.norm3)
    return x@base.output.weight.T,raw


def decode(model, rows, mode, oracle=False):
    outputs=[[] for _ in rows];active=[True]*len(rows);prefix=[[1] for _ in rows];n=rows.shape[1]
    for _ in range(n if oracle else n+3):
        z=reference(model,rows,torch.tensor(prefix),mode)[0][:,-1].tolist()
        allowed=[k for k in range(16) if not oracle or k!=2]
        for i,row in enumerate(z):
            token=max(allowed,key=row.__getitem__);prefix[i].append(token)
            if active[i]:outputs[i].append(token)
            if not oracle and token==2:active[i]=False
        if not any(active):break
    return outputs


def own_details(z,rows,raw):
    n=len(rows[0]);out={}
    for g in ('all','distinct','repeated'):
        ids=[i for i,r in enumerate(rows) if g=='all' or (len(set(r))==n)==(g=='distinct')]
        if not ids:continue
        first=[0]*(n+2);positions=[]
        for i in ids:
            labels=rows[i]+[2];j=next((j for j,r in enumerate(z[i]) if max(range(16),key=r.__getitem__)!=labels[j]),n+1);first[j]+=1
        for j in range(n+1):
            correct=eos=0;ces=[];margins=[];mass=[]
            for i in ids:
                values=z[i][j];label=2 if j==n else rows[i][j];top=max(range(16),key=values.__getitem__)
                correct+=top==label;eos+=top==2;m=max(values)
                ces.append(m+math.log(math.fsum(math.exp(v-m) for v in values))-values[label])
                margins.append(values[label]-max(v for k,v in enumerate(values) if k!=label))
                mass.extend(head[j][j] for head in raw[i])
            positions.append(dict(row=j,kind='EOS' if j==n else 'content',count=len(ids),correct=correct,eos_top=eos,
                mean_ce=math.fsum(ces)/len(ces),mean_correct_margin=math.fsum(margins)/len(margins),native_matching_mass=math.fsum(mass)/len(mass)))
        out[g]=dict(first_error_counts=first,positions=positions)
    return out


def compare(actual, expected):
    if isinstance(actual,dict):
        assert actual.keys()==expected.keys();return max([compare(v,expected[k]) for k,v in actual.items()]+[0.])
    if isinstance(actual,list):
        assert len(actual)==len(expected);return max([compare(a,b) for a,b in zip(actual,expected)]+[0.])
    if isinstance(actual,(int,float)):
        error=abs(actual-expected);assert error<=1e-9;return error
    assert actual==expected;return 0.


def audit_cell(model, rows, mode, stored, expected):
    zs=[];weights=[];normal=[];oracles=[]
    with torch.no_grad():
        for x in torch.tensor(rows).split(64):
            prefix=torch.tensor([[1]+r for r in x.tolist()]);z,w=reference(model,x,prefix,mode)
            zs.append(z);weights.append(w);normal.extend(decode(model,x,mode));oracles.extend(decode(model,x,mode,True))
    z=torch.cat(zs);w=torch.cat(weights)
    error=max(float((z-stored['logits']).abs().max()),float((w-stored['native_weights']).abs().max()));assert error<=1e-9
    assert torch.equal(z.argmax(-1),stored['logits'].argmax(-1))
    assert normal==stored['generated'] and oracles==stored['oracle_outputs']
    actual=dict(metrics=scalar_metrics(z.tolist(),rows,normal,oracles),diagnostics=own_details(z.tolist(),rows,w.tolist()))
    metric_error=compare(actual,{k:expected[k] for k in ('metrics','diagnostics')})
    return actual,dict(max_logit_weight_error=error,max_metric_error=metric_error,sequences=len(rows),
        teacher_rows=len(rows)*(len(rows[0])+1),ordinary_tokens=sum(map(len,normal)),oracle_tokens=len(rows)*len(rows[0]))


def own_decision(cells):
    pairs=[]
    for seed in SEEDS:
        selected={(c['pool'],c['mode']):c['metrics'] for c in cells if c['seed']==seed};long=[];short=[]
        for p in ('test_8','test_10'):
            for g in ('distinct','repeated'):
                if g not in selected[p,'native']:continue
                gain=selected[p,'aligned'][g]['oracle_content_exact']-selected[p,'native'][g]['oracle_content_exact']
                long.append(dict(pool=p,group=g,oracle_gain=gain))
        for p in ('train_6','validation_6','test_6'):
            for g in ('distinct','repeated'):
                short.append(selected[p,'aligned'][g]['exact_accuracy']-selected[p,'native'][g]['exact_accuracy'])
        pairs.append(dict(seed=seed,long_groups=long,short_worst_change=min(short),gain_rule=min(x['oracle_gain'] for x in long)>=.1,short_preserved=min(short)>=-.05))
    full=min(c['metrics']['all']['exact_accuracy'] for c in cells if c['mode']=='aligned' and c['pool'] in ('test_8','test_10'))==1
    return dict(paired=pairs,conditional_recovery=all(p['gain_rule'] and p['short_preserved'] for p in pairs),full_long_recovery=full)


def main():
    report=json.loads((DIRECTORY/'report.json').read_text(encoding='utf-8'));assert report['status']=='complete' and not (DIRECTORY/'audit.json').exists()
    assert all(digest(ROOT/p)==h for p,h in report['source_dependencies'].items())
    assert digest(DIRECTORY/'evidence.pt')==report['evidence_sha256'] and digest(DIRECTORY/'metrics.csv')==report['metrics_sha256']
    old=json.loads(OLD.read_text(encoding='utf-8'));saved=torch.load(DIRECTORY/'evidence.pt',weights_only=True)
    expected_keys={(s,p,m) for s in SEEDS for p in POOLS for m in MODES};assert set(saved)==expected_keys
    cells=[];checks=[]
    for seed in SEEDS:
        run=next(r for r in old['runs'] if r['seed']==seed and r['arm']=='rope');model=make_model('rope').eval()
        model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model']);archive=torch.load(ROOT/run['scores'],weights_only=True)
        for p in POOLS:
            rows=old['evaluation_pools'][p];assert rows==report['pools'][p]
            for mode in MODES:
                cell=next(c for c in report['cells'] if (c['seed'],c['pool'],c['mode'])==(seed,p,mode))
                actual,check=audit_cell(model,rows,mode,saved[seed,p,mode],cell)
                if mode=='native':
                    assert float((saved[seed,p,mode]['logits']-archive[p]['logits']).abs().max())<=1e-9
                    assert saved[seed,p,mode]['generated']==archive[p]['generated'] and saved[seed,p,mode]['oracle_outputs']==archive[p]['oracle_outputs']
                cells.append(dict(seed=seed,pool=p,mode=mode,**actual));checks.append(check)
        print(json.dumps(dict(seed=seed,cells=len(POOLS)*len(MODES))),flush=True)
    assert len(cells)==len(expected_keys)==report['expected_cells']==len(report['cells'])
    assert own_decision(cells)==report['decision']==decision(cells)
    assert all(digest(ROOT/p)==h for p,h in report['source_dependencies'].items())
    totals={k:sum(c[k] for c in checks) for k in ('sequences','teacher_rows','ordinary_tokens','oracle_tokens')}
    audit=dict(passed=True,cells_checked=len(cells),**totals,report_sha256=digest(DIRECTORY/'report.json'),
        source_sha256=digest(Path(__file__)),max_logit_weight_error=max(c['max_logit_weight_error'] for c in checks),
        max_metric_error=max(c['max_metric_error'] for c in checks),teacher_argmax_changes=0,ordinary_sequence_changes=0,
        oracle_sequence_changes=0,native_archive_matches=True,prior_hashes_unchanged=True,decision=own_decision(cells))
    write(DIRECTORY/'audit.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()

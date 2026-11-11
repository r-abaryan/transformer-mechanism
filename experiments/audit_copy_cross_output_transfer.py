"""Explicit equations and independent scalar decoding for cross-output transfers."""
import json
import math

import torch

from copy_rope_model import attention
from audit_kv_cache_equivalence import norm
from audit_copy_length_exposure import scalar_metrics
from audit_copy_position_exposure import close


def ff(layer,x):
    return (x@layer.linear1.weight.T+layer.linear1.bias).clamp_min(0)@layer.linear2.weight.T+layer.linear2.bias


def parts(model,content,prefix):
    base=model.base; enc,dec=base.encoder,base.decoder
    ids=torch.cat((content,torch.full((len(content),1),2,dtype=torch.long)),1)
    source=base.source(ids)*math.sqrt(8); target=base.target(prefix)*math.sqrt(8)
    sp,tp=torch.zeros_like(ids,dtype=torch.bool),torch.zeros_like(prefix,dtype=torch.bool)
    ep,dp=torch.arange(ids.shape[1]),torch.arange(prefix.shape[1])
    memory=norm(source+attention(enc.self_attn,source,source,source,sp,ep,ep,reference=True),enc.norm1)
    memory=norm(memory+ff(enc,memory),enc.norm2)
    x=norm(target+attention(dec.self_attn,target,target,target,tp,dp,dp,True,True),dec.norm1)
    cross=attention(dec.multihead_attn,x,memory,memory,sp,dp,ep,reference=True)
    return x,cross


def reference(recipient,donor,source,prefix,mode):
    x,c=parts(recipient,source,prefix); _,d=parts(donor,source,prefix)
    rn,dn=c.square().sum(-1).sqrt(),d.square().sum(-1).sqrt()
    ratio=torch.ones_like(rn);eligible=rn>1e-12;ratio[eligible]=dn[eligible]/rn[eligible]
    stats=torch.stack((rn,dn,ratio,(~eligible).to(rn.dtype)),-1)
    if mode in ('self','transfer'):c=d
    elif mode=='magnitude':c=c*ratio[...,None]
    else:assert mode=='native'
    dec=recipient.base.decoder
    x=norm(x+c,dec.norm2);x=norm(x+ff(dec,x),dec.norm3)
    assert recipient.base.output.bias is None
    return x@recipient.base.output.weight.T,stats


@torch.no_grad()
def audit_cell(recipient,donor,rows,mode,metrics,saved):
    zs=[];generated=[];ts=[];gs=[]
    for begin in range(0,len(rows),64):
        rr=rows[begin:begin+64];source=torch.tensor(rr);prefix=torch.tensor([[1]+r for r in rr])
        z,s=reference(recipient,donor,source,prefix,mode);zs.append(z);ts.append(s.reshape(-1,4))
        histories=[[1] for _ in rr];outputs=[[] for _ in rr];live=[True]*len(rr)
        for _ in range(len(rr[0])+3):
            z,s=reference(recipient,donor,source,torch.tensor(histories),mode)
            gs.append(s[:,-1][torch.tensor(live)])
            for i,values in enumerate(z[:,-1].tolist()):
                token=max(range(len(values)),key=values.__getitem__)
                histories[i].append(token)
                if live[i]:
                    outputs[i].append(token)
                    if token==2:live[i]=False
            if not any(live):break
        generated.extend(outputs)
    z=torch.cat(zs)
    torch.testing.assert_close(z,saved['logits'],atol=1e-9,rtol=0)
    assert generated==saved['generated'], 'Generated token mismatch'
    torch.testing.assert_close(torch.cat(ts),saved['teacher_stats'],atol=1e-9,rtol=0)
    torch.testing.assert_close(torch.cat(gs),saved['generated_stats'],atol=1e-9,rtol=0)
    m=scalar_metrics(z.tolist(),rows,generated)
    close({g:m.get(g) for g in ('all','distinct','repeated')},metrics)
    return float((z-saved['logits']).abs().max())


def main():
    from copy_cross_output_transfer import ROOT,OUT,ARMS,MODES,load,digest,write,decisions
    plan=json.loads((OUT/'plan.json').read_text());report=json.loads((OUT/'report.json').read_text())
    assert not (OUT/'audit.json').exists()
    assert report['plan_sha256']==digest(OUT/'plan.json')
    assert report['evidence_sha256']==digest(OUT/'evidence.pt')
    assert report['metrics_sha256']==digest(OUT/'metrics.csv')
    assert plan['preflight_sha256']==digest(OUT/'preflight.json')
    assert all(digest(ROOT/p)==h for p,h in plan['protected_hashes'].items())
    models={a:load(ROOT/f'results/copy_learning_trajectory/3767_{a}_6000.pt') for a in ARMS}
    data=json.loads((ROOT/'results/copy_position_sites/data.json').read_text())['pools']
    saved=torch.load(OUT/'evidence.pt',weights_only=True);maximum=0.;cells=0
    for arm in ARMS:
        for mode in MODES:
            donor=models[arm] if mode=='self' else models[ARMS[1-ARMS.index(arm)]]
            for pool in plan['pools']:
                maximum=max(maximum,audit_cell(models[arm],donor,data[pool],mode,
                    report['metrics'][arm][mode][pool],saved[arm][mode][pool]))
                cells+=1
    assert report['decisions']==decisions(report['metrics'])
    import csv
    with (OUT/'metrics.csv').open(newline='',encoding='utf-8') as f: csvrows=list(csv.DictReader(f))
    for row in csvrows:
        expected=report['metrics'][row['recipient']][row['mode']][row['pool']][row['group']]
        for k,v in expected.items():assert abs(float(row[k])-v)<1e-12
    assert len(csvrows)==sum(m is not None for a in report['metrics'].values() for mm in a.values() for gs in mm.values() for m in gs.values())
    write(OUT/'audit.json',dict(passed=True,cells=cells,csv_rows=len(csvrows),max_logit_error=maximum,
        report_sha256=digest(OUT/'report.json'),protected_files=len(plan['protected_hashes'])))
    print('Persisted audit passed:',cells,'cells',flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);main()

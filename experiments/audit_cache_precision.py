"""Independent cache-read arithmetic, quantization and saved-record audit."""
import itertools
import json
import math
import struct
from pathlib import Path

import torch
from torch.nn import functional as F
from audit_kv_cache_equivalence import norm
from audit_mqa_training import independent
from mqa_training import ROOT, SEEDS, model_new
from position_distractors import digest

MODES=('float64','fp32','fp16','int8','int8_k','int8_v')


def rotate_matrix(t,position):
    angles=torch.tensor([1.,.1,.01,.001],dtype=torch.float64)*position
    c,s=angles.cos(),angles.sin();matrices=torch.stack((c,-s,s,c),-1).reshape(4,2,2)
    return torch.einsum('rij,bhtrj->bhtri',matrices,t.reshape(*t.shape[:-1],4,2)).flatten(-2)


def rounded(t,mode,side):
    mode=('int8' if mode[-1]==side else 'float64') if mode in ('int8_k','int8_v') else mode
    if mode=='float64':return t
    if mode in ('fp32','fp16'):return t.to(torch.float32 if mode=='fp32' else torch.float16).double()
    maximum=t.abs().max(dim=-1,keepdim=True).values
    scale=(maximum/127).to(torch.float32).double();scale=torch.where(maximum==0,torch.ones_like(scale),scale)
    z=t/scale;lo=z.floor();fraction=z-lo
    nearest=lo+((fraction>.5)|((fraction==.5)&(lo.remainder(2)!=0))).double()
    return nearest.minimum(torch.tensor(127.)).maximum(torch.tensor(-127.))*scale


def replay(model,ids,mode):
    past=[([],[]) for _ in model.blocks];outputs=[]
    for i in range(ids.shape[1]):
        x=model.embedding.weight[ids[:,i:i+1]]
        for l,b in enumerate(model.blocks):
            z=norm(x,b.norm1);m=b.attn
            q,k,v=[z@getattr(m,name+'_weight').T+getattr(m,name+'_bias') for name in ('q','k','v')]
            q=q.reshape(len(ids),1,2,8).permute(0,2,1,3)
            k,v=[t.reshape(len(ids),1,model.groups,8).permute(0,2,1,3) for t in (k,v)]
            q,k=rotate_matrix(q,i+4),rotate_matrix(k,i+4)
            past[l][0].append(rounded(k,mode,'k'));past[l][1].append(rounded(v,mode,'v'))
            k,v=[torch.cat(parts,2).expand(len(ids),2,i+1,8) for parts in past[l]]
            y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.).permute(0,2,1,3).reshape(len(ids),1,16)
            x=x+y@m.out_proj.weight.T+m.out_proj.bias
            h=norm(x,b.norm2)@b.linear1.weight.T+b.linear1.bias
            x=x+h.clamp_min(0)@b.linear2.weight.T+b.linear2.bias
        outputs.append(norm(x,model.final_norm)@model.output.weight.T)
    return torch.cat(outputs,1)


def fixtures(data,n):
    originals=data['datasets'][f'test_{n}'][:144];rows=[]
    for index,(ids,label) in enumerate(originals):
        pairs=list(zip(ids[:-2:2],ids[1:-2:2]));signature=tuple(tuple(p) for p in data['families'][f'test_{n}'][index//36])
        candidates=[tuple(sorted((k,12+(v-12+s)%6) for k,v in pairs)) for s in range(6)]
        assert min(candidates)==signature
        for q in [ids[-1]]+sorted({k for k,v in pairs}-{ids[-1]}):
            y=next(v for k,v in reversed(pairs) if k==q);rows.append((ids[:-1]+[q],y))
    return torch.tensor([x for x,y in rows]),torch.tensor([y for x,y in rows])


def codec_witness(w):
    vectors=w['codec_input'];checks=0
    for item in w['codecs']:
        mode=item['mode'];outputs=[];codes=[];scales=[]
        for vector in vectors:
            if mode=='int8':
                maximum=max(abs(x) for x in vector);scale=struct.unpack('f',struct.pack('f',maximum/127 if maximum else 1.))[0]
                code=[max(-127,min(127,round(x/scale))) for x in vector]
                outputs.append([x*scale for x in code]);codes.append(code);scales.append([scale])
            else:
                fmt={'float64':'d','fp32':'f','fp16':'e'}[mode]
                outputs.append([struct.unpack(fmt,struct.pack(fmt,x))[0] for x in vector])
        assert outputs==item['decoded']
        assert item['encoded']==([codes,scales] if mode=='int8' else [outputs])
        assert item['bytes']==3*({'float64':64,'fp32':32,'fp16':16,'int8':12}[mode]);checks+=24
    witness=w['attention'];q=witness['q'][0];k=witness['k'];v=witness['v']
    s=[sum(a*b for a,b in zip(q,row))/math.sqrt(8) for row in k];p=[math.exp(x-max(s)) for x in s]
    result=sum(a*b[0] for a,b in zip(p,v))/sum(p)
    assert abs(result-witness['reference_output'])<1e-15 and result>.5
    assert witness['quantized_output']==.5 and witness['reference_argmax']==1 and witness['quantized_argmax']==0
    return checks


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rp=ROOT/'results/cache_precision.json';ap=ROOT/'results/cache_precision_audit.json'
    assert not ap.exists(),'Preserve completed audit.'
    r=json.loads(rp.read_text());assert r['phase']=='complete'
    for name,h in r['provenance'].items():assert digest(ROOT/name)==h
    pp=ROOT/'results/mqa_training.json';assert digest(pp)==r['prior_sha256'];prior=json.loads(pp.read_text())
    assert digest(ROOT/'results/mqa_training_audit.json')==r['prior_audit_sha256']
    ip=ROOT/'results/mqa_training_data.json';assert digest(ip)==r['input_sha256'];data=json.loads(ip.read_text())
    tp=ROOT/'results/cache_precision_tensors.pt';assert digest(tp)==r['tensors_sha256'];saved=torch.load(tp,weights_only=True)
    assert len(saved)==len(r['cases'])==180
    keys={(c['seed'],c['group'],c['pairs'],c['mode']) for c in r['cases']}
    assert keys==set(itertools.product(SEEDS,('mha','mqa'),(3,4,6),MODES))
    witness_checks=codec_witness(r['witnesses']);errors=[];models={};references={}
    with torch.inference_mode():
        for c,s in zip(r['cases'],saved):
            key=(c['seed'],c['group']);n=c['pairs'];mode=c['mode']
            assert all(c[k]==s[k] for k in ('seed','group','pairs','mode'))
            run=next(x for x in prior['runs'] if (x['seed'],x['group'])==key)
            cp=ROOT/c['checkpoint'];assert c['checkpoint']==run['checkpoint'] and digest(cp)==c['checkpoint_sha256']==run['checkpoint_sha256']
            if key not in models:
                model=model_new(c['group']).eval();model.load_state_dict(torch.load(cp,weights_only=True)['model']);models[key]=model
            model=models[key];ids,labels=fixtures(data,n)
            assert torch.equal(ids,s['ids']) and torch.equal(labels,s['labels'])
            if (key,n) not in references:references[key,n]=independent(model,ids)
            reference=references[key,n];scores=replay(model,ids,mode)
            error=max(float((reference[:,-1]-s['reference']).abs().max()),float((scores[:,-1]-s['scores']).abs().max()))
            assert error<=1e-10;errors.append(error)
            pa,pb=s['reference'].argmax(-1),s['scores'].argmax(-1)
            e=(s['reference']-s['scores']).abs().max(-1).values;assert torch.equal(e,s['rowwise_error'])
            gaps=s['reference'].sort(-1,descending=True).values;gap=gaps[:,0]-gaps[:,1]
            m=c['metrics']
            counts=dict(final_changes=int((pa!=pb).sum()),reference_correct=int((pa==labels).sum()),candidate_correct=int((pb==labels).sum()),
                        damaged=int(((pa==labels)&(pb!=labels)).sum()),repaired=int(((pa!=labels)&(pb==labels)).sum()),certified_rows=int((gap>2*e).sum()),rows=len(ids))
            assert all(m[k]==v for k,v in counts.items());assert m['max_final_error']==float(e.max())
            assert abs(m['max_all_token_error']-float((reference-scores).abs().max()))<=2e-10
            assert m['all_token_argmax_changes']==int((reference.argmax(-1)!=scores.argmax(-1)).sum())
            assert not ((gap>2*e)&(pa!=pb)).any()
            widths={'float64':64,'fp32':32,'fp16':16,'int8':12}
            km=mode if mode not in ('int8_k','int8_v') else ('int8' if mode=='int8_k' else 'float64')
            vm=mode if mode not in ('int8_k','int8_v') else ('int8' if mode=='int8_v' else 'float64')
            assert c['payload_bytes']==len(ids)*2*model.groups*ids.shape[1]*(widths[km]+widths[vm])
            if mode=='int8_v':print(f"Audited {key} / {n} pairs",flush=True)
    decision=dict(control_passed=all(c['metrics']['max_all_token_error']<=1e-10 and c['metrics']['all_token_argmax_changes']==0 for c in r['cases'] if c['mode']=='float64'),
                  preservation={m:all(c['metrics']['final_changes']==0 for c in r['cases'] if c['mode']==m) for m in MODES[1:]})
    assert decision==r['decision'] and decision['control_passed']
    result=dict(passed=True,cells=len(errors),max_logit_disagreement=max(errors),codec_scalar_checks=witness_checks,report_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    ap.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

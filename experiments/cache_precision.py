"""Bounded storage-only approximation study on frozen trained recall models."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from mqa_training import ROOT, SEEDS, model_new, forward, expand_rows
from positional_comparison import rotate
from position_distractors import digest

MODES=('float64','fp32','fp16','int8','int8_k','int8_v')


def side_mode(mode,side):
    return ('int8' if mode.endswith('_'+side) else 'float64') if mode in ('int8_k','int8_v') else mode


def encode(x,mode):
    if mode!='int8':return (x.to({'float64':torch.float64,'fp32':torch.float32,'fp16':torch.float16}[mode]),)
    maximum=x.abs().amax(-1,keepdim=True)
    scale=torch.where(maximum==0,torch.ones_like(maximum),maximum/127).float()
    assert torch.isfinite(scale).all() and (scale>0).all()
    codes=(x/scale.double()).round().clamp(-127,127).to(torch.int8)
    assert ((x-codes.double()*scale.double()).abs()<=scale.double()/2+1e-12).all()
    return codes,scale


def decode(parts):
    return parts[0].double() if len(parts)==1 else parts[0].double()*parts[1].double()


def payload(history):
    return sum(t.numel()*t.element_size() for layer in history for pair in layer for parts in pair for t in parts)


def expected_bytes(batch,length,groups,mode):
    sizes={'float64':64,'fp32':32,'fp16':16,'int8':12}
    return batch*2*groups*length*sum(sizes[side_mode(mode,s)] for s in ('k','v'))


def cached(model,ids,mode):
    history=[[] for _ in model.blocks];outputs=[]
    for cursor in range(ids.shape[1]):
        x=model.embedding(ids[:,cursor:cursor+1]);positions=torch.tensor([cursor+4])
        for layer,block in enumerate(model.blocks):
            z=block.norm1(x);m=block.attn
            q,k,v=[F.linear(z,getattr(m,t+'_weight'),getattr(m,t+'_bias')) for t in ('q','k','v')]
            q=q.reshape(len(ids),1,2,8).transpose(1,2)
            k,v=[t.reshape(len(ids),1,model.groups,8).transpose(1,2) for t in (k,v)]
            q,k=rotate(q,positions),rotate(k,positions)
            history[layer].append((encode(k,side_mode(mode,'k')),encode(v,side_mode(mode,'v'))))
            k,v=[torch.cat([decode(pair[i]) for pair in history[layer]],2).repeat_interleave(2//model.groups,1) for i in (0,1)]
            weights=(q@k.transpose(-1,-2)/(8**.5)).softmax(-1)
            y=(weights@v).transpose(1,2).reshape(len(ids),1,16)
            x=x+m.out_proj(y)
            x=x+block.linear2(F.relu(block.linear1(block.norm2(x))))
        outputs.append(model.output(model.final_norm(x)))
    return torch.cat(outputs,1),payload(history)


def metrics(reference,scores,labels):
    a,b=reference[:,-1],scores[:,-1];pa,pb=a.argmax(-1),b.argmax(-1)
    error=(a-b).abs().amax(-1);gap=a.topk(2,-1).values.diff(dim=-1).abs().squeeze(-1)
    return dict(max_all_token_error=float((reference-scores).abs().max()),
                all_token_argmax_changes=int((reference.argmax(-1)!=scores.argmax(-1)).sum()),
                max_final_error=float(error.max()),final_changes=int((pa!=pb).sum()),
                reference_correct=int((pa==labels).sum()),candidate_correct=int((pb==labels).sum()),
                damaged=int(((pa==labels)&(pb!=labels)).sum()),repaired=int(((pa!=labels)&(pb==labels)).sum()),
                certified_rows=int((gap>2*error).sum()),rows=len(labels))


def decision(cases):
    return dict(control_passed=all(c['metrics']['max_all_token_error']<=1e-10 and c['metrics']['all_token_argmax_changes']==0 for c in cases if c['mode']=='float64'),
                preservation={m:all(c['metrics']['final_changes']==0 for c in cases if c['mode']==m) for m in MODES[1:]})


def witnesses():
    x=torch.tensor([[0.]*8,[127.,.5,1.5,2.5,-.5,-1.5,-2.5,-127.],[1.,.001,0.,0.,0.,0.,0.,0.]],dtype=torch.float64)
    tests=[]
    for mode in ('float64','fp32','fp16','int8'):
        parts=encode(x,mode);tests.append(dict(mode=mode,encoded=[p.tolist() for p in parts],decoded=decode(parts).tolist(),bytes=sum(p.numel()*p.element_size() for p in parts)))
    q=torch.tensor([[0.,100.,0.,0.,0.,0.,0.,0.]],dtype=torch.float64)
    k=torch.tensor([[1.,0.,0.,0.,0.,0.,0.,0.],[1.,.001,0.,0.,0.,0.,0.,0.]],dtype=torch.float64)
    v=torch.tensor([[0.],[1.]],dtype=torch.float64)
    y=((q@k.T/(8**.5)).softmax(-1)@v).item()
    rounded=((q@decode(encode(k,'int8')).T/(8**.5)).softmax(-1)@v).item()
    assert y>.5 and rounded==.5
    return dict(codec_input=x.tolist(),codecs=tests,attention=dict(q=q.tolist(),k=k.tolist(),v=v.tolist(),reference_output=y,quantized_output=rounded,reference_logits=[.5-y,y-.5],quantized_logits=[.5-rounded,rounded-.5],reference_argmax=1,quantized_argmax=0))


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rp=ROOT/'results/cache_precision.json';tp=ROOT/'results/cache_precision_tensors.pt'
    assert not rp.exists() and not tp.exists(),'Preserve opened experiment artifacts.'
    prior_path=ROOT/'results/mqa_training.json';prior=json.loads(prior_path.read_text())
    audit_path=ROOT/'results/mqa_training_audit.json';audit=json.loads(audit_path.read_text())
    assert prior['phase']=='complete' and audit['passed'] and audit['report_sha256']==digest(prior_path)
    for name,h in prior['provenance'].items():assert digest(ROOT/('docs/mqa-training-protocol.md' if name=='protocol' else 'experiments/'+name))==h
    ip=ROOT/'results/mqa_training_data.json';assert digest(ip)==prior['input_sha256'];data=json.loads(ip.read_text())
    frozen=('experiments/cache_precision.py','experiments/audit_cache_precision.py','docs/cache-precision-protocol.md')
    report=dict(phase='running',torch=torch.__version__,device='cpu',compute_dtype='float64',threads=1,
                prior_sha256=digest(prior_path),prior_audit_sha256=digest(audit_path),input_sha256=digest(ip),
                provenance={p:digest(ROOT/p) for p in frozen},witnesses=witnesses(),cases=[])
    saved=[]
    with torch.inference_mode():
        for run in prior['runs']:
            cp=ROOT/run['checkpoint'];assert run['status']=='trained' and digest(cp)==run['checkpoint_sha256']
            model=model_new(run['group']).eval();model.load_state_dict(torch.load(cp,weights_only=True)['model'])
            for n in (3,4,6):
                rows=expand_rows(data['datasets'][f'test_{n}'][:144]);ids=torch.tensor([x for x,y in rows]);labels=torch.tensor([y for x,y in rows])
                reference=forward(model,ids)
                for mode in MODES:
                    scores,size=cached(model,ids,mode);assert size==expected_bytes(len(ids),ids.shape[1],model.groups,mode)
                    metric=metrics(reference,scores,labels)
                    case=dict(seed=run['seed'],group=run['group'],pairs=n,mode=mode,checkpoint=run['checkpoint'],checkpoint_sha256=run['checkpoint_sha256'],payload_bytes=size,metrics=metric)
                    report['cases'].append(case)
                    saved.append(dict(seed=run['seed'],group=run['group'],pairs=n,mode=mode,ids=ids,labels=labels,reference=reference[:,-1],scores=scores[:,-1],rowwise_error=(reference[:,-1]-scores[:,-1]).abs().amax(-1)))
            print(f"Checked {run['seed']} / {run['group']}",flush=True)
    assert len(report['cases'])==180
    torch.save(saved,tp);report.update(phase='complete',tensors_sha256=digest(tp),decision=decision(report['cases']))
    rp.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['decision']))
    if not report['decision']['control_passed']:raise SystemExit('Unrounded cache control failed; evidence retained.')


if __name__=='__main__':main()

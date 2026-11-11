"""Frozen native-Windows full-decoder benchmark, explicit compact KV buffers."""
import argparse
import gc
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from mqa_training import ROOT,model_new,forward
from position_distractors import digest


def rotate_fp(t,c,s):
    a,b=t[...,0::2],t[...,1::2]
    return torch.stack((a*c-b*s,a*s+b*c),-1).flatten(-2)


def chunk(model,ids,cache,start,c,s):
    n=ids.shape[1];x=model.embedding(ids)
    positions=torch.arange(start,start+n,device=ids.device)
    mask=torch.arange(start+n,device=ids.device)[None,:]<=positions[:,None]
    for i,block in enumerate(model.blocks):
        z=block.norm1(x);m=block.attn
        q,k,v=[F.linear(z,getattr(m,t+'_weight'),getattr(m,t+'_bias')) for t in ('q','k','v')]
        q=q.reshape(len(ids),n,2,8).transpose(1,2)
        k,v=[t.reshape(len(ids),n,model.groups,8).transpose(1,2) for t in (k,v)]
        q,k=rotate_fp(q,c[start:start+n],s[start:start+n]),rotate_fp(k,c[start:start+n],s[start:start+n])
        cache[i][0][:,:,start:start+n].copy_(k);cache[i][1][:,:,start:start+n].copy_(v)
        k,v=[t[:,:,:start+n].expand(len(ids),2,start+n,8) for t in cache[i]]
        scores=q@k.transpose(-1,-2)/(8**.5)
        # Single new-token queries see every occupied cache slot.
        if n>1:scores=scores.masked_fill(~mask,-torch.inf)
        y=(scores.softmax(-1)@v).transpose(1,2).reshape(len(ids),n,16)
        x=x+m.out_proj(y)
        x=x+block.linear2(F.relu(block.linear1(block.norm2(x))))
    return model.output(model.final_norm(x))


def decode(model,ids,cache,prefix,c,s):
    return torch.cat([chunk(model,ids[:,i:i+1],cache,i,c,s) for i in range(prefix,prefix+8)],1)


def accuracy(output,ref):
    difference=output.double()-ref
    return dict(max_absolute_error=float(difference.abs().max()),
                relative_l2_error=float(difference.norm()/ref.norm().clamp_min(1e-12)),
                argmax_differences=int((output.argmax(-1)!=ref.argmax(-1)).sum()))


def paths(worker):
    return ROOT/f'results/mqa_decoding_worker{worker}.json',ROOT/f'results/mqa_decoding_worker{worker}_tensors.pt'


def worker(index):
    assert torch.cuda.is_available()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rp,tp=paths(index);assert not rp.exists() and not tp.exists()
    prior_path=ROOT/'results/mqa_training.json';prior=json.loads(prior_path.read_text());assert prior['phase']=='complete'
    for name,h in prior['provenance'].items():assert digest(ROOT/('docs/mqa-training-protocol.md' if name=='protocol' else 'experiments/'+name))==h
    audit_path=ROOT/'results/mqa_training_audit.json';audit=json.loads(audit_path.read_text())
    assert audit['passed'] and audit['report_sha256']==digest(prior_path)
    frozen=('experiments/mqa_decoding.py','experiments/audit_mqa_decoding.py','docs/mqa-decoding-protocol.md')
    report=dict(phase='running',worker=index,torch=torch.__version__,cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0),platform=platform.platform(),dtype='float32',tf32=False,
                prior_sha256=digest(prior_path),prior_audit_sha256=digest(audit_path),
                provenance={p:digest(ROOT/p) for p in frozen},
                smi_before=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,cells=[])
    saved=[];shapes=[(b,p) for b in (1,16) for p in (14,128,512)]
    if index%2:shapes.reverse()
    groups=('mqa','mha') if index%2 else ('mha','mqa')
    with torch.inference_mode():
        for batch,prefix in shapes:
            rng=torch.Generator().manual_seed(20270109+prefix)
            cpu_ids=torch.randint(19,(batch,prefix+8),generator=rng);ids=cpu_ids.cuda()
            for group in groups:
                run=next(r for r in prior['runs'] if r['seed']==1411 and r['group']==group)
                checkpoint=ROOT/run['checkpoint'];assert digest(checkpoint)==run['checkpoint_sha256']
                cpu=model_new(group).eval();cpu.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
                cpu.float().double() # Reference the FP32-rounded deployment weights.
                ref=forward(cpu,cpu_ids)[:,prefix:].clone()
                model=model_new(group).eval().float().cuda();model.load_state_dict(cpu.state_dict())
                cache=[tuple(torch.empty(batch,model.groups,prefix+8,8,device='cuda') for _ in range(2)) for layer in range(2)]
                angles=(torch.arange(prefix+8,device='cuda',dtype=torch.float64)+4)[:,None]*torch.tensor([1.,.1,.01,.001],device='cuda',dtype=torch.float64)
                c,s=angles.cos().float(),angles.sin().float()
                prefill=chunk(model,ids[:,:prefix],cache,0,c,s);del prefill
                output=decode(model,ids,cache,prefix,c,s);torch.cuda.synchronize()
                gpu_output=output.cpu();metrics=accuracy(gpu_output,ref);del output
                payload=sum(t.untyped_storage().nbytes() for pair in cache for t in pair)
                assert payload==batch*2*model.groups*(prefix+8)*16*4
                cell=dict(batch=batch,prefix=prefix,group=group,checkpoint=run['checkpoint'],checkpoint_sha256=run['checkpoint_sha256'],
                          correctness=metrics,cache_shapes=[list(t.shape) for pair in cache for t in pair],cache_bytes=payload)
                passed=metrics['max_absolute_error']<=1e-4 and metrics['relative_l2_error']<=1e-4
                if passed:
                    for _ in range(3):out=decode(model,ids,cache,prefix,c,s);del out
                    torch.cuda.synchronize();gc.collect()
                    cell.update(baseline_allocated=torch.cuda.memory_allocated(),baseline_reserved=torch.cuda.memory_reserved())
                    torch.cuda.reset_peak_memory_stats();events=[];wall=[]
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    for _ in range(10):
                        torch.cuda.synchronize();tick=time.perf_counter();begin.record()
                        out=decode(model,ids,cache,prefix,c,s);end.record();torch.cuda.synchronize()
                        wall.append((time.perf_counter()-tick)*1000);events.append(begin.elapsed_time(end));del out
                    cell.update(status='passed',event_ms=events,wall_ms=wall,
                                event_median_ms=statistics.median(events),wall_median_ms=statistics.median(wall),
                                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
                else:cell.update(status='failed',reason='Fixed numerical tolerance failed; timing skipped.')
                saved.append(dict(batch=batch,prefix=prefix,group=group,ids=cpu_ids,reference=ref,output=gpu_output))
                report['cells'].append(cell);rp.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(dict(worker=index,batch=batch,prefix=prefix,group=group,status=cell['status'],
                                      correctness=metrics,event_ms=cell.get('event_median_ms'),wall_ms=cell.get('wall_median_ms'))),flush=True)
                del model,cache,c,s,cpu,ref,gpu_output;gc.collect();torch.cuda.empty_cache()
            del ids
    torch.save(saved,tp)
    report.update(phase='complete',tensors_sha256=digest(tp),smi_after=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout)
    rp.write_text(json.dumps(report,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,choices=(0,1,2));args=parser.parse_args()
    if args.worker is not None:worker(args.worker);return
    target=ROOT/'results/mqa_decoding.json';assert not target.exists()
    for index in range(3):
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(index)],check=True)
    reports=[json.loads(paths(i)[0].read_text()) for i in range(3)]
    summary=[]
    for batch in (1,16):
        for prefix in (14,128,512):
            rows=[]
            for index,r in enumerate(reports):
                pair=[next(c for c in r['cells'] if c['batch']==batch and c['prefix']==prefix and c['group']==g) for g in ('mha','mqa')]
                if all(c['status']=='passed' for c in pair):
                    b,m=pair;rows.append(dict(worker=index,event_ratio=m['event_median_ms']/b['event_median_ms'],wall_ratio=m['wall_median_ms']/b['wall_median_ms']))
            summary.append(dict(batch=batch,prefix=prefix,process_ratios=rows,
                                speed_claim_pass=len(rows)==3 and all(x['event_ratio']<=.9 and x['wall_ratio']<=.9 for x in rows)))
    result=dict(phase='complete',workers={str(i):digest(paths(i)[0]) for i in range(3)},summary=summary)
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

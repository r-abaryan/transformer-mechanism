"""Frozen forward-plus-backward matrix using previously verified forward functions."""
import gc
import json
import platform
import subprocess
import time
import torch
from native_attention import ROOT, digest, dense, efficient, cudnn


def errors(actual, reference):
    delta=actual.double()-reference
    return dict(max_absolute=delta.abs().max().item(),
                relative_l2=(delta.norm()/reference.norm().clamp_min(1e-12)).item())


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    rp=ROOT/'results/native_backward.json'; tp=ROOT/'results/native_backward_tensors.pt'
    assert not rp.exists() and not tp.exists()
    paths=['experiments/native_backward.py','experiments/native_attention.py',
           'experiments/gpu_attention.py','docs/native-backward-protocol.md']
    report=dict(hashes={p:digest(ROOT/p) for p in paths},torch=torch.__version__,
                cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),platform=platform.platform(),
                smi_before=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,cells=[])
    saved={}
    methods={'dense':dense,'efficient':efficient,'cudnn':cudnn}
    for index,n in enumerate((512,1024,2048,4096)):
        torch.manual_seed(919)
        cpu=[torch.randn(n,64).half() for _ in range(3)]
        g=torch.randn(n,64,generator=torch.Generator().manual_seed(929)).half()
        q,k,v=[a.double().requires_grad_() for a in cpu]
        score=q@k.T/8
        score=score.masked_fill(torch.ones(n,n,dtype=torch.bool).triu(1),float('-inf'))
        ref=score.softmax(-1)@v
        ref_grad=torch.autograd.grad(ref,(q,k,v),g.double())
        ref=ref.detach()
        del q,k,v,score
        inputs=[a.cuda().requires_grad_() for a in cpu]; upstream=g.cuda()
        saved[str(n)]=dict(inputs=cpu,upstream=g,reference=ref,reference_gradients=ref_grad,outputs={})
        names=list(methods); names=names[index%3:]+names[:index%3]
        for name in names:
            row=dict(n=n,method=name)
            def step():
                out=methods[name](*inputs)
                grads=torch.autograd.grad(out,inputs,upstream)
                return out,grads
            try:
                for _ in range(5):
                    out,grads=step(); del out,grads
                torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
                row.update(baseline_allocated=torch.cuda.memory_allocated(),baseline_reserved=torch.cuda.memory_reserved())
                torch.cuda.reset_peak_memory_stats()
                events=[]; walls=[]
                for _ in range(10):
                    begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(); tick=time.perf_counter(); begin.record()
                    out,grads=step()
                    end.record(); torch.cuda.synchronize()
                    walls.append((time.perf_counter()-tick)*1000); events.append(begin.elapsed_time(end))
                    del out,grads
                row.update(event_ms=events,wall_ms=walls,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
                out,grads=step()
                out=out.detach().cpu(); grads=[a.cpu() for a in grads]
                output_error=errors(out,ref); gradient_errors=[errors(a,b) for a,b in zip(grads,ref_grad)]
                passed=bool(torch.isfinite(out).all()) and output_error['max_absolute']<=.005
                passed=passed and all(bool(torch.isfinite(a).all()) for a in grads)
                passed=passed and all(a['max_absolute']<=.02 and a['relative_l2']<=.005 for a in gradient_errors)
                row.update(status='passed' if passed else 'numerical_failure',output_error=output_error,gradient_errors=gradient_errors)
                saved[str(n)]['outputs'][name]=dict(output=out,gradients=grads)
                del out,grads
            except RuntimeError as exc:
                row.update(status='runtime_failure',error=str(exc))
                gc.collect(); torch.cuda.empty_cache()
            report['cells'].append(row)
            rp.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row),flush=True)
        del inputs,upstream
    torch.save(saved,tp)
    report.update(complete=True,tensors_sha256=digest(tp),smi_after=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout)
    rp.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()

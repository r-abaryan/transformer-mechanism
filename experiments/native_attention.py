"""Bounded CUDA attention comparison with explicit backend and allocator metrics."""
import gc
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

ROOT=Path(__file__).resolve().parents[1]


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


from gpu_attention import dense


def native(q,k,v,backend):
    with sdpa_kernel(backend):
        flags=[torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled()]
        expected=[False, backend==SDPBackend.EFFICIENT_ATTENTION, False, backend==SDPBackend.CUDNN_ATTENTION]
        assert flags==expected
        return F.scaled_dot_product_attention(q[None,None],k[None,None],v[None,None],dropout_p=0.,is_causal=True)[0,0]


def efficient(q,k,v): return native(q,k,v,SDPBackend.EFFICIENT_ATTENTION)
def cudnn(q,k,v): return native(q,k,v,SDPBackend.CUDNN_ATTENTION)


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    rp=ROOT/'results/native_attention.json'; dp=ROOT/'results/native_attention_tensors.pt'; assert not rp.exists() and not dp.exists()
    device=torch.cuda.get_device_properties(0)
    r=dict(source_sha256=digest(__file__),dense_source_sha256=digest(ROOT/'experiments/gpu_attention.py'),protocol_sha256=digest(ROOT/'docs/native-attention-protocol.md'),torch=torch.__version__,cuda=torch.version.cuda,
           device=device.name,total_memory=device.total_memory,capability=list(torch.cuda.get_device_capability()),platform=platform.platform(),
           smi_before=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,cells=[])
    tensors={}; methods={'dense':dense,'efficient':efficient,'cudnn':cudnn}
    with torch.inference_mode():
        for index,n in enumerate((512,1024,2048,4096)):
            torch.manual_seed(919); cpu=[torch.randn(n,64).half() for _ in range(3)]; q,k,v=[t.cuda() for t in cpu]
            a,b,c=[t.double() for t in cpu]; scores=a@b.T/8; scores.masked_fill_(torch.ones(n,n,dtype=torch.bool).triu(1),float('-inf')); reference=scores.softmax(-1)@c
            tensors[str(n)]=dict(inputs=cpu,reference=reference,outputs={})
            names=list(methods); names=names[index%3:]+names[:index%3]
            for name in names:
                fn=methods[name]; row=dict(n=n,method=name,status='running')
                try:
                    for _ in range(5): out=fn(q,k,v); del out
                    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
                    base=torch.cuda.memory_allocated(); reserved=torch.cuda.memory_reserved(); torch.cuda.reset_peak_memory_stats()
                    gpu=[]; wall=[]
                    for _ in range(10):
                        begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize(); tick=time.perf_counter(); begin.record(); out=fn(q,k,v); end.record(); torch.cuda.synchronize()
                        wall.append((time.perf_counter()-tick)*1000); gpu.append(begin.elapsed_time(end))
                        del out
                    row.update(event_ms=gpu,wall_ms=wall,baseline_allocated=base,baseline_reserved=reserved,
                               peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
                    # Separate correctness call after measurement counters have been read.
                    out=fn(q,k,v).cpu(); error=(out.double()-reference).abs().max().item()
                    assert torch.isfinite(out).all() and error<=.005
                    row.update(status='passed',max_absolute_error=error); tensors[str(n)]['outputs'][name]=out; del out
                except RuntimeError as exc:
                    row.update(status='failed',error=str(exc)); gc.collect(); torch.cuda.empty_cache()
                r['cells'].append(row); rp.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(row),flush=True)
            del q,k,v
        torch.save(tensors,dp)
    r.update(complete=True,tensors_sha256=digest(dp),smi_after=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout)
    rp.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__': main()

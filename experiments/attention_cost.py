"""Isolated Windows CPU attention timing and process working-set measurements."""
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory():
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True); psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
    counters=Counters(); counters.cb=ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb): raise ctypes.WinError(ctypes.get_last_error())
    return dict(working_set=counters.WorkingSetSize,peak_working_set=counters.PeakWorkingSetSize)


def dense(q,k,v):
    n=len(q); s=q@k.T/math.sqrt(q.shape[1]); s.masked_fill_(torch.ones(n,n,dtype=torch.bool).triu(1),float('-inf'))
    return s.softmax(-1)@v


def tiled(q,k,v):
    n,d=q.shape; out=torch.empty_like(v)
    for i in range(0,n,128):
        qi=q[i:i+128]; count=len(qi); m=torch.full((count,1),float('-inf'),dtype=q.dtype); l=torch.zeros_like(m); u=torch.zeros(count,v.shape[1],dtype=v.dtype)
        for j in range(0,i+count,128):
            s=qi@k[j:j+128].T/math.sqrt(d)
            mask=torch.arange(j,min(j+128,n))[None,:]>torch.arange(i,i+count)[:,None]
            s.masked_fill_(mask,float('-inf')); new=torch.maximum(m,s.max(-1,keepdim=True).values)
            scale=(m-new).exp(); p=(s-new).exp(); u=scale*u+p@v[j:j+128]; l=scale*l+p.sum(-1,keepdim=True); m=new
        out[i:i+count]=u/l
    return out


def worker(n,seed,method):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(seed)
    q,k,v=[torch.randn(n,64,dtype=torch.float64) for _ in range(3)]
    # Hash before memory baseline; JSON allocations are also part of process history.
    sha=hashlib.sha256(json.dumps([t.tolist() for t in (q,k,v)]).encode()).hexdigest()
    before=memory(); fn=dense if method=='dense' else tiled; times=[]
    with torch.inference_mode():
        output=fn(q,k,v); del output
        for _ in range(3):
            tick=time.perf_counter(); output=fn(q,k,v); times.append(time.perf_counter()-tick)
            if len(times)<3: del output
    after=memory(); assert torch.isfinite(output).all()
    path=ROOT/f'results/attention_cost_outputs/{method}_{n}_{seed}.pt'; torch.save(output,path)
    print(json.dumps(dict(n=n,seed=seed,method=method,input_sha256=sha,times_seconds=times,before=before,after=after,
                         peak_growth=after['peak_working_set']-before['peak_working_set'],output=str(path.relative_to(ROOT)),output_sha256=digest(path))))


def main():
    if len(sys.argv)>1: worker(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]); return
    rp=ROOT/'results/attention_cost.json'; assert not rp.exists()
    directory=ROOT/'results/attention_cost_outputs'; directory.mkdir(exist_ok=True)
    r=dict(platform=platform.platform(),processor=platform.processor(),torch=torch.__version__,dtype='float64',threads=1,
           source_sha256=digest(__file__),protocol_sha256=digest(ROOT/'docs/attention-cost-protocol.md'),runs=[])
    for n in (256,512,1024,2048):
        for seed in (901,907):
            for method in (('dense','tiled') if seed==901 else ('tiled','dense')):
                result=subprocess.run([sys.executable,str(Path(__file__).resolve()),str(n),str(seed),method],capture_output=True,text=True,check=True)
                row=json.loads(result.stdout); r['runs'].append(row); rp.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(row),flush=True)
    errors=[]
    for n in (256,512,1024,2048):
        for seed in (901,907):
            pair=[a for a in r['runs'] if a['n']==n and a['seed']==seed]; assert len({a['input_sha256'] for a in pair})==1
            a,b=[torch.load(ROOT/x['output'],weights_only=True) for x in pair]; err=(a-b).abs().max().item(); assert err<1e-10; errors.append(err)
    torch.set_num_threads(1); torch.manual_seed(911)
    q,k,v=[torch.randn(129,64,dtype=torch.float64) for _ in range(3)]
    with torch.inference_mode(): boundary=(dense(q,k,v)-tiled(q,k,v)).abs().max().item()
    assert boundary<1e-10
    r.update(max_output_error=max(errors),boundary_error=boundary,complete=True); rp.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__': main()

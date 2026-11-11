"""Matched normalization counts, common initialization and actual-vocabulary GPU preflight."""
import copy
import json
import torch
from torch.nn import functional as F
from wikitext2_norm import ROOT,SEEDS,ARMS,configure,model_new,common_hash,forward_device,provenance,load_stream
from audit_wikitext2_norm import independent
from wikitext2_training import forward_device as previous_forward
from verify_wikitext2_baseline import model_new as previous_model
from position_distractors import digest


def relative(a,b):return (a-b).norm().item()/max(b.norm().item(),1e-12)


def main():
    configure();assert torch.cuda.is_available()
    vocabulary=json.loads((ROOT/'results/wikitext2_encoding.json').read_text())['vocabulary']
    math_report=json.loads((ROOT/'results/rmsnorm_math.json').read_text())
    assert all(digest(ROOT/name)==h for name,h in math_report['provenance'].items())
    math_audit=json.loads((ROOT/'results/rmsnorm_math_audit.json').read_text())
    assert math_audit['passed'] and math_audit['report_sha256']==digest(ROOT/'results/rmsnorm_math.json')
    assert math_audit['source_sha256']==digest(ROOT/'experiments/audit_rmsnorm_math.py')
    stream=load_stream('train')
    ids=torch.stack((stream[:64],stream[64:128]));targets=torch.stack((stream[1:65],stream[65:129]))
    torch.manual_seed(1861)
    previous=previous_model('mqa',vocabulary).float().double()
    for block in previous.blocks:
        block.norm1.bias = None; block.norm2.bias = None
    previous.final_norm.bias = None
    current=model_new('ln',vocabulary,1861).double()
    a,b=forward_device(current,ids),previous_forward(previous,ids)
    compatibility=(a-b).abs().max().item();assert compatibility<=1e-10
    ca=torch.autograd.grad(a.square().mean(),list(current.parameters()))
    cb=torch.autograd.grad(b.square().mean(),list(previous.parameters()))
    compatibility_grad=max((x-y).abs().max().item() for x,y in zip(ca,cb));assert compatibility_grad<=1e-10
    del previous,a,b,ca,cb
    records=[]
    for seed in SEEDS:
        hashes=[]
        for arm in ARMS:
            base=model_new(arm,vocabulary,seed);hashes.append(common_hash(base))
            parameters=sum(p.numel() for p in base.parameters());assert parameters==32*vocabulary+3856
            cpu=copy.deepcopy(base).double();gpu=copy.deepcopy(base).cuda()
            ref=independent(cpu,ids);actual=forward_device(cpu,ids)
            error=(actual-ref).abs().max().item();assert error<=1e-10
            ga=torch.autograd.grad(F.cross_entropy(actual.flatten(0,1),targets.flatten()),list(cpu.parameters()))
            gr=torch.autograd.grad(F.cross_entropy(ref.flatten(0,1),targets.flatten()),list(cpu.parameters()))
            grad_error=max((a-b).abs().max().item() for a,b in zip(ga,gr));assert grad_error<=1e-10
            output=forward_device(gpu,ids.cuda())
            F.cross_entropy(output.flatten(0,1),targets.flatten().cuda()).backward()
            absolute=(output.detach().cpu().double()-ref.detach()).abs().max().item()
            rel=relative(output.detach().cpu().double(),ref.detach())
            gradients=torch.cat([p.grad.detach().cpu().double().flatten() for p in gpu.parameters()])
            reference=torch.cat([g.flatten() for g in gr])
            grad_abs=(gradients-reference).abs().max().item();grad_rel=relative(gradients,reference)
            assert absolute<=1e-4 and rel<=1e-5 and grad_abs<=1e-5 and grad_rel<=1e-4
            changed=ids.clone();changed[:,-1]=(changed[:,-1]+1)%vocabulary
            with torch.no_grad():other=forward_device(gpu,changed.cuda())
            prefix=(output[:,:-1]-other[:,:-1]).abs().max().item()
            final=(output[:,-1]-other[:,-1]).abs().max().item()
            assert prefix<=1e-6 and final>1e-6
            with torch.no_grad():wrong=(independent(cpu,ids,wrong=True)-ref).abs().max().item()
            assert wrong>1e-4
            records.append(dict(seed=seed,arm=arm,parameters=parameters,common_initial_sha256=hashes[-1],
                                cpu_output_error=error,cpu_gradient_error=grad_error,gpu_output_error=absolute,gpu_output_relative=rel,
                                gpu_gradient_error=grad_abs,gpu_gradient_relative=grad_rel,causal_prefix_error=prefix,
                                changed_final_error=final,wrong_centering_error=wrong))
    result=dict(passed=True,provenance=provenance(),math_sha256=digest(ROOT/'results/rmsnorm_math.json'),math_audit_sha256=digest(ROOT/'results/rmsnorm_math_audit.json'),
                torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),checks=records,
                previous_relu_compatibility_error=compatibility,previous_relu_gradient_error=compatibility_grad)
    for seed in SEEDS:
        assert len({r['common_initial_sha256'] for r in records if r['seed']==seed})==1
    out=ROOT/'results/wikitext2_norm_verification.json';assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,models=len(records),max_output_error=max(r['gpu_output_error'] for r in records),max_gradient_relative=max(r['gpu_gradient_relative'] for r in records))))


if __name__=='__main__':main()

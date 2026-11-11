"""Matched FFN counts, common initialization and actual-vocabulary GPU preflight."""
import copy
import json
import torch
from torch.nn import functional as F
from swiglu_training import ROOT,SEEDS,ARMS,configure,model_new,common_hash,forward_device,provenance,load_stream
from audit_swiglu_training import independent
from wikitext2_training import forward_device as previous_forward
from verify_wikitext2_baseline import model_new as previous_model
from position_distractors import digest


def relative(a,b):return (a-b).norm().item()/max(b.norm().item(),1e-12)


def main():
    configure();assert torch.cuda.is_available()
    vocabulary=json.loads((ROOT/'results/wikitext2_encoding.json').read_text())['vocabulary']
    math_report=json.loads((ROOT/'results/swiglu_math.json').read_text())
    assert math_report['passed'] and math_report['source_sha256']==digest(ROOT/'experiments/swiglu_math.py')
    stream=load_stream('train')
    ids=torch.stack((stream[:64],stream[64:128]));targets=torch.stack((stream[1:65],stream[65:129]))
    torch.manual_seed(1783)
    previous=previous_model('mqa',vocabulary).double()
    a,b=forward_device(previous,ids),previous_forward(previous,ids)
    compatibility=(a-b).abs().max().item();assert compatibility<=1e-10
    ca=torch.autograd.grad(a.square().mean(),list(previous.parameters()))
    cb=torch.autograd.grad(b.square().mean(),list(previous.parameters()))
    compatibility_grad=max((x-y).abs().max().item() for x,y in zip(ca,cb));assert compatibility_grad<=1e-10
    del previous,a,b,ca,cb
    records=[]
    for seed in SEEDS:
        hashes=[]
        for arm in ARMS:
            base=model_new(arm,vocabulary,seed);hashes.append(common_hash(base))
            parameters=sum(p.numel() for p in base.parameters());assert parameters==32*vocabulary+3904
            for block in base.blocks:
                count=sum(p.numel() for module in (block.linear1,block.linear2) for p in module.parameters())
                if hasattr(block,'content'):count+=sum(p.numel() for p in block.content.parameters())
                assert count==1056 and block.linear1.bias is None and block.linear2.bias is None
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
            wrong=None
            if arm=='swiglu':
                with torch.no_grad():wrong=(independent(cpu,ids,wrong_product=True)-ref).abs().max().item()
                assert wrong>1e-4
            records.append(dict(seed=seed,arm=arm,parameters=parameters,common_initial_sha256=hashes[-1],
                                cpu_output_error=error,cpu_gradient_error=grad_error,gpu_output_error=absolute,gpu_output_relative=rel,
                                gpu_gradient_error=grad_abs,gpu_gradient_relative=grad_rel,causal_prefix_error=prefix,
                                changed_final_error=final,omitted_product_error=wrong))
    result=dict(passed=True,provenance=provenance(),math_sha256=digest(ROOT/'results/swiglu_math.json'),
                torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),checks=records,
                previous_relu_compatibility_error=compatibility,previous_relu_gradient_error=compatibility_grad)
    for seed in SEEDS:
        assert len({r['common_initial_sha256'] for r in records if r['seed']==seed})==1
    out=ROOT/'results/swiglu_training_verification.json';assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,models=len(records),max_output_error=max(r['gpu_output_error'] for r in records),max_gradient_relative=max(r['gpu_gradient_relative'] for r in records))))


if __name__=='__main__':main()

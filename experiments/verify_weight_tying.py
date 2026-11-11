"""Matched vocabulary sharing, common initialization and actual-vocabulary GPU preflight."""
import copy
import json
import torch
from torch.nn import functional as F
from weight_tying import ROOT,SEEDS,ARMS,configure,model_new,common_hash,forward_device,provenance,load_stream
from audit_weight_tying import independent
from wikitext2_training import forward_device as previous_forward
from verify_wikitext2_baseline import model_new as previous_model
from position_distractors import digest


def relative(a,b):return (a-b).norm().item()/max(b.norm().item(),1e-12)


def main():
    configure();assert torch.cuda.is_available()
    vocabulary=json.loads((ROOT/'results/wikitext2_encoding.json').read_text())['vocabulary']
    math_report=json.loads((ROOT/'results/weight_tying_math.json').read_text())
    assert math_report['passed'] and math_report['source_sha256']==digest(ROOT/'experiments/weight_tying_math.py')
    assert math_report['protocol_sha256']==digest(ROOT/'docs/decisions/0006-wikitext2-weight-tying.md')
    stream=load_stream('train')
    ids=torch.stack((stream[:64],stream[64:128]));targets=torch.stack((stream[1:65],stream[65:129]))
    records=[]
    for seed in SEEDS:
        pair=[model_new(arm,vocabulary,seed).double() for arm in ARMS]
        untied,tied=pair
        assert untied.output.weight is not untied.embedding.weight
        assert tied.output.weight is tied.embedding.weight
        assert all(torch.equal(v,tied.state_dict()[k]) for k,v in untied.state_dict().items())
        ua,ta=[forward_device(m,ids) for m in pair]
        initial_error=(ua-ta).abs().max().item();assert initial_error<=1e-10
        ug=dict(zip(dict(untied.named_parameters()),torch.autograd.grad(F.cross_entropy(ua.flatten(0,1),targets.flatten()),list(untied.parameters()))))
        tg=dict(zip(dict(tied.named_parameters()),torch.autograd.grad(F.cross_entropy(ta.flatten(0,1),targets.flatten()),list(tied.parameters()))))
        combined=ug['embedding.weight']+ug['output.weight']
        shared_error=(tg['embedding.weight']-combined).abs().max().item();assert shared_error<=1e-10
        for name,g in tg.items():
            if name!='embedding.weight':assert (g-ug[name]).abs().max().item()<=1e-10
        # Equal-valued clones are a negative control for true sharing.
        clone=copy.deepcopy(tied)
        clone.output.weight=torch.nn.Parameter(clone.output.weight.detach().clone())
        assert clone.output.weight is not clone.embedding.weight
        with torch.no_grad():
            clone.embedding.weight[0,0]+=1
        assert not torch.equal(clone.output.weight,clone.embedding.weight)
        del ua,ta,ug,tg,untied,tied,pair,clone
    for seed in SEEDS:
        hashes=[]
        for arm in ARMS:
            base=model_new(arm,vocabulary,seed);hashes.append(common_hash(base))
            parameters=sum(p.numel() for p in base.parameters());assert parameters==(16*vocabulary+3936 if arm=='tied' else 32*vocabulary+3936)
            cpu=copy.deepcopy(base).double();gpu=copy.deepcopy(base).cuda()
            assert (cpu.output.weight is cpu.embedding.weight)==(arm=='tied')
            assert (gpu.output.weight is gpu.embedding.weight)==(arm=='tied')
            optimizer=torch.optim.Adam(gpu.parameters(),lr=.001)
            assert sum(p is gpu.embedding.weight for group in optimizer.param_groups for p in group['params'])==1
            loaded=model_new(arm,vocabulary,seed)
            loaded.load_state_dict(base.state_dict())
            assert (loaded.output.weight is loaded.embedding.weight)==(arm=='tied')
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
            records.append(dict(seed=seed,arm=arm,parameters=parameters,common_initial_sha256=hashes[-1],
                                cpu_output_error=error,cpu_gradient_error=grad_error,gpu_output_error=absolute,gpu_output_relative=rel,
                                gpu_gradient_error=grad_abs,gpu_gradient_relative=grad_rel,causal_prefix_error=prefix,
                                changed_final_error=final,paired_initial_output_error=initial_error,shared_gradient_sum_error=shared_error,clone_sharing_control_detected=True))
    result=dict(passed=True,provenance=provenance(),math_sha256=digest(ROOT/'results/weight_tying_math.json'),
                torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),checks=records)
    for seed in SEEDS:
        assert len({r['common_initial_sha256'] for r in records if r['seed']==seed})==1
    out=ROOT/'results/weight_tying_verification.json';assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,models=len(records),max_output_error=max(r['gpu_output_error'] for r in records),max_gradient_relative=max(r['gpu_gradient_relative'] for r in records))))


if __name__=='__main__':main()

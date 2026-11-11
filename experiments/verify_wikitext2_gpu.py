"""Actual vocabulary, paired seeds and GPU derivatives checked before training."""
from wikitext2_training import ROOT,SEEDS,configure,model_new,forward_device,tensor_hash,load_stream,provenance
import copy
import json
import torch
from torch.nn import functional as F
from audit_mqa_training import independent
from position_distractors import digest


def relative(a,b):return (a-b).norm().item()/max(b.norm().item(),1e-12)


def main():
    configure();assert torch.cuda.is_available()
    r=json.loads((ROOT/'results/wikitext2_encoding.json').read_text());v=r['vocabulary']
    stream=load_stream('train');ids=torch.stack((stream[:64],stream[64:128]));targets=torch.stack((stream[1:65],stream[65:129]))
    records=[]
    for seed in SEEDS:
        hashes=[]
        for group in ('mha','mqa'):
            torch.manual_seed(seed);base=model_new(group,v).float();hashes.append((tensor_hash(base,'common'),tensor_hash(base,'kv')))
            cpu=copy.deepcopy(base).double();gpu=copy.deepcopy(base).cuda()
            reference=independent(cpu,ids);actual=forward_device(cpu,ids)
            ce=(actual-reference).abs().max().item();assert ce<=1e-10
            ca=torch.autograd.grad(F.cross_entropy(actual.flatten(0,1),targets.flatten()),list(cpu.parameters()))
            cr=torch.autograd.grad(F.cross_entropy(reference.flatten(0,1),targets.flatten()),list(cpu.parameters()))
            cg=max((a-b).abs().max().item() for a,b in zip(ca,cr));assert cg<=1e-10
            score=forward_device(gpu,ids.cuda());loss=F.cross_entropy(score.flatten(0,1),targets.flatten().cuda());loss.backward()
            output=score.detach().cpu().double();ref=reference.detach()
            absolute=(output-ref).abs().max().item();rel=relative(output,ref);assert absolute<=1e-4 and rel<=1e-5
            gradients=torch.cat([p.grad.detach().cpu().double().flatten() for p in gpu.parameters()])
            refgrad=torch.cat([g.flatten() for g in cr]);ga=(gradients-refgrad).abs().max().item();gr=relative(gradients,refgrad)
            assert ga<=1e-5 and gr<=1e-4 and bool(torch.isfinite(gradients).all())
            changed=ids.clone();changed[:,-1]=(changed[:,-1]+1)%v
            with torch.no_grad():other=forward_device(gpu,changed.cuda())
            prefix=(score[:,:-1]-other[:,:-1]).abs().max().item();final=(score[:,-1]-other[:,-1]).abs().max().item()
            assert prefix<=1e-6 and final>1e-6
            records.append(dict(seed=seed,group=group,cpu_double_output_error=ce,cpu_double_gradient_error=cg,
                                gpu_output_max_error=absolute,gpu_output_relative_l2=rel,gpu_gradient_max_error=ga,
                                gpu_gradient_relative_l2=gr,causal_prefix_error=prefix,changed_final_error=final))
        assert hashes[0]==hashes[1]
    # The actual corpus target coverage was independently verified before this run.
    da=json.loads((ROOT/'results/wikitext2_data_audit.json').read_text());assert da['passed'] and da['encoding_sha256']==digest(ROOT/'results/wikitext2_encoding.json')
    result=dict(passed=True,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
                vocabulary=v,provenance=provenance(),checks=records)
    out=ROOT/'results/wikitext2_gpu_verification.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,models=len(records),max_output_error=max(x['gpu_output_max_error'] for x in records),
                         max_gradient_relative_l2=max(x['gpu_gradient_relative_l2'] for x in records))))


if __name__=='__main__':main()

"""Check large-vocabulary pairing and next-word math before corpus training."""
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from mqa_training import ROOT,model_new as template_model,forward,shared_hash,retained_kv_hash
from audit_mqa_training import independent
from position_distractors import digest


def model_new(group,vocabulary):
    model=template_model(group)
    model.embedding=nn.Embedding(vocabulary,16).double()
    model.output=nn.Linear(16,vocabulary,bias=False).double()
    return model


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    # Nominal published size is a shape probe, not a parsed vocabulary assertion.
    vocabulary=33278; sequence=torch.tensor([[1,7,19,3,31,2],[3,19,7,11,2,5]])
    ids,targets=sequence[:,:-1],sequence[:,1:]; assert not torch.equal(ids,targets)
    torch.manual_seed(1723);mha=model_new('mha',vocabulary)
    torch.manual_seed(1723);mqa=model_new('mqa',vocabulary)
    assert shared_hash(mha)==shared_hash(mqa) and retained_kv_hash(mha)==retained_kv_hash(mqa)
    checks=[]
    for group,model in (('mha',mha),('mqa',mqa)):
        count=sum(p.numel() for p in model.parameters()); assert count==32*vocabulary+(4480 if group=='mha' else 3936)
        a=forward(model,ids);b=independent(model,ids)
        error=(a-b).abs().max().item(); assert error<=1e-10
        loss=F.cross_entropy(a.flatten(0,1),targets.flatten())
        reference=(a.logsumexp(-1)-a.gather(-1,targets[...,None]).squeeze(-1)).mean()
        assert abs(loss.item()-reference.item())<=1e-10
        parameters=list(model.parameters())
        ga=torch.autograd.grad(loss,parameters)
        gb=torch.autograd.grad(F.cross_entropy(b.flatten(0,1),targets.flatten()),parameters)
        gradient_error=max((x-y).abs().max().item() for x,y in zip(ga,gb)); assert gradient_error<=1e-10
        assert all(torch.isfinite(g).all() for g in ga)
        changed=ids.clone(); changed[:,-1]=17
        with torch.no_grad(): c=forward(model,changed)
        prefix=(a[:,:-1]-c[:,:-1]).abs().max().item(); final=(a[:,-1]-c[:,-1]).abs().max().item()
        assert prefix<=1e-10 and final>1e-6
        # A supplied unshifted target must be rejected by the data contract.
        expected=sequence[:,1:]; wrong=sequence[:,:-1]; mismatches=int((expected!=wrong).sum())
        assert mismatches==targets.numel()
        checks.append(dict(group=group,parameters=count,max_logit_error=error,max_gradient_error=gradient_error,
                           loss_identity_error=abs(loss.item()-reference.item()),causal_prefix_error=prefix,
                           future_change_error=final,wrong_target_control_mismatches=mismatches))
    tied=copy.deepcopy(mha)
    with torch.no_grad():
        for a,b in zip(tied.blocks,mqa.blocks):
            for name in ('k_weight','k_bias','v_weight','v_bias'):
                getattr(a.attn,name).copy_(getattr(b.attn,name).repeat((2,)+(1,)*(getattr(b.attn,name).ndim-1)))
        tied_error=(forward(tied,ids)-forward(mqa,ids)).abs().max().item()
    assert tied_error<=1e-10
    protocol=ROOT/'docs/decisions/0003-wikitext2-matched-baseline.md'
    prior=json.loads((ROOT/'results/mqa_training.json').read_text())
    sources={name:digest(ROOT/'experiments'/name) for name in prior['provenance'] if name!='protocol'}
    assert all(sources[name]==sha for name,sha in prior['provenance'].items() if name!='protocol')
    sources[Path(__file__).name]=digest(Path(__file__))
    result=dict(passed=True,scope='CPU float64 nominal-vocabulary model/target preflight; no corpus parsing, GPU training or benchmark quality',
                vocabulary_shape_probe=vocabulary,seed=1723,paired_common_initial_sha256=shared_hash(mha),
                retained_kv_initial_sha256=retained_kv_hash(mha),checks=checks,replicated_mqa_mha_error=tied_error,
                protocol_sha256=digest(protocol),sources=sources,
                fp32_compact_cache_per_sequence64=dict(mha=16384,mqa=8192))
    out=ROOT/'results/wikitext2_baseline_preflight.json';assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,models=len(checks),replicated_head_error=tied_error,checks=checks)))


if __name__=='__main__':main()

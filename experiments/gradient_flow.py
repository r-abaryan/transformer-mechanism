"""Instrument deterministic training replays with residual and LN VJPs."""
import hashlib
import json
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from reserved_cycle import model_new
from position_training import shifted,weight_hash
from position_distractors import digest,pe

ROOT=Path(__file__).resolve().parents[1]


def norm(t): return t.norm().item()
def rms(t): return t.square().mean().sqrt().item()


def trace(model,ids):
    x=model.embedding(ids)+pe(torch.arange(ids.shape[1])+4,16)
    nodes={'input':x}; adds=[]; lns=[]
    mask=torch.ones(ids.shape[1],ids.shape[1],dtype=torch.bool).triu(1)
    for i,b in enumerate(model.blocks):
        z=b.norm1(x); lns.append((f'b{i+1}_attention_ln',b.norm1,x,z))
        a=b.self_attn(z,z,z,attn_mask=mask,need_weights=False)[0]; u=x+a
        adds.append((f'b{i+1}_attention',x,a,u)); nodes[f'b{i+1}_after_attention']=u
        z=b.norm2(u); lns.append((f'b{i+1}_ffn_ln',b.norm2,u,z))
        f=b.linear2(b.activation(b.linear1(z))); v=u+f
        adds.append((f'b{i+1}_ffn',u,f,v)); nodes[f'b{i+1}_after_ffn']=v; x=v
    z=model.final_norm(x); lns.append(('final_ln',model.final_norm,x,z)); nodes['after_final_ln']=z
    return model.output(z)[:,-1],nodes,adds,lns


def probe(model,x,y):
    s,nodes,adds,lns=trace(model,x); loss=F.cross_entropy(s,y)
    def grad(output,input,upstream=None): return torch.autograd.grad(output,input,grad_outputs=upstream,retain_graph=True)[0]
    gradients={name:grad(loss,t) for name,t in nodes.items()}
    lib=shifted(model,x,4); forward_error=(s-lib).abs().max().item()
    p=list(model.parameters()); ga=torch.autograd.grad(loss,p,retain_graph=True); gb=torch.autograd.grad(F.cross_entropy(lib,y),p)
    parameter_error=max((a-b).abs().max().item() for a,b in zip(ga,gb)); assert max(forward_error,parameter_error)<1e-10
    residuals={}; max_sum_error=0.
    for name,inp,branch,out in adds:
        skip=grad(loss,out); total=grad(loss,inp); via=grad(branch,inp,skip)
        err=(total-skip-via).abs().max().item(); max_sum_error=max(max_sum_error,err); assert err<1e-10
        residuals[name]=dict(skip_norm=norm(skip),branch_norm=norm(via),total_norm=norm(total),total_over_skip=norm(total)/norm(skip),
                             cosine=(skip*via).sum().item()/(norm(skip)*norm(via)),sum_error=err)
    norms={}
    for name,layer,inp,out in lns:
        gout=grad(loss,out); local=grad(out,inp,gout)
        sigma=(inp.var(-1,unbiased=False,keepdim=True)+layer.eps).sqrt(); h=(inp-inp.mean(-1,keepdim=True))/sigma; w=layer.weight*gout
        predicted=(w-w.mean(-1,keepdim=True)-h*(w*h).mean(-1,keepdim=True))/sigma
        error=(predicted-local).abs().max().item(); assert error<1e-10
        point=inp[0,-1].detach(); upstream=gout[0,-1].detach(); direction=torch.linspace(-1,1,16,dtype=torch.float64); direction/=direction.norm(); eps=1e-6
        numerical=((layer(point+eps*direction)-layer(point-eps*direction))*upstream).sum().item()/(2*eps)
        analytic=(local[0,-1]*direction).sum().item(); fd_error=abs(numerical-analytic); assert fd_error<1e-7
        norms[name]=dict(input_norm=norm(local),output_norm=norm(gout),input_over_output=norm(local)/norm(gout),vjp_error=error,
                         finite_difference_error=fd_error,input_gradient_feature_sum_max=local.sum(-1).abs().max().item())
    return dict(loss=loss.item(),accuracy=(s.argmax(-1)==y).double().mean().item(),gradient_rms={n:rms(g) for n,g in gradients.items()},
                residuals=residuals,normalizations=norms,forward_error=forward_error,parameter_gradient_error=parameter_error)


def counterexamples():
    x=torch.tensor([1.,2.],dtype=torch.float64,requires_grad=True); loss=(x-x).sum(); cancel=torch.autograd.grad(loss,x)[0]
    assert cancel.abs().max()==0
    ln=torch.nn.LayerNorm(4).double(); v=torch.tensor([1.,-1.,1.,-1.],dtype=torch.float64)
    z=torch.zeros(4,dtype=torch.float64,requires_grad=True)
    amplify=torch.autograd.grad(ln(z),z,v,retain_graph=True)[0]; constant=torch.autograd.grad(ln(z),z,torch.ones_like(z))[0]
    assert constant.abs().max()==0 and norm(amplify)>norm(v)
    return dict(residual_cancellation_gradient=cancel.tolist(),ln_constant_gradient=constant.tolist(),ln_zero_variance_amplification=norm(amplify)/norm(v),epsilon=ln.eps)


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out=ROOT/'results/gradient_flow.json'; ip=ROOT/'results/gradient_flow_probe.json'; assert not out.exists() and not ip.exists()
    pp=ROOT/'results/position_training.json'; prior=json.loads(pp.read_text()); sp=ROOT/'results/position_training_splits.json'
    assert digest(sp)==prior['split_sha256']
    for n,h in prior['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    data=json.loads(sp.read_text()); train_x=torch.tensor([x for x,y in data['train']]); train_y=torch.tensor([y for x,y in data['train']])
    indices=torch.randperm(len(data['validation']),generator=torch.Generator().manual_seed(20261227))[:128].tolist(); ip.write_text(json.dumps(indices)+'\n')
    px=torch.tensor([data['validation'][i][0] for i in indices]); py=torch.tensor([data['validation'][i][1] for i in indices])
    report=dict(phase='training',prior_sha256=digest(pp),split_sha256=digest(sp),probe_sha256=digest(ip),protocol_sha256=digest(ROOT/'docs/gradient-flow-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in ('gradient_flow.py','position_training.py','reserved_cycle.py','compare_depth.py','position_distractors.py')},counterexamples=counterexamples(),runs=[])
    def save(): out.write_text(json.dumps(report,indent=2)+'\n')
    directory=ROOT/'results/gradient_flow_checkpoints'; directory.mkdir(exist_ok=True); save()
    for old in prior['runs']:
        if old['group']!='varied': continue
        seed=old['seed']; torch.manual_seed(seed); model=model_new(); assert weight_hash(model)==old['initial_sha256']
        batches=torch.Generator().manual_seed(seed+1000); offsets=random.Random(seed+2000); stream=hashlib.sha256()
        opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
        run=dict(seed=seed,snapshots=[]); report['runs'].append(run)
        def snapshot(step):
            state=probe(model,px,py); path=directory/f'seed_{seed}_step_{step}.pt'; assert not path.exists(); torch.save(model.state_dict(),path)
            run['snapshots'].append(dict(step=step,checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=digest(path),**state)); save()
            print(json.dumps(dict(seed=seed,step=step,probe_loss=state['loss'],probe_accuracy=state['accuracy'])),flush=True)
        snapshot(0); max_grad=0.
        for step in range(1,3001):
            idx=torch.randint(len(train_x),(64,),generator=batches); stream.update(json.dumps(idx.tolist()).encode()); shift=offsets.randrange(5)
            opt.zero_grad(); loss=F.cross_entropy(shifted(model,train_x[idx],shift),train_y[idx]); assert torch.isfinite(loss); loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            max_grad=max(max_grad,max(p.grad.abs().max().item() for p in model.parameters())); opt.step()
            if step in (500,1500,3000): snapshot(step)
        assert stream.hexdigest()==old['minibatch_sha256']
        original=ROOT/old['checkpoint']; assert digest(original)==old['checkpoint_sha256']; expected=torch.load(original,weights_only=True)
        replay_error=max((t-expected[n]).abs().max().item() for n,t in model.state_dict().items()); assert replay_error==0
        run.update(minibatch_sha256=stream.hexdigest(),final_replay_error=replay_error,max_training_parameter_gradient=max_grad); save()
    report['phase']='complete'; save()


if __name__=='__main__': main()

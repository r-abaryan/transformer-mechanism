"""FFN branch interventions and local algebra checks."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from reserved_cycle import model_new
from activation_paths import forward
from position_distractors import digest,logits
from position_routing import measures

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out=ROOT/'results/ffn_branches.json'; dp=ROOT/'results/ffn_branches_scores.json'; ip=ROOT/'results/ffn_branches_permutation.json'
    assert not any(p.exists() for p in (out,dp,ip))
    pp=ROOT/'results/position_training.json'; prior=json.loads(pp.read_text())
    for n,h in prior['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    sp=ROOT/'results/canonical_positions_inputs.json'; rows=json.loads(sp.read_text())
    x=torch.tensor([x for x,y in rows]); y=torch.tensor([y for x,y in rows])
    idx=torch.randperm(len(x),generator=torch.Generator().manual_seed(20261226)); ip.write_text(json.dumps(idx.tolist())+'\n')
    report=dict(prior_sha256=digest(pp),inputs_sha256=digest(sp),permutation_sha256=digest(ip),
                protocol_sha256=digest(ROOT/'docs/ffn-branch-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in ('ffn_branches.py','reserved_cycle.py','compare_depth.py','activation_paths.py','position_distractors.py','position_routing.py')},
                fixed_points=int((idx==torch.arange(len(x))).sum()),same_answer=(y==y[idx]).double().mean().item(),runs=[])
    out.write_text(json.dumps(report,indent=2)+'\n'); details=[]
    with torch.no_grad():
        for old in prior['runs']:
            if old['group']!='varied': continue
            path=ROOT/old['checkpoint']; assert digest(path)==old['checkpoint_sha256']
            model=model_new().eval(); model.load_state_dict(torch.load(path,weights_only=True))
            fs=[[],[]]; ss=[]; branch=[dict(active=0,total=0,fnorm=0.,unorm=0.,tokens=0) for _ in range(2)]
            errors=dict(reconstruction=0.,baseline=0.,sham=0.); derivative=[]; shortcut=[]
            for start in range(0,len(x),256):
                chunk=x[start:start+256]; s,c=forward(model,chunk,4); ss.append(s)
                errors['baseline']=max(errors['baseline'],(s-logits(model,chunk,(torch.arange(8)+4).expand_as(chunk))).abs().max().item())
                for i,b in enumerate(model.blocks):
                    u=c[f'r{i}']+c[f'a{i}']; z=b.norm2(u); h=b.activation(b.linear1(z)); f=c[f'f{i}']; fs[i].append(f)
                    reconstructed=(h.unsqueeze(-1)*b.linear2.weight.T).sum(-2)+b.linear2.bias
                    errors['reconstruction']=max(errors['reconstruction'],(reconstructed-f).abs().max().item())
                    st=branch[i]; st['active']+=int((h>0).sum()); st['total']+=h.numel(); st['fnorm']+=f.norm(dim=-1).sum().item(); st['unorm']+=u.norm(dim=-1).sum().item(); st['tokens']+=u.shape[0]*u.shape[1]
                    if start==0:
                        z0=z[0,-1]; direction=torch.linspace(-1,1,16,dtype=torch.float64); direction/=direction.norm()
                        pre=b.linear1(z0); dz=b.linear1.weight@direction
                        eps=min(1e-5,pre.abs().min().item()/(4*dz.abs().max().item())); assert eps>1e-12
                        plus=b.linear1(z0+eps*direction); minus=b.linear1(z0-eps*direction)
                        assert torch.equal(plus>0,pre>0) and torch.equal(minus>0,pre>0)
                        numerical=(b.linear2(F.relu(plus))-b.linear2(F.relu(minus)))/(2*eps)
                        analytical=b.linear2.weight@((pre>0)*dz)
                        err=(numerical-analytical).abs().max().item(); assert err<1e-6
                        derivative.append(dict(epsilon=eps,error=err,min_gate_distance=pre.abs().min().item()))
                u=(c['r1']+c['a1'])[:,-1]; f=c['f1'][:,-1]
                delta=model.output(model.final_norm(u+f))-model.output(model.final_norm(u))
                raw=model.output(f); shortcut.append((delta-raw).abs())
            base=torch.cat(ss); full=[torch.cat(parts) for parts in fs]; scores={'ordinary':base}; conditions={}
            for name in ('sham','zero_first','zero_second','zero_both','permute_first','permute_second'):
                outputs=[]
                for start in range(0,len(x),256):
                    end=start+256; donor={}; patches=[]
                    targets=(0,1) if name in ('sham','zero_both') else ((0,) if name.endswith('first') else (1,))
                    for i in targets:
                        key=f'f{i}'; patches.append(key)
                        donor[key]=full[i][start:end] if name=='sham' else (full[i][idx[start:end]] if name.startswith('permute') else torch.zeros_like(full[i][start:end]))
                    outputs.append(forward(model,x[start:end],4,donor,tuple(patches))[0])
                scores[name]=torch.cat(outputs)
            errors['sham']=(scores['sham']-base).abs().max().item(); assert max(errors.values())<1e-10
            for name,s in scores.items(): conditions[name]=measures(s,y,base.argmax(-1),y[idx] if name.startswith('permute') else None)[0]
            stats=[dict(relu_active_fraction=b['active']/b['total'],mean_output_norm=b['fnorm']/b['tokens'],mean_residual_norm=b['unorm']/b['tokens']) for b in branch]
            diff=torch.cat(shortcut); assert digest(path)==old['checkpoint_sha256']
            report['runs'].append(dict(seed=old['seed'],checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],conditions=conditions,errors=errors,derivative=derivative,branch_stats=stats,raw_logit_shortcut=dict(mean_absolute_error=diff.mean().item(),max_absolute_error=diff.max().item())))
            details.append(dict(seed=old['seed'],scores={k:s.tolist() for k,s in scores.items()}))
            print(json.dumps(dict(seed=old['seed'],accuracy={k:v['accuracy'] for k,v in conditions.items()})),flush=True)
    dp.write_text(json.dumps(details,separators=(',',':'))+'\n'); report.update(scores_sha256=digest(dp),phase='complete'); out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()

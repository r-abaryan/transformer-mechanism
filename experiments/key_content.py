"""Matched key swaps and bidirectional value-row activation transfers."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from head_ablation import ROOT, forward
from norm_placement import model_for
from position_distractors import digest, logits, pe


def transfer(model,x,donor,rows,library=False):
    outputs=[]
    with torch.no_grad():
        for start in range(0,len(x),256):
            a=x[start:start+256]; d=donor[start:start+256]; rr=rows[start:start+256]
            if library:
                b=model.blocks[0]; z=b.norm1(model.embedding(d)+pe(torch.arange(8),16))
                replacement=b.self_attn(z,z,z,attn_mask=torch.ones(8,8,dtype=torch.bool).triu(1),need_weights=False)[0]
            else:
                _,cache,_=forward(model,d)
                replacement=cache[0][1].sum(1)+model.blocks[0].self_attn.out_proj.bias
            def hook(module,args,out):
                value=out[0].clone(); idx=torch.arange(len(a))[:,None]
                value[idx,rr]=replacement[idx,rr]
                return value,out[1]
            handle=model.blocks[0].self_attn.register_forward_hook(hook)
            try: outputs.append(model(a)[:,-1])
            finally: handle.remove()
    return torch.cat(outputs)


def measures(s,y,changed,eligible):
    pred=s.argmax(-1)
    return dict(original_accuracy=(pred==y).double().mean().item(),changed_accuracy=(pred==changed).double().mean().item(),
                original_loss=F.cross_entropy(s,y).item(),changed_loss=F.cross_entropy(s,changed).item(),
                eligible_count=int(eligible.sum()),eligible_changed=(pred[eligible]==changed[eligible]).double().mean().item())


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    rp=ROOT/'results/key_content.json'; dp=ROOT/'results/key_content_scores.pt'; ip=ROOT/'results/key_content_inputs.json'
    assert not any(p.exists() for p in (rp,dp,ip))
    priorpath=ROOT/'results/head_ablation.json'; prior=json.loads(priorpath.read_text())
    for n,h in prior['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    inp=ROOT/'results/norm_extension_inputs.json'; assert digest(inp)==prior['input_sha256']
    data=json.loads(inp.read_text()); x=torch.tensor([a for a,y in data]); y=torch.tensor([y for a,y in data])
    rank=(x[:,:6:2]==x[:,-1:]).long().argmax(-1); partner=(rank+1)%3; idx=torch.arange(len(x))
    changed=x.clone(); changed[idx,2*rank]=x[idx,2*partner]; changed[idx,2*partner]=x[idx,2*rank]
    cy=x[idx,2*partner+1]; rr=torch.stack((2*rank+1,2*partner+1),1)
    control=x.clone(); other=(rank+2)%3
    control[idx,2*partner]=x[idx,2*other]; control[idx,2*other]=x[idx,2*partner]
    cr=torch.stack((2*partner+1,2*other+1),1)
    for a,d,c,label,new in zip(x.tolist(),changed.tolist(),control.tolist(),y.tolist(),cy.tolist()):
        assert a[1::2]==d[1::2]==c[1::2] and a[-2:]==d[-2:]==c[-2:]
        assert dict(zip(d[:6:2],d[1:6:2]))[d[-1]]==new!=label
        assert dict(zip(c[:6:2],c[1:6:2]))[c[-1]]==label
    ip.write_text(json.dumps(dict(recipient=x.tolist(),changed=changed.tolist(),control=control.tolist(),original_labels=y.tolist(),changed_labels=cy.tolist(),rows=rr.tolist(),control_rows=cr.tolist()),separators=(',',':'))+'\n')
    def identity(a): return tuple(sorted(zip(a[:6:2],a[1:6:2])))
    train=json.loads((ROOT/'results/norm_placement_splits.json').read_text())['train']; tm={identity(a) for a,b in train}
    overlap={n:dict(unique_mappings=len({identity(a) for a in t.tolist()}),training_overlap=len({identity(a) for a in t.tolist()}&tm)) for n,t in [('changed',changed),('control',control)]}
    report=dict(source_sha256=digest(Path(__file__)),protocol_sha256=digest(ROOT/'docs/key-content-protocol.md'),prior_sha256=digest(priorpath),input_sha256=digest(ip),overlap=overlap,runs=[])
    saved=[]
    for original in prior['runs']:
        cp=ROOT/original['checkpoint']; assert digest(cp)==original['checkpoint_sha256']
        model=model_for('pre').eval(); model.load_state_dict(torch.load(cp,weights_only=True)['model'])
        scores=dict(recipient=logits(model,x),changed_donor=logits(model,changed),control_donor=logits(model,control)); err=0.
        assert abs(F.cross_entropy(scores['recipient'],y).item()-original['conditions']['ordinary']['loss'])<1e-10
        for name,a,d,rows in [('self',x,x,rr),('forward',x,changed,rr),('reverse',changed,x,rr),('control',x,control,cr)]:
            s=transfer(model,a,d,rows); check=transfer(model,a,d,rows,True)
            err=max(err,(s-check).abs().max().item()); assert err<1e-10 and torch.isfinite(s).all(); scores[name]=s
        sham=(scores['self']-scores['recipient']).abs().max().item(); assert sham<1e-10
        eligible=(scores['recipient'].argmax(-1)==y)&(scores['changed_donor'].argmax(-1)==cy)
        run=dict(seed=original['seed'],checkpoint=original['checkpoint'],checkpoint_sha256=digest(cp),conditions={n:measures(s,y,cy,eligible) for n,s in scores.items()},verification=dict(library_error=err,self_error=sham))
        assert digest(cp)==original['checkpoint_sha256']; report['runs'].append(run); saved.append(dict(seed=run['seed'],scores=scores))
        print(json.dumps(dict(seed=run['seed'],forward=run['conditions']['forward'],reverse=run['conditions']['reverse'])),flush=True)
    torch.save(saved,dp); report['scores_sha256']=digest(dp); rp.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()

"""Localize projected head removals without changing the archived forward."""
import copy
import json
from pathlib import Path
import torch
from head_ablation import ROOT, forward, summarize
from norm_placement import model_for
from position_distractors import digest, logits

GROUPS={'values':[1,3,5],'query':[7],'remaining':[0,2,4,6],'all':list(range(8))}


def scores_for(model,ids,block,heads,rows,independent=False):
    parts=[]
    attention=model.blocks[block].self_attn
    alternate=None
    if independent:
        alternate=copy.deepcopy(attention)
        with torch.no_grad():
            for h in heads: alternate.out_proj.weight[:,8*h:8*(h+1)]=0
    with torch.no_grad():
        for chunk in ids.split(256):
            contribution=None
            if not independent:
                _,cache,_=forward(model,chunk)
                contribution=cache[block][1][:,list(heads)].sum(1)
            def hook(module,args,kwargs,output):
                changed=output[0].clone()
                if independent:
                    replacement=alternate(*args,**kwargs)[0]
                    changed[:,rows]=replacement[:,rows]
                else: changed[:,rows]-=contribution[:,rows]
                return changed,output[1]
            handle=attention.register_forward_hook(hook,with_kwargs=True)
            try: parts.append(model(chunk)[:,-1])
            finally: handle.remove()
    return torch.cat(parts)


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    rp=ROOT/'results/head_rows.json'; dp=ROOT/'results/head_rows_scores.pt'
    assert not rp.exists() and not dp.exists()
    priorpath=ROOT/'results/head_ablation.json'; prior=json.loads(priorpath.read_text())
    for n,h in prior['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    oldp=ROOT/'results/head_ablation_scores.pt'; assert digest(oldp)==prior['scores_sha256']
    old=torch.load(oldp,weights_only=True)
    inp=ROOT/'results/norm_extension_inputs.json'; assert digest(inp)==prior['input_sha256']
    data=json.loads(inp.read_text()); ids=torch.tensor([x for x,y in data]); y=torch.tensor([y for x,y in data])
    report=dict(prior_sha256=digest(priorpath),input_sha256=digest(inp),source_sha256=digest(Path(__file__)),
                protocol_sha256=digest(ROOT/'docs/head-row-protocol.md'),groups=GROUPS,runs=[])
    saved=[]
    for original in prior['runs']:
        cp=ROOT/original['checkpoint']; assert digest(cp)==original['checkpoint_sha256']
        model=model_for('pre').eval(); model.load_state_dict(torch.load(cp,weights_only=True)['model'])
        previous=next(x['scores'] for x in old if x['seed']==original['seed'])
        base=logits(model,ids); assert (base-previous['ordinary']).abs().max()<1e-10
        run=dict(seed=original['seed'],checkpoint=original['checkpoint'],checkpoint_sha256=digest(cp),conditions={},verification={},value_all_error_overlap={})
        tensors={'ordinary':base}; parity=0.; reproduction=0.; structural=0.
        for b in range(2):
            for label,heads in [('h0',(0,)),('h1',(1,)),('both',(0,1))]:
                prefix=f'b{b+1}_{label}'
                for group,rows in GROUPS.items():
                    s=scores_for(model,ids,b,heads,rows)
                    ref=scores_for(model,ids,b,heads,rows,True)
                    err=(s-ref).abs().max().item(); parity=max(parity,err); assert err<1e-10 and torch.isfinite(s).all()
                    tensors[prefix+'_'+group]=s; run['conditions'][prefix+'_'+group]=summarize(s,y,base,ids)
                    if group=='all':
                        err=(s-previous[prefix]).abs().max().item(); reproduction=max(reproduction,err); assert err<1e-10
                    if b==1:
                        expected=base if group in ('values','remaining') else previous[prefix]
                        err=(s-expected).abs().max().item(); structural=max(structural,err); assert err<1e-10
                va=tensors[prefix+'_values'].argmax(-1)!=y; aa=tensors[prefix+'_all'].argmax(-1)!=y
                run['value_all_error_overlap'][prefix]=dict(both_wrong=int((va&aa).sum()),values_only_wrong=int((va&~aa).sum()),all_only_wrong=int((~va&aa).sum()))
        run['verification']=dict(library_parity=parity,prior_reproduction=reproduction,structural=structural)
        assert digest(cp)==original['checkpoint_sha256']
        report['runs'].append(run); saved.append(dict(seed=run['seed'],scores=tensors))
        print(json.dumps(dict(seed=run['seed'],verification=run['verification'])),flush=True)
    torch.save(saved,dp); report['scores_sha256']=digest(dp)
    rp.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()

"""Balanced matched value edits with prediction and routing measurements."""
import itertools
import json
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from activation_paths import forward
from compare_depth import StackedDecoder
from qkv_paths import project,attention
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    output=ROOT/'results/value_counterfactuals.json'; splitpath=ROOT/'results/value_counterfactual_inputs.json'
    assert not output.exists() and not splitpath.exists()
    triples=random.Random(20261210).sample(list(itertools.permutations(range(8),3)),16)
    families=[]
    names=('distinct','repeat_first','repeat_second','all_equal','wrong_repeat')
    for keys in triples:
        for rank in range(3):
            for target in range(8,12):
                other=[i for i in range(3) if i!=rank]
                for d1,d2 in itertools.permutations([v for v in range(8,12) if v!=target],2):
                    values=[target]*3; values[other[0]]=d1; values[other[1]]=d2
                    variants={}
                    for name in names:
                        vs=values.copy()
                        if name in ('repeat_first','all_equal'): vs[other[0]]=target
                        if name in ('repeat_second','all_equal'): vs[other[1]]=target
                        if name=='wrong_repeat': vs[other[1]]=d1
                        ids=[t for kv in zip(keys,vs) for t in kv]+[12,keys[rank]]
                        assert ids[2*rank+1]==target and ids[:-2:2]==list(keys)
                        variants[name]=ids
                    families.append(dict(rank=rank,target=target,inputs=variants))
    assert len(families)==1152
    assert all(sum(f['target']==v for f in families)==288 for v in range(8,12))
    assert all(sum(f['rank']==i for f in families)==384 for i in range(3))
    splitpath.write_text(json.dumps(families,separators=(',',':'))+'\n')
    y=torch.tensor([f['target'] for f in families]); targetpos=torch.tensor([2*f['rank']+1 for f in families])
    priorpath=ROOT/'results/pair_shift_confirmation.json'; prior=json.loads(priorpath.read_text())
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),family_count=len(families),
        protocol_sha256=digest(ROOT/'docs/value-counterfactual-protocol.md'),
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('value_counterfactuals.py','activation_paths.py','qkv_paths.py','compare_depth.py','position_distractors.py')},runs=[])
    for run in prior['runs']:
        path=ROOT/run['checkpoint']; assert digest(path)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval(); model.load_state_dict(torch.load(path,weights_only=True))
        allscores={}; routes={}; error_max=0.
        with torch.no_grad():
            for shift,mode,name in itertools.product((0,8),('original','pair'),names):
                x=torch.tensor([f['inputs'][name] for f in families]); pieces=[]; route=[]
                for start in range(0,len(x),256):
                    chunk=x[start:start+256]; s,cache=forward(model,chunk,shift)
                    if mode=='pair':
                        layer=model.blocks[0].self_attn; z=model.blocks[0].norm1(cache['r0'])
                        q,k,v=[t.reshape(len(chunk),8,2,8).transpose(1,2) for t in F.linear(z,layer.in_proj_weight,layer.in_proj_bias).chunk(3,-1)]
                        mask=torch.ones(8,8,dtype=torch.bool).triu(1)
                        w=(q@k.transpose(-2,-1)/(8**.5)).masked_fill(mask,float('-inf')).softmax(-1)
                        for row in (3,5): w[:,:,row]=0; w[:,:,row,row-1]=1
                        a=F.linear((w@v).transpose(1,2).reshape(len(chunk),8,16),layer.out_proj.weight,layer.out_proj.bias)
                        s,cache=forward(model,chunk,shift,{'a0':a},('a0',))
                    q,k,v=project(model,cache['r1'])
                    error=(attention(model,q,k,v)-cache['a1']).abs().max().item(); error_max=max(error_max,error); assert error<1e-10
                    qh=q[:,7].reshape(-1,2,8); kh=k.reshape(-1,8,2,8).transpose(1,2)
                    probs=((qh[:,:,None,:]*kh).sum(-1)/(8**.5)).softmax(-1)
                    tp=targetpos[start:start+len(chunk),None,None].expand(-1,2,1)
                    targetmass=probs.gather(-1,tp).squeeze(-1)
                    answer_mask=(chunk[:,1:6:2]==y[start:start+len(chunk),None])
                    answer_mass=(probs[:,:,[1,3,5]]*answer_mask[:,None,:]).sum(-1)
                    top=(targetmass>=probs[:,:,[1,3,5]].max(-1).values).double()
                    route.append(torch.stack((targetmass,answer_mass,top),-1)); pieces.append(s)
                key=f'{shift}_{mode}_{name}'
                allscores[key]=torch.cat(pieces); routes[key]=torch.cat(route).mean(0).tolist()
        conditions={}
        for key,s in allscores.items():
            assert torch.isfinite(s).all()
            reference='_'.join(key.split('_')[:2])+'_distinct'
            before=allscores[reference].argmax(-1)==y; now=s.argmax(-1)==y
            conditions[key]=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),reference=reference,
                rescued=int((now&~before).sum()),harmed=int((~now&before).sum()),routing_per_head=routes[key],predictions=s.argmax(-1).tolist())
        assert digest(path)==run['checkpoint_sha256']
        report['runs'].append(dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(path),reconstruction_error=error_max,conditions=conditions))
    report['all_equal_gain_5_every_seed']=all(r['conditions']['8_pair_all_equal']['accuracy']-r['conditions']['8_pair_distinct']['accuracy']>=.05 for r in report['runs'])
    report['routing_columns']=['target_slot_probability','answer_value_slots_probability','target_top_among_value_slots_ties_included']
    output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(dict(all_equal_gain_5_every_seed=report['all_equal_gain_5_every_seed'])))


if __name__=='__main__': main()

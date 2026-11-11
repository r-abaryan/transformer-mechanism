"""New vocabulary, whole-mapping reservation, fixed training and intervention."""
import itertools
import json
import random
import statistics
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from compare_depth import StackedDecoder
from activation_paths import forward
from position_distractors import digest,logits
from qkv_paths import project,attention

ROOT=Path(__file__).resolve().parents[1]


def model_new():
    model=StackedDecoder(16,32,2)
    model.embedding=nn.Embedding(19,16)
    model.output=nn.Linear(16,19,bias=False)
    return model.double()


def evaluate(model,data):
    model.eval()
    x,y=data
    with torch.no_grad(): s=torch.cat([model(c)[:,-1] for c in x.split(256)])
    model.train()
    return dict(accuracy=(s.argmax(-1)==y).double().mean().item(),loss=F.cross_entropy(s,y).item())


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    output=ROOT/'results/reserved_cycle.json'; splitpath=ROOT/'results/reserved_cycle_splits.json'
    assert not output.exists() and not splitpath.exists()
    rng=random.Random(20261220); maps={n:[] for n in ('train','validation','test')}
    for values in itertools.combinations(range(12,18),3):
        candidates=[tuple(zip(keys,vs)) for keys in itertools.combinations(range(12),3) for vs in itertools.permutations(values)]
        rng.shuffle(candidates)
        for n,start,end in [('train',0,80),('validation',80,100),('test',100,120)]: maps[n].extend(candidates[start:end])
    assert not set(maps['train'])&set(maps['validation']) and not set(maps['train'])&set(maps['test']) and not set(maps['validation'])&set(maps['test'])
    splits={}
    for name,ms in maps.items():
        rows=[]
        for m in ms:
            for order in itertools.permutations(m):
                for query,target in m:
                    ids=[t for pair in order for t in pair]+[18,query]
                    assert dict(order)[query]==target and len(set(ids[1:6:2]))==3
                    rows.append((ids,target))
        assert all(sum(y==v for x,y in rows)==len(rows)//6 for v in range(12,18))
        assert all(sum(x[:-2:2].index(x[-1])==i for x,y in rows)==len(rows)//3 for i in range(3))
        splits[name]=rows
    splitpath.write_text(json.dumps(splits,separators=(',',':'))+'\n')
    data={n:(torch.tensor([x for x,y in rows]),torch.tensor([y for x,y in rows])) for n,rows in splits.items()}
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',split_sha256=digest(splitpath),
        split_counts={n:dict(mappings=len(maps[n]),sequences=len(splits[n])) for n in maps},
        protocol_sha256=digest(ROOT/'docs/reserved-cycle-protocol.md'),
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('reserved_cycle.py','compare_depth.py','activation_paths.py','position_distractors.py','qkv_paths.py')},runs=[],phase='training')
    directory=ROOT/'results/reserved_cycle_checkpoints'; directory.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    for seed in (307,331,367,401,433):
        torch.manual_seed(seed); model=model_new()
        assert sum(p.numel() for p in model.parameters())==5088
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
        generator=torch.Generator().manual_seed(seed+1000)
        curve=[dict(step=0,validation=evaluate(model,data['validation']))]
        x,y=data['train']
        for step in range(1,3001):
            idx=torch.randint(len(x),(64,),generator=generator)
            optimizer.zero_grad(); loss=F.cross_entropy(model(x[idx])[:,-1],y[idx]); assert torch.isfinite(loss)
            loss.backward(); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            if step%500==0:
                metric=evaluate(model,data['validation']); curve.append(dict(step=step,validation=metric))
                print(json.dumps(dict(seed=seed,step=step,validation=metric)),flush=True)
        path=directory/f'seed_{seed}.pt'; torch.save(model.state_dict(),path)
        report['runs'].append(dict(seed=seed,parameters=5088,steps=3000,examples_processed=192000,tokens_processed=1536000,
            curve=curve,training=evaluate(model,data['train']),validation=evaluate(model,data['validation']),checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=digest(path)))
        output.write_text(json.dumps(report,indent=2)+'\n')
    # No test forward call occurs above this point.
    assert len(report['runs'])==5
    report['phase']='all_checkpoints_saved_before_test'; output.write_text(json.dumps(report,indent=2)+'\n')
    x,y=data['test']; rank=(x[:,:6:2]==x[:,-1,None]).long().argmax(1)
    for run in report['runs']:
        model=model_new().eval(); path=ROOT/run['checkpoint']; assert digest(path)==run['checkpoint_sha256']
        model.load_state_dict(torch.load(path,weights_only=True)); conditions={}; error_max=0.
        with torch.no_grad():
            for shift in (0,1,2,4,6,8):
                pieces={m:[] for m in ('original','pair')}; route={m:[] for m in pieces}
                for start in range(0,len(x),256):
                    chunk=x[start:start+256]; base,cache=forward(model,chunk,shift)
                    err=(base-logits(model,chunk,(torch.arange(8)+shift).expand_as(chunk))).abs().max().item(); error_max=max(error_max,err); assert err<1e-10
                    layer=model.blocks[0].self_attn; z=model.blocks[0].norm1(cache['r0'])
                    q,k,v=[t.reshape(len(chunk),8,2,8).transpose(1,2) for t in F.linear(z,layer.in_proj_weight,layer.in_proj_bias).chunk(3,-1)]
                    mask=torch.ones(8,8,dtype=torch.bool).triu(1)
                    w=(q@k.transpose(-2,-1)/(8**.5)).masked_fill(mask,float('-inf')).softmax(-1)
                    for row in (3,5): w[:,:,row]=0; w[:,:,row,row-1]=1
                    assert (w.sum(-1)-1).abs().max()<1e-12 and w.masked_select(mask[None,None].expand_as(w)).abs().max()==0
                    a=F.linear((w@v).transpose(1,2).reshape(len(chunk),8,16),layer.out_proj.weight,layer.out_proj.bias)
                    patched,pc=forward(model,chunk,shift,{'a0':a},('a0',))
                    for mode,s,c in [('original',base,cache),('pair',patched,pc)]:
                        q,k,v=project(model,c['r1']); err=(attention(model,q,k,v)-c['a1']).abs().max().item(); error_max=max(error_max,err); assert err<1e-10
                        p=((q[:,7].reshape(-1,2,8)[:,:,None,:]*k.reshape(-1,8,2,8).transpose(1,2)).sum(-1)/(8**.5)).softmax(-1)
                        pos=(2*rank[start:start+len(chunk)]+1)[:,None,None].expand(-1,2,1)
                        route[mode].append(p.gather(-1,pos).squeeze(-1)); pieces[mode].append(s)
                scores={m:torch.cat(v) for m,v in pieces.items()}; before=scores['original'].argmax(-1)==y
                for mode,s in scores.items():
                    assert torch.isfinite(s).all(); now=s.argmax(-1)==y
                    conditions[f'{shift}_{mode}']=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),
                        every_order_query_correct=now.reshape(400,18).all(1).double().mean().item(),
                        rescued=int((now&~before).sum()),harmed=int((~now&before).sum()),target_attention=torch.cat(route[mode]).mean(0).tolist())
        assert digest(path)==run['checkpoint_sha256']; run['test']=conditions; run['reconstruction_error']=error_max
        print(json.dumps(dict(seed=run['seed'],test={s:conditions[f'{s}_pair']['accuracy'] for s in (0,1,2,4,6,8)})),flush=True)
        output.write_text(json.dumps(report,indent=2)+'\n')
    report['criteria']=dict(original_zero_all_90=all(r['test']['0_original']['accuracy']>=.9 for r in report['runs']),
        compact_preserved=all(r['test']['0_original']['accuracy']-r['test']['0_pair']['accuracy']<=.01 for r in report['runs']),shifts={})
    for shift in (1,2,4,6,8):
        gain=statistics.mean(r['test'][f'{shift}_pair']['accuracy']-r['test'][f'{shift}_original']['accuracy'] for r in report['runs'])
        report['criteria']['shifts'][str(shift)]=dict(all_90=all(r['test'][f'{shift}_pair']['accuracy']>=.9 for r in report['runs']),mean_gain=gain,gain_5=gain>=.05)
    report['criteria']['all_shifts_pass']=all(c['all_90'] and c['gain_5'] for c in report['criteria']['shifts'].values())
    report['phase']='complete'; output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report['criteria']))


if __name__=='__main__': main()

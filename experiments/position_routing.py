"""Within-model offset donors separate block-2 routing and value effects."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from reserved_cycle import model_new
from activation_paths import forward
from qkv_paths import project, attention
from position_distractors import digest

ROOT = Path(__file__).resolve().parents[1]


def weights(q, k):
    b, n, d = q.shape
    q, k = [t.reshape(b, n, 2, 8).transpose(1, 2) for t in (q, k)]
    mask = torch.ones(n, n, dtype=torch.bool).triu(1)
    w = (q@k.transpose(-2, -1)/(8**.5)).masked_fill(mask, float('-inf')).softmax(-1)
    assert (w.sum(-1)-1).abs().max() < 1e-12
    assert w.masked_select(mask[None, None].expand_as(w)).abs().max() == 0
    return w


def measures(scores, y, original=None, donor_y=None):
    assert torch.isfinite(scores).all()
    pred = scores.argmax(-1)
    now = pred == y
    others = scores.clone()
    others.scatter_(1, y[:, None], float('-inf'))
    margin = scores.gather(1, y[:, None]).squeeze(1)-others.max(1).values
    result = dict(accuracy=now.double().mean().item(), loss=F.cross_entropy(scores, y).item(), mean_margin=margin.mean().item())
    if original is not None:
        before = original == y
        result.update(rescued=int((now & ~before).sum()), harmed=int((~now & before).sum()))
    if donor_y is not None:
        result['donor_answer_agreement'] = (pred == donor_y).double().mean().item()
    return result, dict(predictions=pred.tolist(), margins=margin.tolist())


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = ROOT/'results/position_routing.json'
    detailpath = ROOT/'results/position_routing_details.json'
    assert not out.exists() and not detailpath.exists()
    priorpath = ROOT/'results/position_training.json'
    prior = json.loads(priorpath.read_text())
    sp = ROOT/'results/position_training_splits.json'
    assert digest(sp) == prior['split_sha256']
    for name, sha in prior['source_sha256'].items():
        assert digest(ROOT/'experiments'/name) == sha
    rows = json.loads(sp.read_text())['test']
    x = torch.tensor([x for x, y in rows]); y = torch.tensor([y for x, y in rows])
    rank = (x[:, :6:2] == x[:, -1, None]).long().argmax(1)
    unrelated = (torch.arange(len(x))+18) % len(x)
    report = dict(prior_sha256=digest(priorpath), split_sha256=digest(sp),
                  protocol_sha256=digest(ROOT/'docs/position-routing-protocol.md'),
                  source_sha256={n: digest(ROOT/'experiments'/n) for n in ('position_routing.py', 'qkv_paths.py', 'activation_paths.py', 'reserved_cycle.py', 'compare_depth.py', 'position_distractors.py')},
                  unrelated_same_answer=(y == y[unrelated]).double().mean().item(), runs=[])
    details = dict(labels=y.tolist(), unrelated_indices=unrelated.tolist(), runs=[])
    with torch.no_grad():
        for old in prior['runs']:
            path = ROOT/old['checkpoint']
            assert digest(path) == old['checkpoint_sha256']
            model = model_new().eval(); model.load_state_dict(torch.load(path, weights_only=True))
            cache = {}; records = {}; detail = {}; error = 0.
            for shift in (4, 8):
                ss=[]; qq=[[], [], []]; aa=[]; paired=[]; target=[]
                for start in range(0, len(x), 256):
                    chunk=x[start:start+256]
                    s,c=forward(model, chunk, shift)
                    qkv=project(model, c['r1'])
                    a=attention(model, *qkv)
                    error=max(error, (a-c['a1']).abs().max().item())
                    sham=forward(model, chunk, shift, {'a1': a}, ('a1',))[0]
                    error=max(error, (sham-s).abs().max().item())
                    z=model.blocks[0].norm1(c['r0']); layer=model.blocks[0].self_attn
                    q,k,v=F.linear(z, layer.in_proj_weight, layer.in_proj_bias).chunk(3, -1)
                    w=weights(q,k)
                    paired.append(torch.stack([w[:,:,i,i-1] for i in (1,3,5)], -1))
                    w2=weights(qkv[0],qkv[1])[:,:,7]
                    pos=(2*rank[start:start+len(chunk)]+1)[:,None,None].expand(-1,2,1)
                    target.append(w2.gather(-1,pos).squeeze(-1))
                    ss.append(s); aa.append(a)
                    for parts,t in zip(qq,qkv): parts.append(t)
                scores=torch.cat(ss); pair=torch.cat(paired); route=torch.cat(target)
                cache[shift]=dict(scores=scores,qkv=tuple(torch.cat(t) for t in qq),a=torch.cat(aa))
                m,d=measures(scores,y)
                for k in ('accuracy','loss'): assert abs(m[k]-old['test'][str(shift)][k])<1e-10
                m.update(block1_paired_key_by_head_row=pair.mean(0).tolist(),block2_target_by_head=route.mean(0).tolist())
                d.update(block1_paired_key=pair.tolist(),block2_target=route.tolist())
                records[str(shift)]=m; detail[str(shift)]=d
            conditions={}
            for recipient,donor in ((8,4),(4,8)):
                for source,idx in [('matched',torch.arange(len(x))),('unrelated',unrelated)]:
                    for mode,replace in [('QK',(0,1)),('V',(2,)),('QKV',(0,1,2))]:
                        outputs=[]
                        for start in range(0,len(x),256):
                            end=start+256
                            qkv=[cache[donor]['qkv'][i][idx[start:end]] if i in replace else cache[recipient]['qkv'][i][start:end] for i in range(3)]
                            a=attention(model,*qkv)
                            if mode=='QKV': error=max(error,(a-cache[donor]['a'][idx[start:end]]).abs().max().item())
                            outputs.append(forward(model,x[start:end],recipient,{'a1':a},('a1',))[0])
                        name=f'{donor}_to_{recipient}_{source}_{mode}'
                        m,d=measures(torch.cat(outputs),y,cache[recipient]['scores'].argmax(-1),y[idx])
                        conditions[name]=m; detail[name]=d
            assert error<1e-10 and digest(path)==old['checkpoint_sha256']
            run=dict(seed=old['seed'],group=old['group'],checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],references=records,conditions=conditions,max_control_error=error)
            report['runs'].append(run); details['runs'].append(dict(seed=old['seed'],group=old['group'],conditions=detail))
            print(json.dumps(dict(seed=old['seed'],group=old['group'],rescue={n:m['accuracy'] for n,m in conditions.items() if n.startswith('4_to_8_matched')})),flush=True)
    detailpath.write_text(json.dumps(details,separators=(',',':'))+'\n')
    report['details_sha256']=digest(detailpath)
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()

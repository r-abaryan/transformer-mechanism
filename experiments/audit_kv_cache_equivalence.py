"""Independent arithmetic and cache construction, including intentional errors."""
import itertools
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from reserved_cycle import model_new
from position_distractors import digest

ROOT = Path(__file__).resolve().parents[1]


def norm(x, module):
    centered = x-x.mean(-1,keepdim=True)
    return centered/(centered.square().mean(-1,keepdim=True)+module.eps).sqrt()*module.weight+module.bias


def rotation(t, positions):
    angles = positions[:,None].double()*torch.tensor([1.,.1,.01,.001],dtype=torch.float64)
    c,s = angles.cos(),angles.sin()
    matrices = torch.stack((c,-s,s,c),-1).reshape(len(positions),4,2,2)
    return torch.einsum('trij,bhtrj->bhtri',matrices,t.reshape(t.shape[0],2,len(positions),4,2)).flatten(-2)


def independent(model, ids, offset, chunks, mode='correct'):
    history = [[] for b in model.blocks]
    output = []; cursor = 0
    for count in chunks:
        x = model.embedding.weight[ids[:,cursor:cursor+count]]
        positions = torch.arange(0 if mode=='reset' else cursor,
                                 count if mode=='reset' else cursor+count)+offset
        for i,block in enumerate(model.blocks):
            z = norm(x,block.norm1)
            projections = []
            for start in (0,16,32):
                w = block.self_attn.in_proj_weight[start:start+16]
                b = block.self_attn.in_proj_bias[start:start+16]
                projections.append((z@w.T+b).reshape(len(ids),count,2,8).permute(0,2,1,3))
            q,k,v = projections
            q,k = rotation(q,positions),rotation(k,positions)
            history[i].append((k,v))
            k = torch.cat([pair[0] for pair in history[i]],dim=2)
            v = torch.cat([pair[1] for pair in history[i]],dim=2)
            visible = torch.arange(cursor+count)[None,:] <= torch.arange(cursor,cursor+count)[:,None]
            if mode=='omit' and cursor:
                k,v,visible = k[:,:,1:],v[:,:,1:],visible[:,1:]
            attention = F.scaled_dot_product_attention(q,k,v,attn_mask=visible,dropout_p=0.)
            h = attention.permute(0,2,1,3).reshape(len(ids),count,16)
            x = x+h@block.self_attn.out_proj.weight.T+block.self_attn.out_proj.bias
            h = norm(x,block.norm2)@block.linear1.weight.T+block.linear1.bias
            x = x+h.clamp_min(0)@block.linear2.weight.T+block.linear2.bias
        output.append(norm(x,model.final_norm)@model.output.weight.T)
        cursor += count
    cache = [(torch.cat([p[0] for p in parts],2),torch.cat([p[1] for p in parts],2)) for parts in history]
    return torch.cat(output,1),cache


def metric(a,b):
    sorted_logits = a.sort(-1,descending=True).values
    gaps = sorted_logits[...,0]-sorted_logits[...,1]
    return dict(max_logit_error=float((a-b).abs().max()),
                argmax_differences=int((a.argmax(-1)!=b.argmax(-1)).sum()),
                min_reference_top2_gap=float(gaps.min()),reference_near_ties=int((gaps<=1e-10).sum()))


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    path = ROOT/'results/kv_cache_equivalence.json'
    report = json.loads(path.read_text()); assert report['phase']=='complete'
    for name,expected in report['provenance'].items():
        assert digest(ROOT/name)==expected
    prior_path = ROOT/'results/query_exposure.json'; assert digest(prior_path)==report['prior_sha256']
    prior = json.loads(prior_path.read_text())
    for name,expected in prior['provenance'].items():
        assert digest(ROOT/('docs/query-exposure-protocol.md' if name=='protocol' else 'experiments/'+name))==expected
    data_path = ROOT/'results/query_exposure_data.json'
    assert digest(data_path)==report['input_sha256']==prior['input_sha256']
    data = json.loads(data_path.read_text())['datasets']
    tensor_path = ROOT/'results/kv_cache_equivalence_tensors.pt'
    assert digest(tensor_path)==report['tensors_sha256']
    saved = torch.load(tensor_path,weights_only=True)
    assert len(saved)==len(report['cases'])==120 and report['rows']==[0,1,36,37]
    expected = set(itertools.product((1301,1321,1327,1361,1373),('repeated','all'),(3,4,6),(4,17),('single','prefill')))
    actual = {(c['seed'],c['group'],c['pairs'],c['offset'],c['schedule']) for c in report['cases']}
    assert actual==expected
    maxima = dict(logits=0.,kv=0.)
    details = []
    models = {}
    with torch.no_grad():
        for case, tensors in zip(report['cases'],saved):
            assert tensors['key']==case['key']
            run = next(r for r in prior['runs'] if (r['seed'],r['group'])==(case['seed'],case['group']))
            assert case['checkpoint']==run['checkpoint'] and case['checkpoint_sha256']==run['checkpoint_sha256']
            checkpoint = ROOT/run['checkpoint']; assert digest(checkpoint)==run['checkpoint_sha256']
            key = (case['seed'],case['group'])
            if key not in models:
                models[key] = model_new().eval()
                models[key].load_state_dict(torch.load(checkpoint,weights_only=True)['model'])
            model = models[key]
            rows = [data[f"test_{case['pairs']}"][i] for i in (0,1,36,37)]
            ids = torch.tensor([x for x,y in rows]);labels = torch.tensor([y for x,y in rows])
            n = ids.shape[1]
            first = 1 if case['schedule']=='single' else n//2
            chunks = [first]+[1]*(n-first)
            full,full_cache = independent(model,ids,case['offset'],[n])
            err = float((full-tensors['full']).abs().max()); assert err<=1e-10
            maxima['logits'] = max(maxima['logits'],err)
            for name,mode in [('cached','correct'),('reset','reset'),('omit','omit')]:
                scores,cache = independent(model,ids,case['offset'],chunks,mode)
                err = float((scores-tensors[name]).abs().max()); assert err<=1e-10
                maxima['logits'] = max(maxima['logits'],err)
                recorded = case['correct' if name=='cached' else name]
                measured = metric(tensors['full'],tensors[name])
                assert measured==recorded
                if name=='cached':
                    for layer in range(2):
                        for component in range(2):
                            tensor = tensors['cache'][layer][component]
                            for ref in (cache[layer][component],full_cache[layer][component]):
                                error = float((ref-tensor).abs().max()); assert error<=1e-10
                                maxima['kv'] = max(maxima['kv'],error)
            assert int((tensors['full'][:,-1].argmax(-1)==labels).sum())==case['reference_task_correct']
            assert int((tensors['cached'][:,-1].argmax(-1)==labels).sum())==case['cached_task_correct']
            cursor = 0; assert len(case['trace'])==len(chunks)
            for chunk, trace in zip(chunks,case['trace']):
                cursor += chunk
                assert trace==dict(tokens=cursor,shapes=[[4,2,cursor,8]]*2,payload_bytes=4*2*2*cursor*16*8)
            assert sum(t.numel()*t.element_size() for pair in tensors['cache'] for t in pair)==case['trace'][-1]['payload_bytes']
            details.append(case['key'])
    checks = dict(equivalence=all(c['correct']['max_logit_error']<=1e-10 for c in report['cases']),
                  reset_detected=any(c['reset']['max_logit_error']>1e-6 for c in report['cases']),
                  omission_detected=any(c['omit']['max_logit_error']>1e-6 for c in report['cases']),
                  cache_lengths_and_payload=True)
    assert checks==report['checks'] and all(checks.values())==report['passed']
    result = dict(passed=True,cases=len(details),checks=checks,max_independent_logit_error=maxima['logits'],
                  max_independent_kv_error=maxima['kv'],report_sha256=digest(path),
                  source_sha256=digest(Path(__file__)))
    (ROOT/'results/kv_cache_equivalence_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()

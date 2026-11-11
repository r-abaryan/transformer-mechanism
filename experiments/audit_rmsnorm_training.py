"""Independent centering arithmetic, data/stream verification and final audit."""
import hashlib
import itertools
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from rmsnorm_training import ROOT,SEEDS,PRIORS,model_new,parameter_hash
from audit_mqa_training import audit_inputs as previous_data_audit
from audit_kv_cache_equivalence import rotation
from position_distractors import digest


def normalization(x,layer,group):
    z=x-x.mean(-1,keepdim=True) if group=='ln' else x
    return z/(z.square().sum(-1,keepdim=True)/x.shape[-1]+layer.eps).sqrt()*layer.weight


def independent(model,ids,group,force_ln=False):
    if force_ln:group='ln'
    x=model.embedding.weight[ids];n=ids.shape[1];positions=torch.arange(n)+4
    for b in model.blocks:
        z=normalization(x,b.norm1,group);m=b.attn
        q,k,v=[z@getattr(m,t+'_weight').T+getattr(m,t+'_bias') for t in ('q','k','v')]
        q=q.reshape(len(ids),n,2,8).permute(0,2,1,3)
        k,v=[t.reshape(len(ids),n,1,8).permute(0,2,1,3).expand(len(ids),2,n,8) for t in (k,v)]
        q,k=rotation(q,positions),rotation(k,positions)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True).permute(0,2,1,3).reshape(len(ids),n,16)
        x=x+y@m.out_proj.weight.T+m.out_proj.bias
        y=normalization(x,b.norm2,group)@b.linear1.weight.T+b.linear1.bias
        x=x+y.clamp_min(0)@b.linear2.weight.T+b.linear2.bias
    return normalization(x,model.final_norm,group)@model.output.weight.T


def audit_inputs(payload):
    assert set(payload['prior_hashes'])==set(PRIORS) and payload['generator_seed']==20270110
    previous={k:v for k,v in payload.items()}
    previous['prior_hashes']={k:v for k,v in payload['prior_hashes'].items() if k!='mqa_training_data.json'}
    checks=previous_data_audit(previous)
    old=set()
    for name,h in payload['prior_hashes'].items():
        p=ROOT/'results'/name;assert digest(p)==h;prior=json.loads(p.read_text())
        old.update(tuple(tuple(pair) for pair in f) for fs in prior['families'].values() for f in fs)
    fresh={tuple(tuple(pair) for pair in f) for name,fs in payload['families'].items() if not name.endswith('_3') for f in fs}
    assert not fresh&old
    checks['excluded_prior_families']=len(old);return checks


def expanded_fixture(rows):
    expanded=[]
    for ids,label in rows:
        pairs=list(zip(ids[:-2:2],ids[1:-2:2]));keys=[ids[-1]]+sorted({k for k,v in pairs}-{ids[-1]})
        for key in keys:expanded.append((ids[:-1]+[key],next(v for k,v in reversed(pairs) if k==key)))
    return torch.tensor([x for x,y in expanded]),torch.tensor([y for x,y in expanded])


def metrics(scores,labels,ids):
    n=(ids.shape[1]-2)//2;correct=scores.argmax(-1)==labels;repeated=(ids[:,:-2:2]==ids[:,-1:]).sum(-1)==2
    def ce(s,y):return float((s.logsumexp(-1)-s.gather(1,y[:,None]).squeeze(1)).mean())
    result=dict(accuracy=float(correct.double().mean()),loss=ce(scores,labels),all_query_context_accuracy=float(correct.reshape(-1,n-1).all(1).double().mean()))
    for name,mask in (('repeated',repeated),('unique',~repeated)):
        result[name+'_accuracy']=float(correct[mask].double().mean());result[name+'_loss']=ce(scores[mask],labels[mask])
    return result


def decision(runs):
    if len(runs)!=10 or any(x['status']!='trained' for x in runs):return dict(keep=False,reason='training failure or incomplete attempts')
    checks=dict(equal_parameters=all(r['parameters']==4464 for r in runs));gains={}
    for n in (3,4,6):
        for kind in ('repeated','unique'):
            differences=[]
            for seed in SEEDS:
                a=next(r for r in runs if r['seed']==seed and r['group']=='ln');b=next(r for r in runs if r['seed']==seed and r['group']=='rms')
                differences.append(b['test'][str(n)][kind+'_accuracy']-a['test'][str(n)][kind+'_accuracy'])
            gains[f'{n}_{kind}']=differences
            checks[f'{n}_{kind}_floor']=all(r['test'][str(n)][kind+'_accuracy']>=(.9 if n==6 else .99) for r in runs if r['group']=='rms')
            if n==6:
                checks[f'6_{kind}_mean']=sum(differences)>=5*(.01 if kind=='unique' else -.005)
                checks[f'6_{kind}_seed_loss1']=all(g>=-.01 for g in differences)
    return dict(keep=all(checks.values()),checks=checks,paired_gains=gains)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rp=ROOT/'results/rmsnorm_training.json';ap=ROOT/'results/rmsnorm_training_audit.json';assert not ap.exists()
    r=json.loads(rp.read_text());assert r['phase']=='complete'
    for p,h in r['provenance']['sources'].items():assert digest(ROOT/p)==h
    for name,h in r['provenance']['baseline_source_hashes'].items():assert digest(ROOT/('docs/mqa-training-protocol.md' if name=='protocol' else 'experiments/'+name))==h
    assert digest(ROOT/'results/mqa_training.json')==r['provenance']['baseline_sha256']
    ip=ROOT/'results/rmsnorm_training_data.json';assert digest(ip)==r['input_sha256'];payload=json.loads(ip.read_text());inputs=audit_inputs(payload)
    assert inputs==r['input_checks']
    vp=ROOT/'results/rmsnorm_training_verification.json';assert digest(vp)==r['verification_sha256'];verification=json.loads(vp.read_text())
    assert verification['passed'] and verification['provenance']==r['provenance']
    assert {(x['seed'],x['group']) for x in r['runs']}==set(itertools.product(SEEDS,('ln','rms')))
    for seed in SEEDS:
        pair=[x for x in r['runs'] if x['seed']==seed]
        for key in ('initial_sha256','stream3_sha256','stream4_sha256'):assert len({x[key] for x in pair})==1
        for run in pair:
            torch.manual_seed(seed);m=model_new(run['group']);assert parameter_hash(m)==run['initial_sha256']
            assert sum(p.numel() for p in m.parameters())==run['parameters']==4464
        for n,offset in ((3,1000),(4,2000)):
            generator=torch.Generator().manual_seed(seed+offset);stream=hashlib.sha256();repeated=0
            for step in range(1,6001):
                idx=torch.randint(len(payload['datasets'][f'train_{n}'])*(n-1),(64,),generator=generator);stream.update(json.dumps(idx.tolist()).encode())
                if (n==3 and step%2) or (n==4 and not step%2):repeated+=int((idx%(n-1)==0).sum())
            for run in pair:assert run[f'stream{n}_sha256']==stream.hexdigest() and run['query_counts'][str(n)]==dict(repeated=repeated,unique=192000-repeated)
    dp=ROOT/'results/rmsnorm_training_scores.pt';assert digest(dp)==r['scores_sha256'];saved=torch.load(dp,weights_only=True);errors=[]
    for run in r['runs']:
        if run['status']!='trained':continue
        cp=ROOT/run['checkpoint'];assert digest(cp)==run['checkpoint_sha256'];state=torch.load(cp,weights_only=True)
        assert all(v['step'].item()==6000 for v in state['optimizer']['state'].values())
        assert run['tokens']==3456000 and run['attention_pairs']==31488000 and run['step']==6000
        assert [p['step'] for p in run['curve']]==list(range(0,6001,1000))
        milestone=next((p['step'] for p in run['curve'] if all(p['validation'][str(n)][k+'_accuracy']>=.99 for n in (3,4) for k in ('repeated','unique'))),None)
        assert milestone==run['sampled_validation99_step']
        assert all(p['finite_training'] for p in run['curve'][1:])
        m=model_new(run['group']).eval();m.load_state_dict(state['model']);assert all(torch.isfinite(p).all() for p in m.parameters())
        detail=next(s for s in saved if (s['seed'],s['group'])==(run['seed'],run['group']))
        for n in (3,4,6):
            ids,labels=expanded_fixture(payload['datasets'][f'test_{n}'])
            with torch.inference_mode():scores=torch.cat([independent(m,c,run['group'])[:,-1] for c in ids.split(256)])
            error=float((scores-detail['scores'][str(n)]).abs().max());assert error<=1e-10;errors.append(error)
            actual=metrics(scores,labels,ids)
            assert all(abs(v-run['test'][str(n)][k])<=1e-10 for k,v in actual.items())
        print(f"Audited {run['seed']} / {run['group']}",flush=True)
    assert decision(r['runs'])==r['decision']
    result=dict(passed=True,evaluations=len(errors),max_logit_error=max(errors),input_checks=inputs,report_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    ap.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

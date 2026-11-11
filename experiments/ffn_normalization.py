"""Exact final-normalization decomposition of last-FFN removal."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from reserved_cycle import model_new
from activation_paths import forward
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    out=ROOT/'results/ffn_normalization.json'; dp=ROOT/'results/ffn_normalization_scores.json'
    assert not out.exists() and not dp.exists()
    pp=ROOT/'results/ffn_branches.json'; prior=json.loads(pp.read_text()); sp=ROOT/'results/canonical_positions_inputs.json'
    assert digest(sp)==prior['inputs_sha256']
    for n,h in prior['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    rows=json.loads(sp.read_text()); x=torch.tensor([x for x,y in rows]); y=torch.tensor([y for x,y in rows])
    report=dict(prior_sha256=digest(pp),inputs_sha256=digest(sp),protocol_sha256=digest(ROOT/'docs/ffn-normalization-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in ('ffn_normalization.py','reserved_cycle.py','compare_depth.py','activation_paths.py','position_distractors.py')},runs=[]); details=[]
    with torch.no_grad():
        for old in prior['runs']:
            path=ROOT/old['checkpoint']; assert digest(path)==old['checkpoint_sha256']
            model=model_new().eval(); model.load_state_dict(torch.load(path,weights_only=True)); pieces={n:[] for n in ('ordinary','remove_live_stats','remove_frozen_stats')}; direct=[]; stats=[]; error=0.
            for chunk in x.split(256):
                s,c=forward(model,chunk,4); u=(c['r1']+c['a1'])[:,-1]; f=c['f1'][:,-1]; b=u+f; ln=model.final_norm
                mu=b.mean(-1,keepdim=True); sigma=(b.var(-1,unbiased=False,keepdim=True)+ln.eps).sqrt()
                reconstructed=model.output(ln.weight*(b-mu)/sigma+ln.bias)
                frozen=model.output(ln.weight*(u-mu)/sigma+ln.bias); live=model.output(ln(u)); term=model.output(ln.weight*f/sigma)
                change=frozen-live
                error=max(error,(s-reconstructed).abs().max().item(),(s-frozen-term).abs().max().item(),(s-live-term-change).abs().max().item())
                direct.append(term); stats.append(change)
                for n,t in [('ordinary',s),('remove_live_stats',live),('remove_frozen_stats',frozen)]: pieces[n].append(t)
            scores={n:torch.cat(ts) for n,ts in pieces.items()}; conditions={}
            for n,s in scores.items():
                assert torch.isfinite(s).all(); conditions[n]=dict(accuracy=(s.argmax(-1)==y).double().mean().item(),loss=F.cross_entropy(s,y).item())
            for n,previous in [('ordinary','ordinary'),('remove_live_stats','zero_second')]:
                for k in ('accuracy','loss'): assert abs(conditions[n][k]-old['conditions'][previous][k])<1e-10
            assert error<1e-10 and digest(path)==old['checkpoint_sha256']
            report['runs'].append(dict(seed=old['seed'],conditions=conditions,identity_error=error,direct_logit_rms=torch.cat(direct).square().mean().sqrt().item(),statistics_change_logit_rms=torch.cat(stats).square().mean().sqrt().item()))
            details.append(dict(seed=old['seed'],scores={n:s.tolist() for n,s in scores.items()}))
    dp.write_text(json.dumps(details,separators=(',',':'))+'\n'); report['scores_sha256']=digest(dp)
    # Reload persisted scores and recompute using sum-based loss, separate from cross_entropy.
    audit_error=0.
    for run,d in zip(report['runs'],json.loads(dp.read_text())):
        assert run['seed']==d['seed']
        for n,values in d['scores'].items():
            s=torch.tensor(values,dtype=torch.float64); loss=(s.logsumexp(-1)-s[torch.arange(len(y)),y]).mean().item(); acc=(s.argmax(-1)==y).sum().item()/len(y)
            audit_error=max(audit_error,abs(loss-run['conditions'][n]['loss']),abs(acc-run['conditions'][n]['accuracy']))
    assert audit_error<1e-12
    report.update(saved_logit_metric_error=audit_error,phase='complete'); out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report['runs']))


if __name__=='__main__': main()

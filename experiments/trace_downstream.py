"""Frozen-attention residual, FFN and normalization pathway decomposition."""
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from query_position_paths import sample, positional, patched
from inspect_first_token import metrics
from verify_decoder import TinyDecoder

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    rows=sample(20261001,1024,set(sample(20260930,512,set())))
    ids=torch.tensor([list(r)+[4] for r in rows])
    target=torch.tensor([r[0] for r in rows])
    training=json.loads((ROOT/'results/first_token_training.json').read_text())
    data_hash=hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()
    assert data_hash==json.loads((ROOT/'results/query_position_paths.json').read_text())['data_sha256']
    report=dict(scope='Exploratory reuse of pathway data; attention held fixed',runs=[],
                data_sha256=data_hash,torch=torch.__version__,device='cpu',dtype='float64',
                protocol_sha256=hashlib.sha256((ROOT/'docs/downstream-protocol.md').read_bytes()).hexdigest(),
                source_sha256={n:hashlib.sha256((ROOT/'experiments'/n).read_bytes()).hexdigest() for n in
                               ('trace_downstream.py','query_position_paths.py','inspect_first_token.py','verify_decoder.py')})
    with torch.no_grad():
        for run in training['runs']:
            path=ROOT/run['checkpoint']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==run['checkpoint_sha256']
            model=TinyDecoder(5,16,2,32).double().eval()
            model.load_state_dict(torch.load(path,weights_only=True))
            captured={}
            def capture(module,args):
                captured['u']=args[0][:,-1].clone()
            hook=model.block.norm2.register_forward_pre_hook(capture)
            try:
                baseline=model(ids)[:,-1]
            finally:
                hook.remove()
            u=captured['u']
            donor=u+positional(6,16)-positional(10,16)
            def f(x):
                return model.block.linear2(F.relu(model.block.linear1(model.block.norm2(x))))
            f0,f1=f(u),f(donor)
            blocks=dict(baseline=u+f0,full=donor+f1,skip_only=donor+f0,ffn_only=u+f1)
            correct=baseline.argmax(-1)==target
            competitor_scores=baseline.clone()
            competitor_scores[torch.arange(len(target)),target]=-torch.inf
            competitor=competitor_scores.argmax(-1)
            weight_difference=model.output.weight[target]-model.output.weight[competitor]
            norm=model.final_norm
            mu0=blocks['baseline'].mean(-1,keepdim=True)
            std0=torch.sqrt(blocks['baseline'].var(-1,keepdim=True,unbiased=False)+norm.eps)
            results={}
            components={}
            max_error=0.0
            def measure(logits):
                result=metrics(logits,target,correct)
                others=logits.clone()
                others[torch.arange(len(target)),target]=-torch.inf
                margin=logits[torch.arange(len(target)),target]-others.max(-1).values
                result['mean_margin']=margin.mean().item()
                result['mean_margin_original_errors']=margin[~correct].mean().item() if (~correct).any() else None
                return result
            for name,b in blocks.items():
                mu=b.mean(-1,keepdim=True)
                std=torch.sqrt(b.var(-1,keepdim=True,unbiased=False)+norm.eps)
                logits=model.output(norm(b))
                results[name]=measure(logits)
                if name=='baseline':
                    max_error=max(max_error,(logits-baseline).abs().max().item())
                if name=='full':
                    max_error=max(max_error,(logits-patched(model,ids,6,'residual')).abs().max().item())
                for variant,m,s in [('old_mean',mu0,std),('old_std',mu,std0),('old_both',mu0,std0)]:
                    results[f'{name}_{variant}']=measure(model.output((b-m)/s*norm.weight+norm.bias))
                raw=(weight_difference*norm.weight*b).sum(-1)
                centering=(-mu[:,0]*(weight_difference*norm.weight).sum(-1))
                bias=(weight_difference*norm.bias).sum(-1)
                total=(raw+centering)/std[:,0]+bias
                actual=logits[torch.arange(len(target)),target]-logits[torch.arange(len(target)),competitor]
                max_error=max(max_error,(total-actual).abs().max().item())
                components[name]={}
                for label,selection in [('all',torch.ones_like(correct)),('original_errors',~correct)]:
                    components[name][label]=None if not selection.any() else dict(
                        raw_scaled=(raw/std[:,0])[selection].mean().item(),
                        centering_scaled=(centering/std[:,0])[selection].mean().item(),
                        learned_bias=bias[selection].mean().item(),fixed_competitor_margin=total[selection].mean().item())
            # Intermediate margins omit the final norm and are diagnostics only.
            diagnostics={name:measure(model.output(b)) for name,b in dict(pre_ffn=u,post_ffn=blocks['baseline'],
                                                                          donor_pre_ffn=donor,donor_post_ffn=blocks['full']).items()}
            assert max_error<1e-10,max_error
            record=dict(seed=run['seed'],checkpoint_sha256=run['checkpoint_sha256'],verification_error=max_error,
                        results=results,components=components,unnormalized_diagnostics=diagnostics)
            report['runs'].append(record)
            print(json.dumps(dict(seed=run['seed'],results={k:results[k] for k in ('baseline','full','skip_only','ffn_only','full_old_both')})),flush=True)
    (ROOT/'results/downstream_trace.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()

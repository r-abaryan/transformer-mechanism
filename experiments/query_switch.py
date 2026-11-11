"""Fixed-context query changes on five saved narrow models; no training."""
import json
from pathlib import Path
import torch
from ffn_width import ROOT,model_new,forward
from audit_ffn_width import independent
from position_distractors import digest


def main():
    torch.set_num_threads(1)
    rp=ROOT/'results/ffn_width.json'; ip=ROOT/'results/ffn_width_data.json'; sp=ROOT/'results/ffn_width_scores.pt'
    prior=json.loads(rp.read_text());assert digest(ip)==prior['input_sha256'] and digest(sp)==prior['scores_sha256']
    for name,h in prior['provenance'].items():assert digest(ROOT/('docs/ffn-width-protocol.md' if name=='protocol' else 'experiments/'+name))==h
    rows=json.loads(ip.read_text())['datasets']['test_4'];inputs=[];labels=[]
    for x,y in rows:
        pairs=list(zip(x[:-2:2],x[1:-2:2])); queries=[x[-1]]+sorted(set(k for k,v in pairs)-{x[-1]});assert len(queries)==3
        mapping=dict(pairs)
        for q in queries:
            z=x[:-1]+[q]; answer=mapping[q]
            assert z[:-1]==x[:-1] and answer==next(v for k,v in pairs[::-1] if k==q)
            if q!=x[-1]:assert answer!=y
            inputs.append(z);labels.append(answer)
    x=torch.tensor(inputs);y=torch.tensor(labels).reshape(-1,3); saved=torch.load(sp,weights_only=True);runs=[];details=[]
    with torch.no_grad():
        for run in prior['runs']:
            if run['group']!='ff32':continue
            cp=ROOT/run['checkpoint'];assert digest(cp)==run['checkpoint_sha256']
            model=model_new('ff32').eval();model.load_state_dict(torch.load(cp,weights_only=True)['model'])
            logits=torch.cat([forward(model,c,'ff32')[:,-1] for c in x.split(256)]).reshape(-1,3,19)
            check=torch.cat([independent(model,c,'ff32') for c in x.split(256)]).reshape_as(logits)
            error=(logits-check).abs().max().item();assert error<1e-10
            archived=next(d for d in saved if d['seed']==run['seed'] and d['group']=='ff32')['scores']['4']
            old_error=(logits[:,0]-archived).abs().max().item();assert old_error<1e-10
            pred=logits.argmax(-1);correct=pred==y
            result=dict(seed=run['seed'],checkpoint_sha256=digest(cp),original_accuracy=correct[:,0].double().mean().item(),switched_accuracy=correct[:,1:].double().mean().item(),unchanged_prediction_rate=(pred[:,1:]==pred[:,:1]).double().mean().item(),repeated_key_answer_rate=(pred[:,1:]==y[:,:1]).double().mean().item(),all_query_context_accuracy=correct.all(1).double().mean().item(),max_logit_change=(logits[:,1:]-logits[:,:1]).abs().max().item(),independent_error=error,archived_error=old_error)
            runs.append(result);details.append(dict(seed=run['seed'],logits=logits));print(json.dumps(result),flush=True)
    out=ROOT/'results/query_switch_scores.pt';torch.save(dict(inputs=x,labels=y,runs=details),out)
    report=dict(passed=True,contexts=len(rows),families=200,switched_per_seed=2*len(rows),runs=runs,reliable_transfer=all(r['switched_accuracy']>=.9 for r in runs),provenance=dict(source=digest(Path(__file__)),protocol=digest(ROOT/'docs/query-switch-protocol.md'),training_report=digest(rp),data=digest(ip),prior_scores=digest(sp),scores=digest(out),independent_source=digest(ROOT/'experiments/audit_ffn_width.py')))
    (ROOT/'results/query_switch.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

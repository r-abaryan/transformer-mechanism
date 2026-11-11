"""Checkpoint-only reset counterfactuals and retention interventions."""
import copy
import json
import torch
from learned_memory_control import ROOT, pair, encode, metrics, GROUPS
from position_distractors import digest
from recurrent_retrieval import write


def trace(m,x,q,mode='native',independent=False):
    linear=lambda x,l:x@l.weight.T+l.bias
    raw=linear(x,m.key);k=raw/raw.square().sum(-1,keepdim=True).sqrt()
    v=linear(x,m.value);b=linear(x,m.beta).sigmoid()
    a=linear(x,m.alpha).sigmoid() if m.alpha is not None else torch.ones_like(b)
    if mode=='all':a=torch.ones_like(a)
    if mode=='reset':a=torch.where(x[:,:,7:8]==1,torch.ones_like(a),a)
    s=x.new_zeros(len(x),8,8);states=[];eye=torch.eye(8,dtype=x.dtype)
    for i in range(x.shape[1]):
        kk=k[:,i];aa=a[:,i,:,None];bb=b[:,i,:,None]
        if independent:
            s=aa*(eye-bb*kk[:,:,None]*kk[:,None,:])@s+bb*kk[:,:,None]*v[:,i,None,:]
        else:
            p=aa*s;old=(kk[:,:,None]*p).sum(1)
            s=p+bb*kk[:,:,None]*(v[:,i]-old)[:,None,:]
        states.append(s)
    read=(m.query.weight[q][:,None,:]@s).squeeze(1)
    return linear(read,m.output),torch.stack(states,1),a,b,k


def checked(m,rows,mode):
    x,q,y=encode(rows)
    a=trace(m,x,q,mode);b=trace(m,x,q,mode,True)
    oe=float((a[0]-b[0]).abs().max());se=float((a[1]-b[1]).abs().max());changes=int((a[0].argmax(-1)!=b[0].argmax(-1)).sum())
    assert max(oe,se)<=1e-10 and changes==0
    zero=m.output.bias.expand_as(a[0]);negative=float((a[0]-zero).abs().max());assert negative>1e-6
    return a,dict(predictions=len(rows),output_error=oe,state_error=se,argmax_changes=changes,zero_state_effect=negative)


def paired_rows(rows):
    originals=[r for r in rows if r['group'].startswith('reset')];other=copy.deepcopy(originals)
    positions=[]
    for first,second in zip(originals,other):
        pos=next(i for i,(k,v) in enumerate(first['events']) if k==3);positions.append(pos)
        for i in range(pos):second['events'][i][1]=(second['events'][i][1]+1)%4
        assert second['events'][pos:]==first['events'][pos:] and second['events'][:pos]!=first['events'][:pos]
        for row in (first,second):
            label=4
            for key,value in reversed(row['events']):
                if key==3:break
                if key==row['query']:label=value;break
            assert label==row['label']
    return originals,other,torch.tensor(positions)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    target=ROOT/'results/memory_reset_diagnostic.json';assert not target.exists()
    prior=json.loads((ROOT/'results/learned_memory_control.json').read_text());assert prior['phase']=='complete'
    data=json.loads((ROOT/'results/learned_memory_control_data.json').read_text())['pools']
    history={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'results').glob('*.json')}
    files=['experiments/memory_reset_diagnostic.py','experiments/learned_memory_control.py','docs/protocols/memory-reset-diagnostic.md']
    report=dict(status='running',training_updates=0,device='cpu',dtype='float64',provenance={p:digest(ROOT/p) for p in files},historical=history,checkpoints={},evaluations=[],pairs=[]);write(target,report)
    try:
        with torch.no_grad():
            for run in prior['runs']:
                cp=ROOT/run['checkpoint'];assert digest(cp)==run['checkpoint_sha256'];report['checkpoints'][run['checkpoint']]=digest(cp)
                m=pair(run['seed'])[run['arm']];m.load_state_dict(torch.load(cp,weights_only=True)['model'])
                assert digest(ROOT/run['scores'])==run['scores_sha256'];saved=torch.load(ROOT/run['scores'],weights_only=True)
                for split,rows in data.items():
                    if not split.startswith('test'):continue
                    native,check=checked(m,rows,'native')
                    archival=float((native[0]-saved[split]).abs().max());assert archival<=1e-10 and torch.equal(native[0].argmax(-1),saved[split].argmax(-1))
                    for mode in ('native','all','reset') if run['arm']=='gated' else ('native',):
                        outputs,audit=(native,check) if mode=='native' else checked(m,rows,mode)
                        score=outputs[0];group_changes={g:int(((score.argmax(-1)!=native[0].argmax(-1))&torch.tensor([r['group']==g for r in rows])).sum()) for g in GROUPS}
                        report['evaluations'].append(dict(seed=run['seed'],arm=run['arm'],split=split,mode=mode,metrics=metrics(score,rows),prediction_changes=group_changes,audit=audit,archive_error=archival))
                    original,other,pos=paired_rows(rows);left,lc=checked(m,original,'native');right,rc=checked(m,other,'native');ix=torch.arange(len(original))
                    difference=left[1]-right[1];before=difference[ix,pos-1];after=difference[ix,pos]
                    alpha=left[2][ix,pos];beta=left[3][ix,pos];key=left[4][ix,pos]
                    transport=alpha[:,:,None]*(torch.eye(8,dtype=torch.float64)-beta[:,:,None]*key[:,:,None]*key[:,None,:])@before
                    te=float((transport-after).abs().max());assert te<=1e-10
                    distances={'before':before.flatten(1).norm(dim=1),'after':after.flatten(1).norm(dim=1),'final':difference[:,-1].flatten(1).norm(dim=1)}
                    for group in ('reset_present','reset_missing'):
                        mask=torch.tensor([r['group']==group for r in original]);pre=distances['before'][mask];post=distances['after'][mask]
                        summaries={stage:dict(mean=float(d[mask].mean()),maximum=float(d[mask].max()),equal_count=int((d[mask]<=1e-10).sum())) for stage,d in distances.items()}
                        logitdiff=(left[0]-right[0]).abs().amax(-1)
                        report['pairs'].append(dict(seed=run['seed'],arm=run['arm'],split=split,group=group,count=int(mask.sum()),state_distances=summaries,nonzero_before=int((pre>1e-10).sum()),mean_after_before_ratio=float((post[pre>1e-10]/pre[pre>1e-10]).mean()),max_logit_difference=float(logitdiff[mask].max()),prediction_agreement=float((left[0][mask].argmax(-1)==right[0][mask].argmax(-1)).double().mean()),original=metrics(left[0],original)[group],counterfactual=metrics(right[0],other)[group],reset_alpha=float(alpha[mask].mean()),reset_beta=float(beta[mask].mean()),transport_error=te,left_audit=lc,right_audit=rc))
                print(dict(seed=run['seed'],arm=run['arm'],done=True),flush=True)
        assert all(digest(ROOT/p)==h for p,h in history.items())
        assert all(digest(ROOT/p)==h for p,h in report['checkpoints'].items())
        report.update(status='complete',passed=True)
    except Exception as exc:report.update(status='failed',failure=repr(exc));raise
    finally:write(target,report)
    print(dict(evaluations=len(report['evaluations']),pair_groups=len(report['pairs']),passed=True))


if __name__=='__main__':main()

"""Independent donor scope, forward recurrence and factorial contrast audit."""
import copy
import itertools
import json
import torch
import beta_key_transport as study
import audit_write_interference_stages as ref


@torch.no_grad()
def audit(out,persist=True):
    prior=study.prior.prior;lock=json.loads((out/'lock.json').read_text());report=json.loads((out/'report.json').read_text());prior.check_files(lock['files'])
    assert lock['steps']==[3000,12000] and lock['guard']==1e-9 and lock['versions']==prior.versions()
    assert report['status']=='complete' and all(report[k]==0 for k in ('training_updates','optimizer_calls','autograd_calls','solver_calls'))
    assert report['lock_sha256']==prior.digest(out/'lock.json') and report['traces_sha256']==prior.digest(out/'traces.pt')
    bundles=torch.load(out/'traces.pt',weights_only=True);assert len(bundles)==len(lock['cases'])==len(report['cells'])
    ids=[study.identity(c) for c in lock['cases']];assert len(set(ids))==len(ids)
    seeds=sorted({c['seed'] for c in lock['cases']});expected=list(itertools.product(seeds,(3000,12000),(3000,12000),(3000,12000)))
    assert ids==expected,'Factorial coverage/order'
    cached=None if lock['fixture'] else torch.load(study.prior.OUT/'traces.pt',weights_only=True)
    checked=[];maximum=0.;queryrows=outputs=writes=0
    for case,b,cell in zip(lock['cases'],bundles,report['cells']):
        assert b['case']==case==cell['case']
        assert (b['seed'],b['step'])==(case['seed'],case['anchor'])
        payloads={int(t):torch.load(study.ROOT/p,weights_only=True) for t,p in case['checkpoints'].items()}
        assert all(p['step']==t for t,p in payloads.items())
        with torch.random.fork_rng(devices=[]):model=prior.historical.parent.previous.load_model(study.ROOT/case['checkpoints'][str(case['anchor'])]).eval()
        state={k:v.clone() for k,v in payloads[case['anchor']]['model'].items()}
        state['key.weight']=payloads[case['keys']]['model']['key.weight'].clone();state['write_logit']=payloads[case['beta']]['model']['write_logit'].clone()
        model.load_state_dict(state)
        for name,value in model.state_dict().items():assert torch.equal(value,state[name]),'Independent donor state'
        rows=json.loads((study.ROOT/case['data']).read_text(encoding='utf-8-sig'))['pools']['train_32']
        if cached is None:
            with torch.random.fork_rng(devices=[]):endpoint_model=prior.historical.parent.previous.load_model(study.ROOT/case['checkpoints']['12000']).eval()
            _,labels,_,_,_,z=ref.proof.independent_features(endpoint_model,rows);endpoint=z.argmax(-1)==labels
        else:endpoint=next(x for x in cached if x['seed']==case['seed'] and x['step']==12000)['endpoint_correct']
        q,y,z,coef,fields=ref.independent(model,rows,endpoint)
        assert torch.equal(q,b['queries']) and torch.equal(y,b['labels']) and torch.equal(endpoint,b['endpoint_correct'])
        assert torch.equal(z.argmax(-1),b['logits'].argmax(-1))
        for actual,wanted in ((b['logits'],z),(b['coefficients'],coef)):
            torch.testing.assert_close(actual,wanted,atol=1e-9,rtol=1e-9);maximum=max(maximum,float((actual-wanted).abs().max()))
        with torch.random.fork_rng(devices=[]):native=prior.historical.parent.previous.load_model(study.ROOT/case['checkpoints'][str(case['anchor'])]).eval()
        _,_,nativez,_,nativefields=ref.independent(native,rows,endpoint)
        if cached is not None:
            old=next(x for x in cached if x['seed']==case['seed'] and x['step']==case['anchor'])
            torch.testing.assert_close(nativez,old['logits'],atol=1e-9,rtol=1e-9)
            assert torch.equal(nativez.argmax(-1),old['logits'].argmax(-1)) and y.shape==(768,32)
        nativecorrect=nativez[-1].argmax(-1)==y;good=z[-1].argmax(-1)==y
        fields.update(final_ce=torch.logsumexp(z[-1],-1)-z[-1].gather(-1,y[...,None]).squeeze(-1),
            rescued=~nativecorrect&good,harmed=nativecorrect&~good,ratio_change=fields['target_coefficient_ratio']-nativefields['target_coefficient_ratio'])
        assert set(fields)==set(b['fields'])
        for name,wanted in fields.items():
            if wanted.dtype in (torch.bool,torch.int64):assert torch.equal(wanted,b['fields'][name]),name
            else:torch.testing.assert_close(wanted,b['fields'][name],atol=1e-9,rtol=1e-9)
        summary=ref.reduce(fields,endpoint,case['seed'],case['anchor'])
        for r in summary:r['final_accuracy']=r['final_correct_count']/r['count'] if r['count'] else None
        ref.proof.ref.compare(summary,b['summary']);ref.proof.ref.compare(summary,cell['summary'])
        metrics=ref.proof.reduced(z[-1],y,torch.zeros_like(y,dtype=torch.bool));ref.proof.ref.compare(metrics,b['metrics']);ref.proof.ref.compare(metrics,cell['metrics'])
        assert cell['beta']==b['beta']==float(model.write_logit.sigmoid())
        checked.append(dict(case=case,summary=summary));queryrows+=y.numel();outputs+=len(z)*y.numel();writes+=(len(z)-1)*len(y)
    independent=[]
    for seed,anchor in itertools.product(seeds,(3000,12000)):
        cells={ (x['case']['keys'],x['case']['beta']):x['summary'] for x in checked if x['case']['seed']==seed and x['case']['anchor']==anchor}
        for index,row in enumerate(cells[3000,3000]):
            r=dict(seed=seed,anchor=anchor,group=row['group'],position=row['position'],count=row['count'])
            for field,name in (('target_coefficient_ratio_mean','ratio'),('final_accuracy','accuracy'),('final_ce_mean','ce')):
                a=cells[3000,3000][index][field];b=cells[3000,12000][index][field];c=cells[12000,3000][index][field];d=cells[12000,12000][index][field]
                terms={'beta_early_keys':None,'beta_late_keys':None,'keys_early_beta':None,'keys_late_beta':None,'interaction':None} if a is None else dict(
                    beta_early_keys=b-a,beta_late_keys=d-c,keys_early_beta=c-a,keys_late_beta=d-b,interaction=(d-b)-(c-a))
                r.update({name+'_'+k:v for k,v in terms.items()})
            independent.append(r)
    ref.proof.ref.compare(independent,report['contrasts'])
    assert (report['query_rows'],report['answer_outputs'],report['replay_writes'])==(queryrows,outputs,writes)
    prior.check_files(lock['files']);result=dict(passed=True,report_sha256=prior.digest(out/'report.json'),cells=len(bundles),
        summary_rows=sum(len(b['summary']) for b in checked),contrast_rows=len(independent),query_rows=queryrows,answer_outputs=outputs,
        replay_writes=writes,max_equation_error=maximum,protected_files=len(lock['files']),training_updates=0)
    if persist:assert not (out/'audit.json').exists();prior.write(out/'audit.json',result)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    torch.set_num_threads(1);audit(study.OUT)

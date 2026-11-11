"""Independent affine forward and analytic reverse derivatives, without autograd."""
import csv
import json
import torch
import write_credit_assignment as study
import audit_shared_state_feasibility as ref


def analytic(model,rows):
    ids=[];labels=[];repeats=[]
    for row in rows:
        mapping={}
        for key,value in row['events']:
            assert key not in mapping or mapping[key]==value
            mapping[key]=value
        assert mapping[row['query']]==row['label']
        order=sorted(mapping);ids.append(order);labels.append([mapping[k] for k in order])
        repeats.append([sum(key==k for key,value in row['events'])>1 for k in order])
    qids=torch.tensor(ids);y=torch.tensor(labels);repeat=torch.tensor(repeats);e=torch.tensor([r['events'] for r in rows])
    key=model.key.weight.detach();key=key/key.square().sum(-1).sqrt().clamp_min(1e-12)[:,None]
    k=key[e[...,0]];q=key[qids];v=model.value.weight.detach()[e[...,1]]
    beta=float(model.write_logit.detach().sigmoid());w=model.output.weight.detach();b=model.output.bias.detach()
    eye=torch.eye(16,dtype=torch.float64);transports=eye-beta*k[...,None]*k[...,None,:]
    s=torch.zeros(len(rows),16,8,dtype=torch.float64)
    for t in range(e.shape[1]):s=torch.bmm(transports[:,t],s)+beta*k[:,t,:,None]*v[:,t,None,:]
    z=(q@s)@w.T+b;good=z.argmax(-1)==y
    masks=dict(correct=good,error=~good,total=torch.ones_like(good));prob=z.softmax(-1)
    dz=prob.clone();dz.scatter_add_(-1,y[...,None],-torch.ones_like(y[...,None],dtype=torch.float64));dz/=4*y.numel()
    gradients={}
    for group,mask in masks.items():
        g=q.transpose(1,2)@((dz*mask[...,None])@w);reverse=[g];event=[]
        for t in reversed(range(e.shape[1])):
            event.append(beta*(k[:,t,None,:]@g).squeeze(1))
            g=transports[:,t].transpose(1,2)@g;reverse.append(g)
        vg=torch.stack(list(reversed(event)),dim=1);sg=torch.stack(list(reversed(reverse)))
        positions=torch.zeros(e.shape[1],6,8,dtype=torch.float64)
        for t in range(e.shape[1]):
            for value in range(6):positions[t,value]=vg[:,t][e[:,t,1]==value].sum(0)
        gradients[group]=dict(states=sg,events=vg,positions=positions,embedding=positions.sum(0))
    return dict(queries=qids,labels=y,repeat=repeat,logits=z,masks=masks,gradients=gradients)


def reductions(bundle):
    rows=[];summary=[]
    for name in ('correct','error','total'):
        g=bundle['gradients'][name];sg=g['states'];vg=g['events'];pg=g['positions'];count=int(bundle['masks'][name].sum())
        norm=lambda x:float(torch.sqrt(x.square().sum()))
        fraction=lambda x,y:x/y if y else None
        end=norm(sg[-1]);position_norms=[norm(x) for x in pg]
        for t in range(len(sg)):
            rows.append(dict(seed=bundle['seed'],group=name,step=t,queries=count,state_norm=norm(sg[t]),
                state_fraction_of_final=fraction(norm(sg[t]),end),event_norm=norm(vg[:,t-1]) if t else None,
                position_norm=position_norms[t-1] if t else None,
                history_sum_ratio=fraction(position_norms[t-1],sum(norm(x) for x in vg[:,t-1])) if t else None))
        joint=norm(g['embedding']);denominator=sum(position_norms)
        summary.append(dict(seed=bundle['seed'],group=name,queries=count,state_initial_fraction=fraction(norm(sg[0]),end),
            state_first_fraction=fraction(norm(sg[1]),end),final_state_norm=end,embedding_norm=joint,
            position_norm_sum=denominator,temporal_sum_ratio=fraction(joint,denominator),
            first_event_norm=norm(vg[:,0]),last_event_norm=norm(vg[:,-1])))
    return rows,summary


@torch.no_grad()
def audit(out,persist=True):
    report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text())
    for field,name in (('lock_sha256','lock.json'),('gradients_sha256','gradients.pt'),('metrics_sha256','metrics.csv')):
        assert report[field]==study.prior.digest(out/name)
    study.prior.check_files(lock['files']);bundles=torch.load(out/'gradients.pt',weights_only=True)
    assert len(bundles)==len(lock['cases']);all_rows=[];summaries=[];metrics=[];maximum=0.;answers=0;writes=0;events=0
    for case,bundle in zip(lock['cases'],bundles):
        rows=json.loads((study.ROOT/case['data']).read_text(encoding='utf-8-sig'))['pools'][case['pool']]
        model=study.prior.historical.parent.previous.load_model(study.ROOT/case['checkpoint']).eval()
        expected=analytic(model,rows);expected['seed']=case['seed'];assert bundle['seed']==case['seed']
        for field in ('queries','labels','repeat','logits'):
            a,b=bundle[field],expected[field];assert a.shape==b.shape
            if a.is_floating_point():
                maximum=max(maximum,float((a-b).abs().max()));torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-9)
            else:assert torch.equal(a,b),field
        assert bundle['masks'].keys()==expected['masks'].keys() and bundle['gradients'].keys()==expected['gradients'].keys()
        for group in ('correct','error','total'):
            assert torch.equal(bundle['masks'][group],expected['masks'][group])
            assert bundle['gradients'][group].keys()==expected['gradients'][group].keys()
            for field,b in expected['gradients'][group].items():
                a=bundle['gradients'][group][field];assert a.shape==b.shape
                maximum=max(maximum,float((a-b).abs().max()));torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-9)
            norms=expected['gradients'][group]['states'].flatten(1).norm(dim=1)
            assert bool((norms[:-1]<=norms[1:]+1e-9).all()),'Nonexpansive backward transport'
        for field in ('states','events','positions','embedding'):
            torch.testing.assert_close(bundle['gradients']['correct'][field]+bundle['gradients']['error'][field],
                bundle['gradients']['total'][field],atol=1e-9,rtol=1e-9)
        q,y,rep=expected['queries'],expected['labels'],expected['repeat']
        native_metrics=ref.existing.metrics(expected['logits'],y,rep)
        reduced={k:v for k,v in native_metrics['groups'].items()};reduced['complete_maps']=native_metrics['complete_maps']['correct']
        ref.compare(bundle['metrics'],reduced);metrics.append(dict(seed=case['seed'],metrics=reduced))
        records,summary=reductions(expected);all_rows.extend(records);summaries.extend(summary)
        answers+=y.numel();writes+=len(rows)*(len(expected['gradients']['total']['states'])-1);events+=3*(len(expected['gradients']['total']['states'])-1)
    ref.compare(report['summary'],summaries);ref.compare(report['metrics'],metrics)
    assert report['training_updates']==report['optimizer_calls']==0
    assert (report['unique_answers'],report['diagnostic_writes'],report['native_fidelity_writes'],report['state_rows'],report['event_positions'])==(answers,writes,2*writes,len(all_rows),events)
    if not lock['fixture']:assert (len(bundles),answers,writes,len(all_rows),events)==(2,2*768*32,2*768*32,2*3*33,2*3*32)
    with (out/'metrics.csv').open(newline='',encoding='utf-8') as f:csv_rows=list(csv.DictReader(f))
    assert len(csv_rows)==len(all_rows)
    for actual,expected in zip(csv_rows,all_rows):
        assert actual.keys()==expected.keys()
        for key,value in expected.items():
            if value is None:assert actual[key]==''
            elif isinstance(value,float):ref.close(float(actual[key]),value)
            else:assert actual[key]==str(value)
    study.prior.check_files(lock['files'])
    result=dict(passed=True,report_sha256=study.prior.digest(out/'report.json'),state_rows=len(all_rows),
        event_positions=events,unique_answers=answers,max_equation_error=maximum,protected_files=len(lock['files']))
    if persist:
        assert not (out/'audit.json').exists();study.prior.write(out/'audit.json',result)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':torch.set_num_threads(1);audit(study.OUT)

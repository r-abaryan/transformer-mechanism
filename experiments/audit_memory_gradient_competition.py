"""Independent matrix gradients, weighted partitions and scalar dot products."""
import json
import math

import torch
import memory_gradient_competition as study
import audit_query_feedback_training as reference
import audit_feedback_loss_decomposition as prior


def partition_metrics(z,y):
    values=[prior.scalar(score,label) for score,label in zip(z.reshape(-1,6).tolist(),y.flatten().tolist())]
    return {name:dict(count=sum(v['correct']==ok for v in values),
        ce=math.fsum(v['ce'] for v in values if v['correct']==ok)/sum(v['correct']==ok for v in values)
            if any(v['correct']==ok for v in values) else None,
        weighted_loss=math.fsum(v['ce'] for v in values if v['correct']==ok)/(4*len(values)))
        for name,ok in (('correct',True),('error',False))}


def audit_cell(path,rows,saved,record,batch=128):
    events,q,y,repeat=reference.independent.inputs(rows,'all_queries')
    assert torch.equal(y,saved['labels']) and torch.equal(repeat,saved['repeat']), 'Labels/order disagree'
    model=study.training.previous.load_model(path)
    state={n:p.clone() for n,p in model.state_dict().items()};rng=torch.get_rng_state().clone()
    for name,p in model.named_parameters():p.requires_grad_(name in study.NAMES)
    params=tuple(dict(model.named_parameters())[n] for n in study.NAMES)
    gradients={part:{n:torch.zeros_like(p) for n,p in zip(study.NAMES,params)} for part in ('correct','error','total')};scores=[]
    for start in range(0,len(rows),batch):
        z,_=reference.independent.forward_tangent(model,events[start:start+batch],q[start:start+batch]);ys=y[start:start+batch]
        losses=-(z-z.logsumexp(-1,keepdim=True)).gather(-1,ys[...,None]).squeeze(-1)
        mask=z.detach().argmax(-1)==ys
        for j,(part,selection) in enumerate((('correct',mask),('error',~mask),('total',torch.ones_like(mask)))):
            objective=losses[selection].sum()/(4*y.numel())
            for n,g in zip(study.NAMES,torch.autograd.grad(objective,params,retain_graph=j<2)):gradients[part][n]+=g.detach()
        scores.append(z.detach())
    z=torch.cat(scores);logit_error=reference.independent.close(saved['logits'],z)
    assert torch.equal(saved['logits'].argmax(-1),z.argmax(-1)), 'Winners disagree'
    grad_error=0.
    for part in gradients:
        for name,g in gradients[part].items():grad_error=max(grad_error,reference.independent.close(saved['gradients'][part][name],g,gradient=True))
    expected=partition_metrics(z,y);prior.compare(saved['partitions'],expected);prior.compare(record['partitions'],expected)
    metrics=reference.metrics(z,y,repeat);reference.check_metrics(saved['metrics'],metrics);reference.check_metrics(record['metrics'],metrics)
    assert all(torch.equal(v,model.state_dict()[n]) for n,v in state.items()) and torch.equal(rng,torch.get_rng_state())
    return gradients,dict(max_logit_error=logit_error,max_gradient_error=grad_error)


def combine(cells,gradients):
    assert len(cells)==len(gradients)==4 and [c['load'] for c in cells]==[2,8,16,32]
    n=cells[3]['labels'].numel();errors=cells[3]['partitions']['error']['count'];result={}
    for part in ('correct','error32','other_errors','total','conditional_error32'):
        result[part]={}
        for name in study.NAMES:
            if part=='correct':value=sum(g['correct'][name] for g in gradients)
            elif part=='error32':value=gradients[3]['error'][name]
            elif part=='other_errors':value=sum(g['error'][name] for g in gradients[:3])
            elif part=='total':value=sum(g['total'][name] for g in gradients)
            else:value=gradients[3]['error'][name]*(4*n/errors) if errors else torch.zeros_like(gradients[3]['error'][name])
            result[part][name]=value
    return result


def scalar_geometry(gradients):
    result={}
    for block,names in study.BLOCKS.items():
        vectors={part:[v for n in names for v in g[n].reshape(-1).tolist()] for part,g in gradients.items()}
        def dot(a,b):return math.fsum(x*y for x,y in zip(vectors[a],vectors[b]))
        norms={p:math.sqrt(dot(p,p)) for p in vectors}
        def alignment(part):
            cosine=dot(part,'error32')/(norms[part]*norms['error32']) if min(norms[part],norms['error32'])>1e-10 else None
            sign='unresolved' if cosine is None or abs(cosine)<=1e-6 else ('opposed' if cosine<0 else 'aligned')
            return cosine,sign
        cos,sign=alignment('correct');tcos,tsign=alignment('total')
        result[block]=dict(dimensions=len(vectors['correct']),correct_norm=norms['correct'],error32_norm=norms['error32'],
            other_error_norm=norms['other_errors'],total_norm=norms['total'],correct_error_cosine=cos,
            correct_alignment=sign,total_error_cosine=tcos,total_alignment=tsign,
            error_ce_direction_correct=-dot('conditional_error32','correct'),error_ce_direction_total=-dot('conditional_error32','total'),
            error_ce_direction_own=-dot('conditional_error32','error32'),error_ce_direction_other=-dot('conditional_error32','other_errors'))
    return result


def audit_probes(records,gradients):
    assert [(p['block'],p['part']) for p in records]==[(b,p) for b in study.BLOCKS for p in ('correct','total')], 'Missing probe'
    maximum=0.;active=0
    for record in records:
        names=study.BLOCKS[record['block']];part=record['part']
        norm=math.sqrt(math.fsum(x*x for n in names for x in gradients[part][n].reshape(-1).tolist()))
        assert abs(record['norm']-norm)<=1e-9
        assert record['skipped']==(norm<=1e-10)
        if record['skipped']:continue
        active+=1
        expected=-math.fsum(x*y for n in names for x,y in zip(gradients['conditional_error32'][n].reshape(-1).tolist(),gradients[part][n].reshape(-1).tolist()))/norm
        assert abs(expected-record['analytic'])<=1e-9+1e-7*abs(expected), 'Probe sign/gradient disagrees'
        assert [r['h'] for r in record['checks']]==[1e-5,5e-6]
        for check in record['checks']:
            slope=(check['plus_loss']-check['minus_loss'])/(2*check['h'])
            assert slope==check['derivative'], 'Wrong central difference'
            delta=abs(slope-expected);assert delta<=1e-7+1e-4*abs(expected), 'Finite difference disagrees'
            maximum=max(maximum,delta)
    return maximum,active


def main():
    torch.set_num_threads(1);out=study.OUT;replay=study.training.replay
    report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text());replay.check_files(lock['files'])
    assert report['lock_sha256']==replay.digest(out/'lock.json') and report['gradients_sha256']==replay.digest(out/'gradients.pt')
    assert report['training_updates']==report['optimizer_calls']==0 and report['status']=='complete'
    assert report['split']=='train' and report['device']=='cpu' and report['dtype']=='float64'
    saved=torch.load(out/'gradients.pt',weights_only=True);pairs=study.inputs()
    ids=[(s,t,f'train_{m}') for s in (3461,3463) for t in (3000,12000) for m in (2,8,16,32)]
    for data in (saved,report['cells']):assert [(c['seed'],c['step'],c['pool']) for c in data]==ids
    verified=[];errors=[]
    for (source,rows),data,record in zip(pairs,saved,report['cells']):
        for name in ('seed','step','pool','checkpoint'):assert data[name]==record[name]==source[name]
        assert len(rows)==768 and data['load']==record['load']==int(source['pool'].split('_')[1])
        assert torch.equal(source['labels'],data['labels']) and torch.equal(source['repeat'],data['repeat'])
        reference.independent.close(source['logits'],data['logits'])
        gradients,error=audit_cell(study.ROOT/source['checkpoint'],rows,data,record);verified.append(gradients);errors.append(error)
    assert [(o['seed'],o['step']) for o in report['objectives']]==[(s,t) for s in (3461,3463) for t in (3000,12000)]
    scalar_error=0.;probe_error=0.;probe_answers=0
    for i,obj in enumerate(report['objectives']):
        cells=saved[i*4:i*4+4];gradients=combine(cells,verified[i*4:i*4+4])
        scalar_error=max(scalar_error,prior.compare(obj['blocks'],scalar_geometry(gradients)))
        assert obj['error32_count']==cells[3]['partitions']['error']['count']
        assert obj['other_error_count']==sum(c['partitions']['error']['count'] for c in cells[:3])
        delta,active=audit_probes(obj['probes'],gradients);probe_error=max(probe_error,delta)
        probe_answers+=active*2*2*cells[3]['labels'].numel()
    assert len(errors)==report['cells_count']==16 and sum(c['labels'].numel() for c in saved)==report['primary_answers']==178176
    assert probe_answers==report['probe_answers']<=3145728
    replay.check_files(lock['files']);result=dict(passed=True,report_sha256=replay.digest(out/'report.json'),cells=16,
        primary_answers=178176,probe_answers=probe_answers,max_gradient_error=max(e['max_gradient_error'] for e in errors),
        max_logit_error=max(e['max_logit_error'] for e in errors),max_scalar_error=scalar_error,
        max_finite_difference_error=probe_error,protected_files=len(lock['files']),training_updates=0)
    path=out/'audit.json';assert not path.exists();replay.write(path,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()

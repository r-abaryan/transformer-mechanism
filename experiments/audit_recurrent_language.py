"""Independent recurrent equations, shifted labels and every retained token loss."""
import copy
import hashlib
import json
import math

import torch
from torch.nn import functional as F
from audit_mqa_training import independent as attention
from recurrent_language import ROOT, SEEDS, ARMS, pair, forward, shared_hash, write, summarize, decision
from wikitext2_training import configure, load_stream, tensor_hash
from position_distractors import digest


def independent(model, ids, reset=False):
    if not hasattr(model, 'gru'):return attention(model,ids)
    x=model.embedding.weight[ids];g=model.gru
    for layer in range(g.num_layers):
        h=x.new_zeros(len(ids),g.hidden_size);outputs=[]
        wi,wh,bi,bh=[getattr(g,f'{name}_l{layer}') for name in ('weight_ih','weight_hh','bias_ih','bias_hh')]
        for t in range(ids.shape[1]):
            if reset:h=torch.zeros_like(h)
            ir,iz,inc=(x[:,t]@wi.T+bi).chunk(3,-1)
            hr,hz,hn=(h@wh.T+bh).chunk(3,-1)
            r,z=(ir+hr).sigmoid(),(iz+hz).sigmoid()
            candidate=(inc+r*hn).tanh()
            h=z*h+(1-z)*candidate;outputs.append(h)
        x=torch.stack(outputs,1)
    return x@model.output.weight.T


def relative(a,b):return (a-b).norm().item()/max(b.norm().item(),1e-12)


def preflight(train,vocabulary):
    x=torch.stack((train[:64],train[64:128]));y=torch.stack((train[1:65],train[65:129]))
    for row,start in enumerate((0,64)):
        assert all(int(y[row,t])==int(train[start+t+1]) for t in range(64))
    wrong=int((x!=y).sum());assert wrong>0
    records=[]
    for seed in SEEDS:
        models=pair(seed,vocabulary)
        assert shared_hash(models['gru'])==shared_hash(models['transformer'])
        for arm,base in models.items():
            cpu=copy.deepcopy(base).double();gpu=copy.deepcopy(base).cuda()
            tiny=x[:,:8];labels=y[:,:8]
            native=forward(cpu,tiny);other=independent(cpu,tiny)
            error=(native-other).abs().max().item()
            ga=torch.autograd.grad(F.cross_entropy(native.flatten(0,1),labels.flatten()),tuple(cpu.parameters()))
            gb=torch.autograd.grad(F.cross_entropy(other.flatten(0,1),labels.flatten()),tuple(cpu.parameters()))
            ge=max((a-b).abs().max().item() for a,b in zip(ga,gb))
            assert error<=1e-10 and ge<=1e-10
            reference=independent(cpu,x)
            gradients=torch.autograd.grad(F.cross_entropy(reference.flatten(0,1),y.flatten()),tuple(cpu.parameters()))
            actual=forward(gpu,x.cuda());F.cross_entropy(actual.flatten(0,1),y.flatten().cuda()).backward()
            output=actual.detach().cpu().double();ref=reference.detach()
            absolute=(output-ref).abs().max().item();rel=relative(output,ref)
            g=torch.cat([p.grad.detach().cpu().double().flatten() for p in gpu.parameters()])
            rg=torch.cat([a.flatten() for a in gradients]);gabs=(g-rg).abs().max().item();grel=relative(g,rg)
            assert absolute<=1e-4 and rel<=1e-5 and gabs<=1e-5 and grel<=1e-4
            with torch.no_grad():
                changed=x.clone();changed[:,-1]=(changed[:,-1]+1)%vocabulary
                revised=forward(gpu,changed.cuda())
                prefix=(actual[:,:-1]-revised[:,:-1]).abs().max().item()
                effect=(actual[:,-1]-revised[:,-1]).abs().max().item()
                batch=(forward(gpu,x[:1].cuda())-actual[:1]).abs().max().item()
                again=(forward(gpu,x.cuda())-actual).abs().max().item()
                assert max(prefix,batch,again)<=1e-6 and effect>1e-6
                negative=(independent(cpu,tiny,True)-native).abs().max().item() if arm=='gru' else None
                if arm=='gru':assert negative>1e-6
            records.append(dict(seed=seed,arm=arm,cpu_output_error=error,cpu_gradient_error=ge,gpu_output_error=absolute,
                                gpu_output_relative=rel,gpu_gradient_error=gabs,gpu_gradient_relative=grel,
                                causal_prefix_error=prefix,batch_isolation_error=batch,zero_state_repeat_error=again,reset_error=negative))
            del cpu,gpu,reference,actual,ga,gb,gradients
    return dict(passed=True,shifted_targets_checked=y.numel(),unshifted_mismatches=wrong,models=records)


def audit_batches(stream,context,subset):
    total=len(stream)-1
    starts=[(j*((total//64)-1)//1023)*64 for j in range(1024)] if subset else list(range(0,total,context))
    if subset:assert len(set(starts))==1024
    for i in range(0,len(starts),8):
        chunk=starts[i:i+8]
        full=[s for s in chunk if s+context<=total]
        if full:
            x=torch.stack([stream[s:s+context] for s in full]);y=torch.stack([stream[s+1:s+context+1] for s in full])
            yield x,y,torch.arange(context).expand(len(full),context),full[0]
        for start in chunk[len(full):]:
            length=total-start
            yield stream[start:start+length][None],stream[start+1:start+length+1][None],torch.arange(length)[None],start


def checked_metrics(detail):
    losses,predictions,targets,positions=[detail[k] for k in ('losses','predictions','targets','positions')]
    result=summarize(losses,predictions,targets,positions)
    # Check the reduction through separate sums/counts, including the groups.
    masks=dict(early=positions<8,middle=(positions>=8)&(positions<32),late=positions>=32,
               unknown=targets==0,row_end=targets==1,ordinary=targets>1)
    for name,mask in [('all',torch.ones_like(targets,dtype=torch.bool)),*masks.items()]:
        count=int(mask.sum());mean=float(losses[mask].double().sum())/count
        value=result if name=='all' else result['groups'][name]
        assert abs(mean-value['loss'])<1e-12
        assert value['targets']==count and abs(value['accuracy']-int((predictions[mask]==targets[mask]).sum())/count)<1e-12
    return result


def main():
    configure()
    rp=ROOT/'results/recurrent_language.json';r=json.loads(rp.read_text());assert r['phase']=='complete'
    for name,h in r['provenance'].items():assert digest(ROOT/name)==h,name
    pre=json.loads((ROOT/'results/recurrent_language_pretest.json').read_text())
    assert pre['phase']=='all_attempts_finished_before_test' and not any('test' in k for row in pre['runs'] for k in row['metrics'])
    vocabulary=json.loads((ROOT/'results/wikitext2_encoding.json').read_text())['vocabulary']
    streams={s:load_stream(s) for s in ('train','validation','test')}
    specs=dict(train_subset=('train',64,True),validation=('validation',64,False),test64=('test',64,False),test128=('test',128,False))
    counts=[0]*vocabulary
    for token in streams['train'].tolist():counts[token]+=1
    probabilities=torch.tensor([(v+1)/(len(streams['train'])+vocabulary) for v in counts],dtype=torch.float64)
    for key,(split,context,subset) in specs.items():
        ys=torch.cat([y.flatten() for _,y,_,_ in audit_batches(streams[split],context,subset)])
        baseline=-probabilities[ys].log().mean().item()
        assert abs(baseline-r['baselines'][key]['loss'])<=1e-10
    assert {(x['seed'],x['arm']) for x in r['runs']}=={(s,a) for s in SEEDS for a in ARMS}
    out=ROOT/'results/recurrent_language_audit.json';assert not out.exists()
    progress=ROOT/'results/recurrent_language_audit_progress.json'
    cells=[]
    if progress.exists():
        saved=json.loads(progress.read_text());assert saved['report_sha256']==digest(rp) and saved['source_sha256']==digest(ROOT/'experiments/audit_recurrent_language.py')
        cells=saved['cells']
    for seed in SEEDS:
        rows=[x for x in r['runs'] if x['seed']==seed]
        assert len({x['shared_initial_sha256'] for x in rows})==len({x['stream_sha256'] for x in rows})==1
        for run in rows:
            model=pair(seed,vocabulary)[run['arm']]
            assert tensor_hash(model)==run['initial_sha256'] and shared_hash(model)==run['shared_initial_sha256']
            assert sum(p.numel() for p in model.parameters())==run['parameters']
            cp=ROOT/run['checkpoint'];assert digest(cp)==run['checkpoint_sha256']
            prior=next(x for x in pre['runs'] if (x['seed'],x['arm'])==(seed,run['arm']))
            assert prior['checkpoint_sha256']==run['checkpoint_sha256']
            g=torch.Generator().manual_seed(seed+1000);h=hashlib.sha256()
            for _ in range(run['step']):h.update(json.dumps(torch.randint(len(streams['train'])-64,(8,),generator=g).tolist()).encode())
            assert h.hexdigest()==run['stream_sha256']
            state=torch.load(cp,map_location='cpu',weights_only=True)
            assert torch.equal(g.get_state(),state['generator'])
            if run['status']!='trained':continue
            assert state['step']==run['step']==6000 and run['targets']==3072000
            assert all(v['step'].item()==6000 for v in state['optimizer']['state'].values())
            model.load_state_dict(state['model']);model.double().eval()
            assert all(bool(p.isfinite().all()) for p in model.parameters())
            assert digest(ROOT/run['scores'])==run['scores_sha256']
            details=torch.load(ROOT/run['scores'],map_location='cpu',weights_only=True)
            for key,(split,context,subset) in specs.items():
                if any((c['seed'],c['arm'],c['pool'])==(seed,run['arm'],key) for c in cells):continue
                detail=details[key];assert checked_metrics(detail)==run['metrics'][key]
                if key in prior['metrics']:assert prior['metrics'][key]==run['metrics'][key]
                offset=0;losses=[];preds=[];max_loss_error=0.;fixture_error=0.
                with torch.no_grad():
                    for index,(x,y,pos,start) in enumerate(audit_batches(streams[split],context,subset)):
                        scores=independent(model,x)
                        nll=(scores.logsumexp(-1)-scores.gather(-1,y[...,None]).squeeze(-1)).flatten()
                        n=len(nll);sl=slice(offset,offset+n)
                        assert torch.equal(detail['targets'][sl],y.flatten()) and torch.equal(detail['positions'][sl],pos.flatten())
                        max_loss_error=max(max_loss_error,(nll-detail['losses'][sl].double()).abs().max().item())
                        if index==0 or x.shape[1]<context:
                            fixture=next(f for f in detail['fixtures'] if f['start']==start)
                            fixture_error=max(fixture_error,(scores[:2]-fixture['logits'].double()).abs().max().item())
                        losses.append(nll);preds.append(scores.argmax(-1).flatten());offset+=n
                assert offset==(65536 if subset else len(streams[split])-1)
                losses,preds=torch.cat(losses),torch.cat(preds)
                error=abs(losses.mean().item()-run['metrics'][key]['loss'])
                ratio_error=abs(math.exp(losses.mean().item())/run['metrics'][key]['perplexity']-1)
                assert max_loss_error<=1e-4 and fixture_error<=1e-4 and error<=1e-5 and ratio_error<=2e-5
                audited=checked_metrics(dict(losses=losses,predictions=preds,targets=detail['targets'],positions=detail['positions']))
                group_error=max(abs(audited['groups'][group]['loss']-run['metrics'][key]['groups'][group]['loss']) for group in audited['groups'])
                assert group_error<=1e-5
                cell=dict(seed=seed,arm=run['arm'],pool=key,targets=offset,max_token_loss_error=max_loss_error,
                          mean_loss_error=error,max_group_loss_error=group_error,fixture_logit_error=fixture_error,
                          argmax_differences=int((preds!=detail['predictions']).sum()))
                cells.append(cell)
                write(progress,dict(report_sha256=digest(rp),source_sha256=digest(ROOT/'experiments/audit_recurrent_language.py'),cells=cells))
                print(json.dumps(cell),flush=True)
    assert decision(r['runs'],r['baselines'])==r['decision']
    result=dict(passed=True,report_sha256=digest(rp),source_sha256=digest(ROOT/'experiments/audit_recurrent_language.py'),cells=cells,
                evaluations=len(cells),targets_checked=sum(c['targets'] for c in cells),
                max_token_loss_error=max(c['max_token_loss_error'] for c in cells),
                argmax_differences=sum(c['argmax_differences'] for c in cells))
    write(out,result);print(json.dumps({k:v for k,v in result.items() if k!='cells'}),flush=True)


if __name__=='__main__':main()

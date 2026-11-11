"""CPU-double independent audit of every test token, checkpoints and decision."""
import hashlib
import json
import math
from pathlib import Path
import torch
from audit_mqa_training import independent
from wikitext2_training import ROOT,SEEDS,model_new,tensor_hash,load_stream,provenance
from position_distractors import digest


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rp=ROOT/'results/wikitext2_training.json';report=json.loads(rp.read_text());assert report['phase']=='complete' and report['provenance']==provenance()
    encoding=ROOT/'results/wikitext2_encoding.json';assert digest(encoding)==report['encoding_sha256']
    assert digest(ROOT/'results/wikitext2_data_audit.json')==report['data_audit_sha256']
    assert digest(ROOT/'results/wikitext2_gpu_verification.json')==report['gpu_verification_sha256']
    assert {(r['seed'],r['group']) for r in report['runs']}=={(s,g) for s in SEEDS for g in ('mha','mqa')}
    data=json.loads(encoding.read_text());vocabulary=data['vocabulary'];train,test=load_stream('train'),load_stream('test')
    counts=[0]*vocabulary
    for token in train.tolist():counts[token]+=1
    logp=[-math.log((count+1)/(len(train)+vocabulary)) for count in counts]
    baseline=sum(logp[token] for token in test[1:].tolist())/(len(test)-1)
    assert abs(baseline-report['unigram_test']['loss'])<=1e-10
    dp=ROOT/'results/wikitext2_training_scores.pt';assert digest(dp)==report['scores_sha256'];saved=torch.load(dp,map_location='cpu',weights_only=True)
    for seed in SEEDS:
        pair=[r for r in report['runs'] if r['seed']==seed]
        for key in ('common_initial_sha256','retained_kv_initial_sha256','stream_sha256'):assert pair[0][key]==pair[1][key]
    records=[];audited_losses={}
    out=ROOT/'results/wikitext2_training_audit.json';assert not out.exists()
    partial=ROOT/'results/wikitext2_training_audit_progress.json'
    if partial.exists():
        progress=json.loads(partial.read_text());assert progress['report_sha256']==digest(rp) and progress['source_sha256']==digest(Path(__file__))
        records=progress['records'];audited_losses=progress['audited_losses']
    for run in report['runs']:
        if any((x['seed'],x['group'])==(run['seed'],run['group']) for x in records):continue
        torch.manual_seed(run['seed']);base=model_new(run['group'],vocabulary).float()
        assert tensor_hash(base)==run['initial_sha256'] and tensor_hash(base,'common')==run['common_initial_sha256'] and tensor_hash(base,'kv')==run['retained_kv_initial_sha256']
        assert sum(p.numel() for p in base.parameters())==run['parameters']==32*vocabulary+(4480 if run['group']=='mha' else 3936)
        cp=ROOT/run['checkpoint'];assert digest(cp)==run['checkpoint_sha256'];state=torch.load(cp,map_location='cpu',weights_only=True);assert state['provenance']==report['provenance']
        g=torch.Generator().manual_seed(run['seed']+1000);h=hashlib.sha256()
        for _ in range(run['step']):h.update(json.dumps(torch.randint(len(train)-64,(8,),generator=g).tolist()).encode())
        assert h.hexdigest()==run['stream_sha256'] and torch.equal(g.get_state(),state['generator'])
        if run['status']!='trained':records.append(dict(seed=run['seed'],group=run['group'],status='failed_retained'));continue
        assert run['step']==6000 and run['targets']==run['tokens']==3072000
        assert len(state['optimizer']['state'])==len(list(base.parameters())) and all(s['step'].item()==6000 for s in state['optimizer']['state'].values())
        base.load_state_dict(state['model']);model=base.double().eval();detail=next(s for s in saved if (s['seed'],s['group'])==(run['seed'],run['group']))
        losses=[];predictions=[];fixture_error=0.;complete,tail=divmod(len(test)-1,64)
        with torch.no_grad():
            for first in range(0,complete,8):
                starts=torch.arange(first,min(first+8,complete))*64;positions=starts[:,None]+torch.arange(64)[None,:]
                x,y=test[positions],test[positions+1];scores=independent(model,x)
                nll=scores.logsumexp(-1)-scores.gather(-1,y[...,None]).squeeze(-1)
                losses.append(nll.flatten());predictions.append(scores.argmax(-1).flatten())
                if first==0:
                    fixture=next(f for f in detail['fixtures'] if f['start']==0);fixture_error=(scores[:2]-fixture['logits'].double()).abs().max().item()
            if tail:
                start=complete*64;x=test[start:start+tail][None,:];y=test[start+1:start+tail+1]
                scores=independent(model,x);losses.append((scores[0].logsumexp(-1)-scores[0,torch.arange(tail),y]))
                predictions.append(scores[0].argmax(-1))
                fixture=next(f for f in detail['fixtures'] if f['start']==start);fixture_error=max(fixture_error,(scores-fixture['logits'].double()).abs().max().item())
        losses=torch.cat(losses);predictions=torch.cat(predictions);assert len(losses)==len(test)-1
        error=(losses-detail['losses'].double()).abs();maximum=error.max().item();mean=losses.mean().item()
        assert maximum<=1e-4 and abs(mean-run['test']['loss'])<=1e-5 and fixture_error<=1e-4
        assert abs(math.exp(mean)/run['test']['perplexity']-1)<=2e-5
        assert abs((detail['predictions']==test[1:]).double().mean().item()-run['test']['accuracy'])<=1e-10
        record=dict(seed=run['seed'],group=run['group'],status='passed',targets=len(losses),max_token_loss_error=maximum,
                    mean_loss_error=abs(mean-run['test']['loss']),fixture_logit_error=fixture_error,
                    argmax_differences=int((predictions!=detail['predictions']).sum()))
        records.append(record);audited_losses[f'{run["group"]}_{run["seed"]}']=mean
        progress=dict(report_sha256=digest(rp),source_sha256=digest(Path(__file__)),records=records,audited_losses=audited_losses)
        partial.write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(record),flush=True)
    checks=dict(completed=all(r['status']=='trained' and r['step']==6000 for r in report['runs']))
    checks['learned_beyond_unigram']=all('test' in r and baseline-r['test']['loss']>=.05 for r in report['runs'])
    differences=[]
    for seed in SEEDS:
        pair=[next(r for r in report['runs'] if (r['seed'],r['group'])==(seed,g)) for g in ('mha','mqa')]
        if all('test' in r for r in pair):differences.append(pair[1]['test']['loss']-pair[0]['test']['loss'])
    ratios=[math.exp(d) for d in differences];geometric=math.exp(sum(differences)/3) if len(differences)==3 else None
    checks['mean_ratio5']=geometric is not None and geometric<=1.05;checks['seed_ratio10']=len(ratios)==3 and max(ratios)<=1.10
    assert report['decision']==dict(accept_mqa=all(checks.values()),checks=checks,geometric_mean_ratio=geometric,paired_ratios=ratios)
    result=dict(passed=True,records=records,targets_checked=sum(x.get('targets',0) for x in records),
                max_token_loss_error=max((x.get('max_token_loss_error',0) for x in records),default=0),
                report_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,models=len(records),targets_checked=result['targets_checked'])))


if __name__=='__main__':main()

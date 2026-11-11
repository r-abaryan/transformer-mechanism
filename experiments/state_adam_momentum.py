"""Isolate saved first-moment history in disposable Adam probes."""
import copy
import csv
import json
import math
from pathlib import Path

import torch

from state_tracking import ROOT, model_new, tensors, answer, save_json
from audit_state_tracking import oracle
from state_training_gradients import KEYS, WEIGHTS, calculate, scalar_loss, scalar_pairs, set_parameters
from state_reset_exposure import paired_reset_metrics
from position_distractors import digest

DIRECTORY = ROOT/'results/state_adam_momentum'
PROTOCOL = ROOT/'docs/protocols/state-adam-momentum.md'
MODES = ('saved_m', 'zero_prior_m')


def scalar_step(optimizer, gradient, sizes, clear):
    group = optimizer['param_groups'][0]
    beta1, beta2 = group['betas']
    displacement, first, second = [], [], []
    at = 0
    for pid, size in zip(group['params'], sizes):
        state = optimizer['state'][pid]
        old_m = state['exp_avg'].reshape(-1).tolist()
        old_v = state['exp_avg_sq'].reshape(-1).tolist()
        t = int(state['step'])+1
        gs = gradient[at:at+size].tolist()
        for g, m, v in zip(gs, old_m, old_v):
            mm = beta1*(0. if clear else m)+(1-beta1)*g
            vv = beta2*v+(1-beta2)*g*g
            displacement.append(-group['lr']*(mm/(1-beta1**t))/(math.sqrt(vv/(1-beta2**t))+group['eps']))
            first.append(mm); second.append(vv)
        at += size
    assert at==len(gradient)
    return [torch.tensor(x,dtype=torch.float64) for x in (displacement,first,second)]


def native_step(model, checkpoint, gradient, clear):
    probe = copy.deepcopy(model)
    opt = torch.optim.Adam(probe.parameters())
    opt.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    if clear:
        for value in opt.state.values(): value['exp_avg'].zero_()
    before = [p.detach().clone() for p in probe.parameters()]
    at = 0
    for p in probe.parameters():
        p.grad = gradient[at:at+p.numel()].reshape_as(p).clone()
        at += p.numel()
    assert at==len(gradient)
    opt.step()
    delta = torch.cat([(p.detach()-q).reshape(-1) for p,q in zip(probe.parameters(),before)])
    state = opt.state_dict()
    order = state['param_groups'][0]['params']
    moments = [torch.cat([state['state'][pid][key].reshape(-1) for pid in order]) for key in ('exp_avg','exp_avg_sq')]
    assert all(float(state['state'][pid]['step'])==6001 for pid in order)
    return delta, moments, dict(model=probe.state_dict(),optimizer=state)


def score_metrics(scores, baseline, pools):
    result = {}
    scalar_error = 0.
    for key, z in scores.items():
        y = tensors(pools[key])[1]
        ce = scalar_loss(z,y)
        acc = float((z.argmax(-1)==y).double().mean())
        bp, pp = baseline[key].argmax(-1), z.argmax(-1)
        result[key] = dict(count=len(y),loss=ce,accuracy=acc,
                           rescued=int(((bp!=y)&(pp==y)).sum()),harmed=int(((bp==y)&(pp!=y)).sum()))
        # A separate scalar prediction/count reduction.
        orig, after, labels = baseline[key].tolist(),z.tolist(),y.tolist()
        ac = rescued = harmed = 0
        for a,b,truth in zip(orig,after,labels):
            before = int(a[1]>a[0])==truth; good = int(b[1]>b[0])==truth
            ac+=good; rescued+=not before and good; harmed+=before and not good
        assert ac/len(y)==acc and rescued==result[key]['rescued'] and harmed==result[key]['harmed']
    pairs = paired_reset_metrics(scores,pools)
    pair_ref = scalar_pairs(scores,pools)
    for prefix in pairs:
        for key, val in pairs[prefix].items(): scalar_error=max(scalar_error,abs(val-pair_ref[prefix][key]))
    return dict(groups=result,
                cycle_loss=math.fsum(WEIGHTS[k]*v['loss'] for k,v in result.items()),
                length8_loss=math.fsum(2*WEIGHTS[k]*v['loss'] for k,v in result.items() if k.startswith('train/8/')),
                reset_pairs=pairs), scalar_error


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert PROTOCOL.exists() and not DIRECTORY.exists(), 'Preserve executed attempts.'
    parent = ROOT/'results/state_training_gradients'
    gr = json.loads((parent/'report.json').read_text(encoding='utf-8'))
    assert gr['checks_pass'] and gr['status']=='complete'
    assert digest(parent/'evidence.pt')==gr['evidence_sha256']
    assert all(digest(ROOT/p)==h for p,h in gr['source_dependencies'].items())
    gradients = torch.load(parent/'evidence.pt',weights_only=True)
    exposure = json.loads((ROOT/'results/state_reset_exposure/report.json').read_text(encoding='utf-8'))
    pools = json.loads((ROOT/'results/state_reset_exposure/data.json').read_text(encoding='utf-8'))['evaluation']
    pools = {k:pools[k] for k in KEYS}
    assert all(oracle(ids)==answer(ids[0]-4,ids[1:-1])==y for rows in pools.values() for ids,y in rows)
    runs = [r for r in exposure['runs'] if r['arm']=='independent']
    assert {r['seed'] for r in runs}=={3011,3019,3023}
    paths = [PROTOCOL,Path(__file__),parent/'report.json',parent/'evidence.pt']+[ROOT/p for p in gr['source_dependencies']]
    hashes = {p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    report = dict(status='running',device='cpu',dtype='float64',threads=1,torch=str(torch.__version__),
                  source_dependencies=hashes,weights=WEIGHTS,persistent_training_updates=0,
                  hypothetical_gradient='exact mean over length8 sampler support',runs=[])
    DIRECTORY.mkdir();save_json(DIRECTORY/'report.json',report)
    evidence = {}; errors = dict(adam=0.,moment=0.,moment_difference=0.,logit=0.,scalar=0.,baseline=0.)
    evaluations = 0; csvrows = []
    for run in runs:
        seed = run['seed'];old = gradients[seed]
        checkpoint = torch.load(ROOT/run['checkpoint'],weights_only=True)
        model = model_new().eval();model.load_state_dict(checkpoint['model'])
        params = list(model.parameters());original = [p.detach().clone() for p in params]
        sizes = [p.numel() for p in params]
        assert [n for n,p in model.named_parameters()]==old['parameter_names']
        assert [list(p.shape) for p in params]==old['parameter_shapes'] and list(KEYS)==old['keys']
        group = checkpoint['optimizer']['param_groups'][0]
        assert group['lr']==.0001 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
        assert group['weight_decay']==0 and not any(group[k] for k in ('amsgrad','maximize','decoupled_weight_decay'))
        assert all(float(v['step'])==6000 for v in checkpoint['optimizer']['state'].values())
        matrix = old['gradients']
        w8 = torch.tensor([2*WEIGHTS[k] if k.startswith('train/8/') else 0. for k in KEYS],dtype=torch.float64)
        g8 = (w8[:,None]*matrix).sum(0); cycle = old['weighted_gradient']
        baseline = {}; conditions = {}; states = {}
        for key in KEYS:
            z,_,_ = calculate(model,pools[key]);rz,_,_ = calculate(model,pools[key],reference=True)
            errors['baseline']=max(errors['baseline'],float((z-old['conditions']['baseline'][key]).abs().max()))
            errors['logit']=max(errors['logit'],float((z-rz).abs().max()))
            assert torch.equal(z.argmax(-1),rz.argmax(-1));baseline[key]=z; evaluations+=len(z)
        base_metrics,se = score_metrics(baseline,baseline,pools);errors['scalar']=max(errors['scalar'],se)
        conditions['baseline']=baseline
        h = 1e-8*max(1.,float(torch.cat([p.reshape(-1) for p in original]).norm()))
        sr = dict(seed=seed,last_lr=group['lr'],betas=list(group['betas']),epsilon=group['eps'],
                  h=h,length8_gradient_norm=float(g8.norm()),baseline=base_metrics,conditions=[])
        deltas = {}
        for mode in MODES:
            delta,moments,state = native_step(model,checkpoint,g8,mode=='zero_prior_m')
            ud,mm,vv = scalar_step(checkpoint['optimizer'],g8,sizes,mode=='zero_prior_m')
            errors['adam']=max(errors['adam'],float((delta-ud).abs().max()))
            errors['moment']=max(errors['moment'],float((moments[0]-mm).abs().max()),float((moments[1]-vv).abs().max()))
            reversed_error = float((delta+ud).abs().max());assert reversed_error>1e-9
            dn = float(delta.norm());assert dn>0 and bool(torch.isfinite(delta).all())
            direction = delta/dn;slopes = matrix@direction
            probe_metrics = {}
            try:
                for name,distance in (('step',dn),('plus',h),('minus',-h)):
                    set_parameters(params,original,direction,distance)
                    scores = {}
                    for key in KEYS:
                        z,_,m = calculate(model,pools[key]);rz,_,rm = calculate(model,pools[key],reference=True)
                        errors['logit']=max(errors['logit'],float((z-rz).abs().max()))
                        assert torch.equal(z.argmax(-1),rz.argmax(-1))
                        errors['scalar']=max(errors['scalar'],abs(m['loss']-scalar_loss(z,tensors(pools[key])[1])),abs(m['loss']-rm['loss']))
                        scores[key]=z;evaluations+=len(z)
                    met,se = score_metrics(scores,baseline,pools);errors['scalar']=max(errors['scalar'],se)
                    probe_metrics[name]=met;conditions[mode+'/'+name]=scores
            finally:
                set_parameters(params,original,direction,0.)
            assert all(torch.equal(p,q) for p,q in zip(params,original))
            fd = [];groups = []
            for j,key in enumerate(KEYS):
                numerical = (probe_metrics['plus']['groups'][key]['loss']-probe_metrics['minus']['groups'][key]['loss'])/(2*h)
                slope = float(slopes[j]);tol=1e-5+1e-3*abs(slope)
                fd.append(abs(numerical-slope)<=tol)
                row = dict(seed=seed,mode=mode,group=key,slope=slope,fd=numerical,fd_error=abs(numerical-slope),
                           fd_tolerance=tol,fd_pass=fd[-1],step_loss_change=probe_metrics['step']['groups'][key]['loss']-base_metrics['groups'][key]['loss'],
                           **probe_metrics['step']['groups'][key])
                groups.append(row);csvrows.append(row)
            sr['conditions'].append(dict(mode=mode,step_norm=dn,
                 cosine_negative_length8_gradient=float(-torch.dot(g8,direction)/g8.norm()),
                 cosine_negative_cycle_gradient=float(-torch.dot(cycle,direction)/cycle.norm()),
                 length8_slope=float(torch.dot(g8,direction)),cycle_slope=float(torch.dot(cycle,direction)),
                 group_slopes=groups,metrics=probe_metrics,finite_differences_pass=all(fd),
                 reversed_sign_error=reversed_error,parameters_restored=True))
            deltas[mode]=delta;states[mode]=state
        # The two updates have identical v_new; their difference comes only from beta1*m_old.
        old_m = torch.cat([checkpoint['optimizer']['state'][pid]['exp_avg'].reshape(-1) for pid in group['params']])
        old_v = torch.cat([checkpoint['optimizer']['state'][pid]['exp_avg_sq'].reshape(-1) for pid in group['params']])
        denominator = ((.999*old_v+.001*g8.square())/(1-.999**6001)).sqrt()+group['eps']
        contribution = -group['lr']*.9*old_m/(1-.9**6001)/denominator
        err = float((deltas['saved_m']-deltas['zero_prior_m']-contribution).abs().max())
        errors['moment_difference']=max(errors['moment_difference'],err)
        sr['moment_contribution_identity_error']=err
        evidence[seed]=dict(keys=list(KEYS),parameter_names=old['parameter_names'],gradients=matrix,g8=g8,
                            old_first_moment=old_m,old_second_moment=old_v,displacements=deltas,
                            scores=conditions,disposable_states=states)
        report['runs'].append(sr);save_json(DIRECTORY/'report.json',report)
        print(json.dumps(dict(seed=seed,conditions=[{k:c[k] for k in ('mode','cycle_slope','length8_slope','finite_differences_pass')} for c in sr['conditions']])),flush=True)
    assert errors['adam']<=1e-12 and errors['moment']<=1e-12 and errors['moment_difference']<=1e-12
    assert max(errors.values())<=1e-10
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    torch.save(evidence,DIRECTORY/'evidence.pt')
    with (DIRECTORY/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    report.update(status='complete',independent_errors=errors,prior_hashes_unchanged=True,
                  example_evaluations=evaluations,disposable_optimizer_steps=6,
                  checks_pass=all(c['finite_differences_pass'] for r in report['runs'] for c in r['conditions']),
                  evidence_sha256=digest(DIRECTORY/'evidence.pt'),metrics_sha256=digest(DIRECTORY/'metrics.csv'))
    save_json(DIRECTORY/'report.json',report)
    print(json.dumps(dict(status=report['status'],checks_pass=report['checks_pass'],errors=errors,evaluations=evaluations)))


if __name__=='__main__':
    main()

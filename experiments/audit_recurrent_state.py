"""Independent GRU gates and Transformer math; audit the bounded comparison."""
import hashlib
import json

import torch
from recurrent_state import ROOT, SEEDS, ARMS, STEPS, make_pair, provenance
from state_tracking import tensors, save_json
from state_supervision import labels, objective
from audit_state_supervision import targets_independent, loss_independent
from audit_state_tracking import audit_data, independent as transformer_independent
from mqa_training import hash_state
from position_distractors import digest


def gru_independent(model, ids, reset_each=False):
    x = model.embedding.weight[ids]
    module = model.rnn
    h = x.new_zeros(len(ids),28)
    result = []
    for pos in range(ids.shape[1]):
        if reset_each:
            h = torch.zeros_like(h)
        xi = x[:,pos]@module.weight_ih_l0.T+module.bias_ih_l0
        hh = h@module.weight_hh_l0.T+module.bias_hh_l0
        ir,iz,inc = xi.chunk(3,-1)
        hr,hz,hn = hh.chunk(3,-1)
        reset = (ir+hr).sigmoid()
        update = (iz+hz).sigmoid()
        candidate = (inc+reset*hn).tanh()
        h = update*h+(1-update)*candidate
        result.append(h@model.output.weight.T)
    return torch.stack(result,1)


def independent(model, ids, arm):
    return gru_independent(model,ids) if arm=='gru' else transformer_independent(model,ids)


def preflight(cells):
    audit_data(cells)
    checked = 0
    for rows in cells.values():
        x,_ = tensors(rows)
        target = targets_independent(x)
        assert torch.equal(labels(x),target)
        checked += target.numel()
    ids = torch.tensor([[4,2,0,1,1,3,0,1,0,6],[5,3,0,1,1,2,0,1,0,6]])
    target = targets_independent(ids)
    records = []
    resources = {}
    for seed in SEEDS:
        pair = make_pair(seed)
        assert torch.equal(pair['transformer'].embedding.weight,pair['gru'].embedding.weight)
        for arm,model in pair.items():
            count = sum(p.numel() for p in model.parameters())
            assert count==dict(transformer=4080,gru=4032)[arm]
            a,b = model(ids),independent(model,ids,arm)
            error = (a-b).abs().max().item()
            assert error<=1e-10
            la = objective(a,target,'intermediate')
            lb = loss_independent(b,target,'intermediate')
            assert abs(la.item()-lb.item())<=1e-10
            ga = torch.autograd.grad(la,list(model.parameters()))
            gb = torch.autograd.grad(lb,list(model.parameters()))
            gradient_error = max((x-y).abs().max().item() for x,y in zip(ga,gb))
            assert gradient_error<=1e-10 and all(torch.isfinite(g).all() for g in ga)
            assert max(g.abs().max().item() for g in ga)>1e-7
            changed = ids.clone(); changed[:,7]=0
            with torch.no_grad():
                c = model(changed)
                solo = torch.cat([model(row[None]) for row in ids])
            prefix = (a[:,:7]-c[:,:7]).abs().max().item()
            assert prefix<=1e-10 and (a[:,7:]-c[:,7:]).abs().max().item()>1e-6
            batch_error = (a-solo).abs().max().item()
            assert batch_error<=1e-10
            extra = {}
            if arm=='gru':
                with torch.no_grad():
                    x = model.embedding(ids)
                    first,h = model.rnn(x[:,:5])
                    last,_ = model.rnn(x[:,5:],h)
                    chunk_error = (model.output(torch.cat((first,last),1))-a).abs().max().item()
                    wrong = gru_independent(model,ids,reset_each=True)
                wrong_error = (wrong-a).abs().max().item()
                assert chunk_error<=1e-10 and wrong_error>1e-6
                extra = dict(chunk_carry_error=chunk_error,reset_every_token_control_error=wrong_error)
                projection_weights = model.rnn.weight_ih_l0.numel()+model.rnn.weight_hh_l0.numel()+model.output.weight.numel()
                assert projection_weights==3752
                resources[arm] = {str(length):projection_weights*length for length in (10,14,18,26)}
            else:
                projection_weights = model.output.weight.numel()
                for block in model.blocks:
                    projection_weights += sum(getattr(block.attn,k+'_weight').numel() for k in ('q','k','v'))
                    projection_weights += block.attn.out_proj.weight.numel()+block.linear1.weight.numel()+block.linear2.weight.numel()
                assert projection_weights==3616
                resources[arm] = {str(length):projection_weights*length+64*length*length for length in (10,14,18,26)}
            records.append(dict(seed=seed,arm=arm,parameters=count,logit_error=error,gradient_error=gradient_error,
                                causal_prefix_error=prefix,batch_independence_error=batch_error,**extra))
    return dict(passed=True,prefix_labels_checked=checked,models=records,forward_dense_macs_per_sequence=resources)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rp = ROOT/'results/recurrent_state.json'
    r = json.loads(rp.read_text())
    assert r['phase']=='complete' and r['provenance']==provenance()
    assert digest(ROOT/'results/state_supervision.json')==r['prior_report_sha256']
    assert digest(ROOT/'results/state_supervision_audit.json')==r['prior_audit_sha256']
    ip = ROOT/'results/state_tracking_data.json'
    assert digest(ip)==r['input_sha256']
    cells = json.loads(ip.read_text()); audit_data(cells)
    sp = ROOT/'results/recurrent_state_scores.pt'
    assert digest(sp)==r['scores_sha256']
    saved = torch.load(sp,weights_only=True)
    dp = ROOT/'results/recurrent_state_pretest.json'
    assert digest(dp)==r['pretest_sha256']
    frozen = json.loads(dp.read_text())
    assert frozen['provenance']==r['provenance']
    assert {(x['seed'],x['arm']) for x in r['runs']}=={(s,a) for s in SEEDS for a in ARMS}
    max_error = max_loss_error = 0.
    evaluations = final_predictions = operation_predictions = 0
    measured = {}
    for run in r['runs']:
        model = make_pair(run['seed'])[run['arm']]
        assert hash_state(model.state_dict())==run['initial_sha256']
        assert hash_state({'embedding':model.embedding.weight})==run['embedding_sha256']
        assert sum(p.numel() for p in model.parameters())==run['parameters']==dict(transformer=4080,gru=4032)[run['arm']]
        cp = ROOT/run['checkpoint']
        assert digest(cp)==run['checkpoint_sha256']
        state = torch.load(cp,weights_only=True)
        assert state['provenance']==r['provenance']
        snapshot = next(x for x in frozen['runs'] if x['seed']==run['seed'] and x['arm']==run['arm'])
        assert snapshot['checkpoint_sha256']==run['checkpoint_sha256'] and snapshot['status']==run['status'] and snapshot['step']==run['step']
        gen = torch.Generator().manual_seed(run['seed']+1000)
        stream = hashlib.sha256()
        for step in range(1,run['step']+1):
            n = 8 if step%2 else 12
            count = sum(len(v) for k,v in cells.items() if k.startswith(f'train/{n}/'))
            ix = torch.randint(count,(64,),generator=gen)
            stream.update(json.dumps(ix.tolist()).encode())
        assert stream.hexdigest()==run['stream_sha256'] and torch.equal(gen.get_state(),state['generator'])
        if run['status']!='trained': continue
        assert run['step']==STEPS and all(s['step'].item()==STEPS for s in state['optimizer']['state'].values())
        model.load_state_dict(state['model']); model.eval()
        name = f"{run['seed']}/{run['arm']}"
        measured[name] = {}
        for key,rows in cells.items():
            x,y = tensors(rows)
            target = targets_independent(x)
            assert torch.equal(target[:,-1],y)
            with torch.no_grad():
                logits = torch.cat([independent(model,b,run['arm']) for b in x.split(128)])
            error = (logits-saved[name][key]).abs().max().item()
            assert error<=1e-10 and torch.equal(logits.argmax(-1),saved[name][key].argmax(-1))
            m = run['evaluation'][key]
            if not key.startswith('test/'):
                assert snapshot['evaluation'][key]==m
            good = logits[:,-1].argmax(-1)==y
            accuracy = int(good.sum())/len(y)
            assert accuracy==m['accuracy'] and int(good.sum())==m['correct'] and len(y)==m['count']
            for c in (0,1):
                assert int(good[y==c].sum())/int((y==c).sum())==m['class_accuracy'][str(c)]
            assert int((logits[:,1:-1].argmax(-1)==target).sum())/target.numel()==m['operation_accuracy']
            loss = (logits[:,-1].logsumexp(-1)-logits[:,-1].gather(1,y[:,None]).squeeze(1)).mean().item()
            loss_error = abs(loss-m['loss'])
            assert loss_error<=1e-10
            max_error,max_loss_error = max(max_error,error),max(max_loss_error,loss_error)
            evaluations += 1; final_predictions += len(y); operation_predictions += target.numel()
            measured[name][key] = accuracy
    complete = len(r['runs'])==6 and all(x['status']=='trained' and x['step']==STEPS for x in r['runs'])
    checks = {}
    for seed in SEEDS:
        t,g = [next(x for x in r['runs'] if x['seed']==seed and x['arm']==arm) for arm in ARMS]
        assert t['embedding_sha256']==g['embedding_sha256']
        if t['step']==g['step']:
            assert t['stream_sha256']==g['stream_sha256']
        def acc(arm,key):
            return measured.get(f'{seed}/{arm}',{}).get(key,0)
        short = [f'{s}/{n}/{k}' for s in ('train','validation') for n in (8,12) for k in ('reset_free','long_tail','mixed')]
        differences = [acc('gru',f'test/{n}/{k}')-acc('transformer',f'test/{n}/{k}') for n in (16,24) for k in ('reset_free','long_tail')]
        gap = sum(differences)/4
        checks[str(seed)] = dict(gru_short_fit=complete and all(acc('gru',k)>=.99 for k in short),
            both_short_fit=complete and all(acc(arm,k)>=.99 for arm in ARMS for k in short),
            gru_test_reliable=complete and all(acc('gru',f'test/{n}/{k}')>=(.99 if n<=12 else .90) for n in (8,12,16,24) for k in ('reset_free','long_tail','mixed')),
            long_gap_pp=100*gap,long_advantage=complete and gap>=.10-1e-12)
    reliable = complete and all(v['gru_short_fit'] and v['gru_test_reliable'] for v in checks.values())
    advantage = complete and all(v['long_advantage'] for v in checks.values())
    decision = dict(passed=reliable and advantage,completed=complete,gru_reliability=reliable,relative_long_advantage=advantage,
                    fit_comparable=complete and all(v['both_short_fit'] for v in checks.values()),seeds=checks)
    assert decision==r['decision']
    result = dict(passed=True,report_sha256=digest(rp),provenance=provenance(),evaluations=evaluations,
                  final_predictions_checked=final_predictions,operation_predictions_checked=operation_predictions,
                  max_logit_error=max_error,max_loss_error=max_loss_error,argmax_differences=0,
                  decision_verified=True,initialization_and_streams_verified=True,pretest_snapshot_verified=True)
    ap = ROOT/'results/recurrent_state_audit.json'
    assert not ap.exists()
    save_json(ap,result)
    print(json.dumps({k:v for k,v in result.items() if k!='provenance'}))


if __name__=='__main__':
    main()

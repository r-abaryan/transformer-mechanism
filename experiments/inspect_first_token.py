"""Frozen-checkpoint attention inspection and paired interventions."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import torch
from torch.nn import functional as F
from verify_decoder import TinyDecoder

ROOT = Path(__file__).resolve().parents[1]


def query_trace(model, ids, intervention=None, head=0):
    x = model.embedding(ids)
    b, n, d = x.shape
    positions = torch.arange(n, dtype=x.dtype)[:, None]
    frequency = 10000 ** (-torch.arange(0, d, 2, dtype=x.dtype)/d)
    pe = torch.stack(((positions*frequency).sin(), (positions*frequency).cos()), -1).flatten(1)
    if intervention == 'query_position_six':
        pe[-1] = pe[6].clone()
    x = x + pe
    layer = model.block
    z = layer.norm1(x)
    q, k, v = F.linear(z, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias).chunk(3, -1)
    h = layer.self_attn.num_heads
    q, k, v = [a.reshape(b, n, h, d//h).transpose(1, 2) for a in (q, k, v)]
    weights = (q[:, :, -1:] @ k.transpose(-2, -1) / math.sqrt(d//h)).softmax(-1).squeeze(2)
    if intervention in ('first_attention', 'last_attention', 'uniform_attention'):
        weights[:, head] = 0
        if intervention == 'uniform_attention':
            weights[:, head] = 1/n
        else:
            weights[:, head, 0 if intervention == 'first_attention' else n-2] = 1
    contributions = weights[..., None]*v
    if intervention == 'remove_first_value':
        contributions[:, head, 0] = 0
    elif intervention == 'remove_last_value':
        contributions[:, head, n-2] = 0
    elif intervention == 'zero_head':
        contributions[:, head] = 0
    projected = torch.stack([F.linear(contributions[:, j], layer.self_attn.out_proj.weight[:, j*(d//h):(j+1)*(d//h)], None)
                             for j in range(h)], 1)
    mixed = contributions.sum(2).reshape(b, d)
    u = x[:, -1] + layer.self_attn.out_proj(mixed)
    y = u + layer.linear2(F.relu(layer.linear1(layer.norm2(u))))
    logits = model.output(model.final_norm(y))
    return logits, weights, projected.norm(dim=-1)


def metrics(logits, target, original):
    good = logits.argmax(-1)==target
    return dict(accuracy=good.double().mean().item(), loss=F.cross_entropy(logits, target).item(),
                corrected=int((good & ~original).sum()), broken=int((~good & original).sum()))


def main():
    torch.set_num_threads(1)
    training = json.loads((ROOT/'results/first_token_training.json').read_text())
    rows = list(itertools.product(range(4), repeat=6))
    random.Random(20260929).shuffle(rows)
    rng = random.Random(20260930)
    longer = set()
    while len(longer)<512:
        longer.add(tuple(rng.randrange(4) for _ in range(10)))
    splits = dict(test=rows[2560:3072], longer=sorted(longer))
    for name, values in splits.items():
        assert hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()==training['split_hashes'][name]
    report = dict(scope='Exploratory frozen-model interventions on existing test sets', runs=[],
                  torch=torch.__version__, dtype='float64', device='cpu',
                  source_sha256={name: hashlib.sha256((ROOT/'experiments'/name).read_bytes()).hexdigest()
                                 for name in ('inspect_first_token.py','verify_decoder.py')},
                  protocol_sha256=hashlib.sha256((ROOT/'docs/intervention-protocol.md').read_bytes()).hexdigest())
    with torch.no_grad():
        for run in training['runs']:
            checkpoint = ROOT/run['checkpoint']
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==run['checkpoint_sha256']
            model = TinyDecoder(5,16,2,32).double().eval()
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            for name, values in splits.items():
                ids = torch.tensor([list(r)+[4] for r in values])
                target = torch.tensor([r[0] for r in values])
                logits, weights, norms = query_trace(model, ids)
                error = (logits-model(ids)[:, -1]).abs().max().item()
                assert error<1e-10, error
                correct = logits.argmax(-1)==target
                base = metrics(logits,target,correct)
                assert base['accuracy']==run['metrics'][name]['accuracy']
                groups = {}
                for label, selected in [('correct',correct),('incorrect',~correct)]:
                    groups[label] = dict(count=int(selected.sum()))
                    if selected.any():
                        groups[label].update(first_attention=weights[selected,:,0].mean(0).tolist(),
                                             query_attention=weights[selected,:,-1].mean(0).tolist(),
                                             entropy=(-(weights[selected]*weights[selected].clamp_min(1e-300).log()).sum(-1)).mean(0).tolist(),
                                             first_value_norm=norms[selected,:,0].mean(0).tolist())
                interventions = {}
                for kind in ('zero_head','first_attention','last_attention','uniform_attention','remove_first_value','remove_last_value'):
                    for head in (0,1):
                        result,_,_ = query_trace(model,ids,kind,head)
                        interventions[f'{kind}_h{head}'] = metrics(result,target,correct)
                if name=='longer':
                    result,_,_ = query_trace(model,ids,'query_position_six')
                    interventions['query_position_six'] = metrics(result,target,correct)
                report['runs'].append(dict(seed=run['seed'],split=name,checkpoint_sha256=run['checkpoint_sha256'],
                                          reconstruction_error=error,baseline=base,groups=groups,interventions=interventions))
    (ROOT/'results/first_token_interventions.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for run in report['runs']:
        if run['split']=='longer':
            print(json.dumps(run if run['seed']==47 else dict(seed=run['seed'],baseline=run['baseline'],position=run['interventions']['query_position_six'])),flush=True)


if __name__=='__main__':
    main()

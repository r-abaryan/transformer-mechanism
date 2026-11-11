"""Separate query positional effects with hooks on the frozen library model."""
import hashlib
import json
from pathlib import Path
import random

import torch
from inspect_first_token import metrics, query_trace
from verify_decoder import TinyDecoder

ROOT = Path(__file__).resolve().parents[1]


def positional(position, width):
    frequency = 10000 ** (-torch.arange(0,width,2,dtype=torch.float64)/width)
    return torch.stack(((position*frequency).sin(),(position*frequency).cos()),-1).flatten()


def patched(model, ids, donor, path):
    delta = positional(donor,16)-positional(10,16)
    original = model.embedding(ids)[:,-1]+positional(10,16)
    replacement = model.block.norm1(original+delta)
    hooks=[]
    def inputs(module,args):
        values=list(args)
        indices = (0,) if path=='q' else (1,2) if path=='kv' else (0,1,2)
        for i in indices:
            values[i]=values[i].clone()
            values[i][:,-1]=replacement
        return tuple(values)
    def output(module,args,result):
        value,weights=result
        value=value.clone()
        value[:,-1]+=delta
        return value,weights
    if path in ('q','kv','attention','both'):
        hooks.append(model.block.self_attn.register_forward_pre_hook(inputs))
    if path in ('residual','both'):
        hooks.append(model.block.self_attn.register_forward_hook(output))
    try:
        return model(ids)[:,-1]
    finally:
        for hook in hooks:
            hook.remove()


def sample(seed,count,excluded):
    rng=random.Random(seed)
    rows=set()
    while len(rows)<count:
        row=tuple(rng.randrange(4) for _ in range(10))
        if row not in excluded:
            rows.add(row)
    return sorted(rows)


def main():
    torch.set_num_threads(1)
    # Disable fused encoder fast path so hooks always target the explicit modules.
    torch.backends.mha.set_fastpath_enabled(False)
    old=sample(20260930,512,set())
    rows=sample(20261001,1024,set(old))
    assert not set(rows)&set(old)
    ids=torch.tensor([list(r)+[4] for r in rows])
    target=torch.tensor([r[0] for r in rows])
    training=json.loads((ROOT/'results/first_token_training.json').read_text())
    report=dict(data_seed=20261001,examples=1024,excluded_prior_examples=512,
                data_sha256=hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest(),
                torch=torch.__version__,dtype='float64',device='cpu',runs=[],
                protocol_sha256=hashlib.sha256((ROOT/'docs/query-position-protocol.md').read_bytes()).hexdigest(),
                source_sha256={n:hashlib.sha256((ROOT/'experiments'/n).read_bytes()).hexdigest()
                               for n in ('query_position_paths.py','inspect_first_token.py','verify_decoder.py')})
    with torch.no_grad():
        for run in training['runs']:
            checkpoint=ROOT/run['checkpoint']
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==run['checkpoint_sha256']
            model=TinyDecoder(5,16,2,32).double().eval()
            model.load_state_dict(torch.load(checkpoint,weights_only=True))
            baseline=model(ids)[:,-1]
            correct=baseline.argmax(-1)==target
            reference=query_trace(model,ids,'query_position_six')[0]
            interventions={}
            no_change=0.0
            for donor in (0,5,6,7,10):
                for path in ('q','kv','attention','residual','both'):
                    logits=patched(model,ids,donor,path)
                    interventions[f'{path}_position_{donor}']=metrics(logits,target,correct)
                    if donor==10:
                        no_change=max(no_change,(logits-baseline).abs().max().item())
                    if donor==6 and path=='both':
                        combined_error=(logits-reference).abs().max().item()
            assert no_change<1e-10 and combined_error<1e-10
            record=dict(seed=run['seed'],checkpoint_sha256=run['checkpoint_sha256'],
                        baseline=metrics(baseline,target,correct),no_change_error=no_change,
                        combined_reference_error=combined_error,interventions=interventions)
            report['runs'].append(record)
            print(json.dumps(dict(seed=run['seed'],baseline=record['baseline'],
                                  position_six={p:interventions[f'{p}_position_6'] for p in ('q','kv','attention','residual','both')})),flush=True)
    (ROOT/'results/query_position_paths.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()

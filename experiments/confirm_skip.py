"""Fixed-protocol confirmation across new examples and lengths."""
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch
from torch.nn import functional as F
from query_position_paths import positional, sample
from inspect_first_token import metrics
from verify_decoder import TinyDecoder

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    prior=set(sample(20260930,512,set()))
    prior.update(sample(20261001,1024,prior))
    training=json.loads((ROOT/'results/first_token_training.json').read_text())
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',runs=[],datasets={},
                protocol_sha256=hashlib.sha256((ROOT/'docs/skip-confirmation-protocol.md').read_bytes()).hexdigest(),
                source_sha256={n:hashlib.sha256((ROOT/'experiments'/n).read_bytes()).hexdigest() for n in
                               ('confirm_skip.py','query_position_paths.py','inspect_first_token.py','verify_decoder.py')})
    with torch.no_grad():
        for length in (8,10,12,16):
            rng=random.Random(20261010+length)
            rows=set()
            while len(rows)<512:
                row=tuple(rng.randrange(4) for _ in range(length))
                if row not in prior:
                    rows.add(row)
            assert not rows&prior
            rows=sorted(rows)
            ids=torch.tensor([list(r)+[4] for r in rows])
            target=torch.tensor([r[0] for r in rows])
            delta=positional(6,16)-positional(length,16)
            g=torch.Generator().manual_seed(20261020+length)
            control=torch.randn(16,generator=g,dtype=torch.float64)
            control-=torch.dot(control,delta)/torch.dot(delta,delta)*delta
            control=control/control.norm()*delta.norm()
            assert abs(torch.dot(control,delta).item())<1e-10
            assert abs((control.norm()-delta.norm()).item())<1e-10
            report['datasets'][str(length)]=dict(seed=20261010+length,count=512,
                sha256=hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest(),
                delta_norm=delta.norm().item(),control_norm=control.norm().item(),
                control_dot_delta=torch.dot(control,delta).item(),control_vector=control.tolist())
            for run in training['runs']:
                path=ROOT/run['checkpoint']
                assert hashlib.sha256(path.read_bytes()).hexdigest()==run['checkpoint_sha256']
                model=TinyDecoder(5,16,2,32).double().eval()
                model.load_state_dict(torch.load(path,weights_only=True))
                captured={}
                def capture(module,args):
                    captured['u']=args[0][:,-1].clone()
                hook=model.block.norm2.register_forward_pre_hook(capture)
                try:
                    baseline=model(ids)[:,-1]
                finally:
                    hook.remove()
                u=captured['u']
                def f(x):
                    return model.block.linear2(F.relu(model.block.linear1(model.block.norm2(x))))
                original_ffn=f(u)
                blocks=dict(original=u+original_ffn,skip=u+delta+original_ffn,
                            full=u+delta+f(u+delta),control=u+control+original_ffn)
                correct=baseline.argmax(-1)==target
                results={name:metrics(model.output(model.final_norm(b)),target,correct) for name,b in blocks.items()}
                error=(model.output(model.final_norm(blocks['original']))-baseline).abs().max().item()
                assert error<1e-10
                report['runs'].append(dict(seed=run['seed'],length=length,results=results,
                                          reconstruction_error=error,checkpoint_sha256=run['checkpoint_sha256']))
            subset=[r for r in report['runs'] if r['length']==length]
            print(json.dumps(dict(length=length,accuracy={str(r['seed']):{k:v['accuracy'] for k,v in r['results'].items()} for r in subset})),flush=True)
    report['criteria']=dict(all_skip_at_least_95=all(r['results']['skip']['accuracy']>=.95 for r in report['runs']),
        mean_skip_minus_control_by_length={str(n):statistics.mean(r['results']['skip']['accuracy']-r['results']['control']['accuracy'] for r in report['runs'] if r['length']==n) for n in (8,10,12,16)})
    report['criteria']['specificity_pass']=all(v>=.05 for v in report['criteria']['mean_skip_minus_control_by_length'].values())
    (ROOT/'results/skip_confirmation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['criteria']))


if __name__=='__main__':
    main()

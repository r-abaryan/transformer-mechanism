"""Fixed depth replication and separate length-transfer evaluation."""
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F
from compare_depth import StackedDecoder, verify
from train_associative_recall import datasets, generate, tensors, evaluate

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row):
    x, _ = row
    return tuple(sorted(zip(x[:-2:2], x[1:-2:2]))), x[-1]


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    old = {n: {identity(r) for r in rows} for n, rows in datasets().items()}
    splits, seen = {}, set()
    for name, seed, count, pairs in [('train',20261101,8192,3), ('validation',20261102,1024,3),
                                    ('test',20261103,1024,3), ('four_pairs',20261104,1024,4),
                                    ('six_pairs',20261105,1024,6)]:
        excluded = seen | (old[name] if pairs > 3 else set())
        rows, identities = generate(seed, count, pairs, excluded)
        assert len(identities) == count and not identities & excluded
        assert identities == {identity(r) for r in rows}
        for x, y in rows:
            keys, values = x[:-2:2], x[1:-2:2]
            assert len(set(keys)) == pairs and x[-2] == 12
            assert values[keys.index(x[-1])] == y
        splits[name] = rows
        seen.update(identities)
    split_path = ROOT / 'results/depth_confirmation_splits.json'
    output = ROOT / 'results/depth_confirmation.json'
    assert not output.exists() and not split_path.exists(), 'Preserve existing evidence'
    split_path.write_text(json.dumps(splits, separators=(',', ':')) + '\n')
    data = {n: tensors(rows) for n, rows in splits.items()}
    report = dict(torch=torch.__version__, device='cpu', dtype='float64', runs=[], verification=verify(),
                  split_file=str(split_path.relative_to(ROOT)), split_sha256=digest(split_path),
                  protocol_sha256=digest(ROOT/'docs/depth-confirmation-protocol.md'),
                  source_sha256={n:digest(ROOT/'experiments'/n) for n in
                                 ('confirm_depth.py','compare_depth.py','train_associative_recall.py','verify_decoder.py')},
                  overlap_with_original={n:{k:len({identity(r) for r in rows}&ids) for k,ids in old.items()}
                                         for n,rows in splits.items()}, baselines={})
    for n, rows in splits.items():
        report['baselines'][n] = dict(
            frequency=sum(max(range(8,12), key=lambda v:x[1:-2:2].count(v))==y for x,y in rows)/len(rows),
            first=sum(x[1]==y for x,y in rows)/len(rows), last=sum(x[-3]==y for x,y in rows)/len(rows),
            uniform_expected=.25, oracle=1., target_counts=[sum(y==v for x,y in rows) for v in range(8,12)],
            query_position_counts=[sum(x[:-2:2].index(x[-1])==p for x,y in rows)
                                   for p in range((len(rows[0][0])-2)//2)])
    directory = ROOT/'results/depth_confirmation_checkpoints'
    directory.mkdir(exist_ok=True)
    for seed in (137,163,191,223,257):
        for name,width,hidden,depth,count in [('deep',16,32,2,4896),('wide',24,40,1,5152)]:
            torch.manual_seed(seed)
            model = StackedDecoder(width,hidden,depth).double()
            assert sum(p.numel() for p in model.parameters()) == count
            optimizer = torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
            generator = torch.Generator().manual_seed(seed+1000)
            batches = hashlib.sha256()
            curve = []
            start = time.perf_counter()
            x,y = data['train']
            for step in range(1,3001):
                idx = torch.randint(len(x),(64,),generator=generator)
                batches.update(json.dumps(idx.tolist(),separators=(',',':')).encode())
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x[idx])[:,-1],y[idx])
                assert torch.isfinite(loss), (name,seed,step)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step()
                if step % 500 == 0:
                    scores = evaluate(model,data['validation'])
                    curve.append(dict(step=step,**scores))
                    print(json.dumps(dict(model=name,seed=seed,step=step,validation=scores)),flush=True)
            checkpoint = directory/f'{name}_{seed}.pt'
            torch.save(model.state_dict(),checkpoint)
            run = dict(model=name,seed=seed,parameters=count,width=width,hidden=hidden,depth=depth,
                       steps=3000,examples_processed=192000,tokens_processed=1536000,
                       metrics={n:evaluate(model,d) for n,d in data.items()},validation_curve=curve,
                       minibatches_sha256=batches.hexdigest(),checkpoint=str(checkpoint.relative_to(ROOT)),
                       checkpoint_sha256=digest(checkpoint),seconds_including_evaluation=time.perf_counter()-start)
            if name == 'wide':
                assert run['minibatches_sha256'] == report['runs'][-1]['minibatches_sha256']
            report['runs'].append(run)
            output.write_text(json.dumps(report,indent=2)+'\n')
    deep = [r for r in report['runs'] if r['model']=='deep']
    wide = [r for r in report['runs'] if r['model']=='wide']
    gains = [a['metrics']['test']['accuracy']-b['metrics']['test']['accuracy'] for a,b in zip(deep,wide,strict=True)]
    report['criteria'] = dict(deep_all_train_test_90=all(r['metrics'][n]['accuracy']>=.9 for r in deep for n in ('train','test')),
                              mean_test_gain=statistics.mean(gains),sd_test_gain=statistics.stdev(gains),
                              mean_test_gain_at_least_10_points=statistics.mean(gains)>=.1,
                              transfer_all_90={n:all(r['metrics'][n]['accuracy']>=.9 for r in deep)
                                               for n in ('four_pairs','six_pairs')})
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['criteria']),flush=True)


if __name__ == '__main__':
    main()

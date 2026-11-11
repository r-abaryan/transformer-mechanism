"""Post-hoc query and binding sensitivity, frozen models and original test maps."""
import hashlib
import json
from pathlib import Path

import torch
from train_associative_recall import datasets
from verify_decoder import TinyDecoder

ROOT=Path(__file__).resolve().parents[1]


def predict(model,rows):
    return torch.cat([model(chunk)[:,-1].argmax(-1) for chunk in torch.tensor(rows).split(256)])


def main():
    torch.set_num_threads(1)
    splits=datasets()
    training=json.loads((ROOT/'results/associative_recall_training.json').read_text())
    for n,r in splits.items():
        assert hashlib.sha256(json.dumps(r,separators=(',',':')).encode()).hexdigest()==training['split_hashes'][n]
    rows=splits['test']
    originals=[x for x,y in rows]
    target=torch.tensor([y for x,y in rows])
    alternatives=[]
    alternative_targets=[]
    rotations=[]
    rotated_targets=[]
    for x,y in rows:
        keys=x[:-2:2]
        values=x[1:-2:2]
        for k,v in zip(keys,values):
            alternatives.append(x[:-1]+[k])
            alternative_targets.append(v)
        rotated=values[1:]+values[:1]
        rotations.append([i for pair in zip(keys,rotated) for i in pair]+x[-2:])
        rotated_targets.append(rotated[keys.index(x[-1])])
    alt_target=torch.tensor(alternative_targets)
    rotated_target=torch.tensor(rotated_targets)
    report=dict(runs=[],post_hoc=True,
                protocol_sha256=hashlib.sha256((ROOT/'docs/associative-diagnostic-protocol.md').read_bytes()).hexdigest(),
                source_sha256={n:hashlib.sha256((ROOT/'experiments'/n).read_bytes()).hexdigest() for n in
                               ('diagnose_associative_recall.py','train_associative_recall.py','verify_decoder.py')},
                frequency_baseline={})
    def identity(x):
        return tuple(sorted(zip(x[:-2:2],x[1:-2:2]))),x[-1]
    training_identities={identity(x) for x,y in splits['train']}
    report['derived_training_overlap']=dict(
        query_probes=sum(identity(x) in training_identities for x in alternatives),query_total=len(alternatives),
        rotation_probes=sum(identity(x) in training_identities for x in rotations),rotation_total=len(rotations))
    for name,rs in splits.items():
        guesses=[min(range(8,12),key=lambda v:(-x[1:-2:2].count(v),v)) for x,y in rs]
        report['frequency_baseline'][name]=sum(g==y for g,(x,y) in zip(guesses,rs))/len(rs)
    with torch.no_grad():
        for run in training['runs']:
            checkpoint=ROOT/run['checkpoint']
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==run['checkpoint_sha256']
            model=TinyDecoder(13,16,2,32).double().eval()
            model.load_state_dict(torch.load(checkpoint,weights_only=True))
            base=predict(model,originals)
            assert (base==target).double().mean().item()==run['metrics']['test']['accuracy']
            alt=predict(model,alternatives)
            rot=predict(model,rotations)
            different=alt_target!=target.repeat_interleave(3)
            rotated_different=rotated_target!=target
            result=dict(seed=run['seed'],checkpoint_sha256=run['checkpoint_sha256'],
                        all_queries_accuracy=(alt==alt_target).double().mean().item(),
                        query_change_prediction_change=(alt!=base.repeat_interleave(3)).double().mean().item(),
                        changed_answer_queries_count=int(different.sum()),
                        changed_answer_query_accuracy=(alt[different]==alt_target[different]).double().mean().item(),
                        changed_answer_prediction_change=(alt[different]!=base.repeat_interleave(3)[different]).double().mean().item(),
                        rotation_accuracy=(rot==rotated_target).double().mean().item(),
                        rotation_changed_answer_count=int(rotated_different.sum()),
                        rotation_changed_answer_prediction_change=(rot[rotated_different]!=base[rotated_different]).double().mean().item())
            report['runs'].append(result)
    (ROOT/'results/associative_recall_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

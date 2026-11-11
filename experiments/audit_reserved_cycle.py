"""Read saved reservation/checkpoints and verify same-length test metrics."""
import json
from pathlib import Path
import torch
from reserved_cycle import model_new,evaluate
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    path=ROOT/'results/reserved_cycle.json'; report=json.loads(path.read_text())
    assert report['phase']=='complete' and len(report['runs'])==5
    splitpath=ROOT/'results/reserved_cycle_splits.json'
    assert digest(splitpath)==report['split_sha256']
    assert digest(ROOT/'docs/reserved-cycle-protocol.md')==report['protocol_sha256']
    for n,h in report['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    splits=json.loads(splitpath.read_text()); identities={}
    for name,rows in splits.items():
        identities[name]=set()
        for x,y in rows:
            pairs=frozenset((x[i],x[i+1]) for i in (0,2,4))
            assert len({k for k,v in pairs})==3 and len({v for k,v in pairs})==3
            assert dict(pairs)[x[-1]]==y and x[-2]==18
            identities[name].add(pairs)
        assert len(identities[name])==report['split_counts'][name]['mappings']
        assert all(sum(y==v for x,y in rows)==len(rows)//6 for v in range(12,18))
        assert all(sum(x[:-2:2].index(x[-1])==i for x,y in rows)==len(rows)//3 for i in range(3))
    for a,b in [('train','validation'),('train','test'),('validation','test')]: assert not identities[a]&identities[b]
    data=(torch.tensor([x for x,y in splits['test']]),torch.tensor([y for x,y in splits['test']]))
    errors=[]
    for run in report['runs']:
        checkpoint=ROOT/run['checkpoint']; assert digest(checkpoint)==run['checkpoint_sha256']
        model=model_new(); model.load_state_dict(torch.load(checkpoint,weights_only=True))
        actual=evaluate(model,data)
        error=max(abs(actual[k]-run['test']['0_original'][k]) for k in actual); assert error<1e-10
        for shift in (0,1,2,4,6,8):
            pair=run['test'][f'{shift}_pair']; base=run['test'][f'{shift}_original']
            assert abs((pair['rescued']-pair['harmed'])/7200-pair['accuracy']+base['accuracy'])<1e-12
        errors.append(dict(seed=run['seed'],zero_shift_metric_error=error))
    assert all(r['test']['0_original']['accuracy']>=.9 for r in report['runs'])==report['criteria']['original_zero_all_90']
    for shift,criterion in report['criteria']['shifts'].items():
        gain=sum(r['test'][f'{shift}_pair']['accuracy']-r['test'][f'{shift}_original']['accuracy'] for r in report['runs'])/5
        assert abs(gain-criterion['mean_gain'])<1e-12
        assert all(r['test'][f'{shift}_pair']['accuracy']>=.9 for r in report['runs'])==criterion['all_90']
        assert (gain>=.05)==criterion['gain_5']
    audit=dict(report_sha256=digest(path),source_sha256=digest(Path(__file__)),mapping_disjointness=True,labels_and_balance=True,checkpoint_rechecks=errors)
    (ROOT/'results/reserved_cycle_audit.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit))


if __name__=='__main__': main()

"""Evaluate the frozen global-edge dependence criterion from audited scores."""
import json
import torch
from global_retrieval import ROOT, PREFIX, SEEDS, tensors
from position_distractors import digest
from recurrent_retrieval import write


def main():
    path=ROOT/f'results/{PREFIX}_dependence.json';assert not path.exists()
    rp=ROOT/f'results/{PREFIX}.json';r=json.loads(rp.read_text())
    ap=ROOT/f'results/{PREFIX}_audit.json';a=json.loads(ap.read_text())
    assert a['passed'] and a['report_sha256']==digest(rp)
    data=tensors(json.loads((ROOT/f'results/{PREFIX}_data.json').read_text()))
    rows=[]
    for run in r['runs']:
        if run['arm']!='global':continue
        assert run['status']=='trained'
        assert digest(ROOT/run['scores'])==run['scores_sha256']
        saved=torch.load(ROOT/run['scores'],weights_only=True)
        eligible=('1','4','12','16')
        gate=all(run['metrics']['train_17'][g]['accuracy']>=.99 and run['metrics']['validation_17'][g]['accuracy']>=.95 for g in eligible)
        for pool in ('test_17','test_21'):
            _,y,d=data[pool]
            for distance in (12,16):
                selected=d==distance;count=int(selected.sum())
                native=int((saved[pool].argmax(-1)[selected]==y[selected]).sum())
                ablated=int((saved['ablated/'+pool].argmax(-1)[selected]==y[selected]).sum())
                assert native==run['metrics'][pool][str(distance)]['correct']
                assert ablated==run['ablated'][pool][str(distance)]['correct']
                rows.append(dict(seed=run['seed'],pool=pool,distance=distance,count=count,native_correct=native,
                    ablated_correct=ablated,gain_points=100*(native-ablated)/count,
                    passes=gate and native/count>=.95 and ablated*6<=count))
    assert len(rows)==len(SEEDS)*4
    write(path,dict(passed=True,dependence_criterion=all(c['passes'] for c in rows),report_sha256=digest(rp),
        audit_sha256=digest(ap),source_sha256=digest(ROOT/'experiments/audit_global_retrieval_dependence.py'),cells=rows))
    print(json.dumps(dict(dependence_criterion=all(c['passes'] for c in rows),cells=rows)))


if __name__=='__main__':main()

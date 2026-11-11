"""Reload every saved continuation and evaluate explicit reference equations."""
import csv
import json
import torch
from copy_exposure_continuation import ROOT,OUT,START,pools,make_model,digest,write
from audit_copy_position_sites import audit_cell


def main():
    assert not (OUT/'audit.json').exists()
    report=json.loads((OUT/'report.json').read_text());plan=json.loads((OUT/'plan.json').read_text())
    assert report['status']=='complete' and digest(OUT/'plan.json')==report['plan_sha256']
    assert digest(OUT/'preflight.json')==plan['preflight_sha256']
    assert digest(OUT/'metrics.csv')==report['metrics_sha256']
    assert all(digest(ROOT/p)==h for p,h in plan['protected'].items())
    data=pools();maximum=0.;cells=0
    for r in [dict(checkpoint=str(START.relative_to(ROOT)),checkpoint_sha256=digest(START),evaluation=report['initial'])]+report['runs']:
        assert digest(ROOT/r['checkpoint'])==r['checkpoint_sha256']
        ev=r['evaluation'];assert digest(ROOT/ev['scores'])==ev['scores_sha256']
        m=make_model('rope').eval();m.load_state_dict(torch.load(ROOT/r['checkpoint'],weights_only=True)['model'])
        saved=torch.load(ROOT/ev['scores'],weights_only=True)
        for p,rows in data['pools'].items():
            e,_=audit_cell(m,rows,None,'cross_only',saved[p],ev['metrics'][p]);maximum=max(maximum,e);cells+=1
    for i in range(0,len(report['runs']),2):
        a,b=report['runs'][i:i+2]
        assert a['stream']==b['stream'] and a['position_stream']==b['position_stream']
        for r in (a,b):
            prior=START if r['step']==6500 else OUT/f"3767_{r['arm']}_{r['step']-500}.pt"
            assert digest(prior)==r['start_sha256']
            assert r['updates']==500 and r['examples']==79750 and r['active_targets']==420000
    final={r['arm']:r['evaluation']['metrics'] for r in report['runs'] if r['step']==8000}
    own={}
    for arm,ms in final.items():
        s=min(v[g]['exact_accuracy'] for p,v in ms.items() if int(p.rsplit('_',1)[1])<=6 for g in ('distinct','repeated'))>=.99
        ll=[ms[p][g] for p,g in [('test_8','distinct'),('test_8','repeated'),('test_10','repeated')]]
        l=min(min(m['exact_accuracy'],m['on_time_eos']) for m in ll)>=.95
        own[arm]=dict(short_preserved=s,long_recovered=l,repair=s and l)
    assert own==report['gates']
    gain=final['both']['test_10']['all']['exact_accuracy']-final['cross_only']['test_10']['all']['exact_accuracy']
    assert gain==report['length10_gain']
    assert (own['both']['repair'] and gain>=.10)==report['exposure_specific_benefit']
    with (OUT/'metrics.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    for row in rows:
        run=next(r for r in report['runs'] if r['arm']==row['arm'] and r['step']==int(row['step']))
        for k,v in run['evaluation']['metrics'][row['pool']][row['group']].items():assert abs(float(row[k])-v)<1e-10
    assert len(rows)==sum(m is not None for r in report['runs'] for gs in r['evaluation']['metrics'].values() for m in gs.values())
    write(OUT/'audit.json',dict(passed=True,cells=cells,csv_rows=len(rows),max_reference_error=maximum,
        updates=sum(r['updates'] for r in report['runs']),examples=sum(r['examples'] for r in report['runs']),
        targets=sum(r['active_targets'] for r in report['runs']),protected_files=len(plan['protected']),report_sha256=digest(OUT/'report.json')))
    print('Persisted audit passed',cells,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);main()

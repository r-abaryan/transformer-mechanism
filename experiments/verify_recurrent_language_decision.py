"""Check the frozen decision independently, including CPU audit error bounds."""
import copy
import json
import math
from pathlib import Path
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def verify(report,audit):
    runs=report['runs']
    cells={(c['seed'],c['arm'],c['pool']):c for c in audit['cells']}
    completed=len(runs)==6 and all(r['status']=='trained' and r['step']==6000 for r in runs)
    guaranteed_baseline=completed and all(
        report['baselines'][pool]['loss']-1e-10-r['metrics'][pool]['loss']-
        cells[r['seed'],r['arm'],pool]['mean_loss_error'] >= .05
        for r in runs for pool in ('train_subset','validation','test64'))
    differences=[];radii=[]
    for seed in (2183,2203,2213):
        t,g=[next(r for r in runs if r['seed']==seed and r['arm']==arm) for arm in ('transformer','gru')]
        differences.append(g['metrics']['validation']['loss']-t['metrics']['validation']['loss'])
        radii.append(sum(cells[seed,arm,'validation']['mean_loss_error'] for arm in ('transformer','gru')))
    lower=math.exp(sum(d-e for d,e in zip(differences,radii))/3)
    upper=math.exp(sum(d+e for d,e in zip(differences,radii))/3)
    winner=None
    if guaranteed_baseline and lower>=1/.98 and all(d-e>0 for d,e in zip(differences,radii)):
        winner='transformer'
    elif guaranteed_baseline and upper<=.98 and all(d+e<0 for d,e in zip(differences,radii)):
        winner='gru'
    stored=report['decision']
    ratios=[math.exp(d) for d in differences]
    return dict(agrees=stored['completed']==completed and stored['learned_beyond_unigram']==guaranteed_baseline and
                stored['scoped_winner']==winner and max(abs(a-b) for a,b in zip(ratios,stored['validation_gru_over_transformer']))<1e-12 and
                abs(math.exp(sum(differences)/3)-stored['geometric_validation_ratio'])<1e-12,
                independently_derived_winner=winner,baseline_guard_with_audit_error=guaranteed_baseline,
                cpu_double_geometric_ratio_bounds=[lower,upper])


def main():
    rp=ROOT/'results/recurrent_language.json';ap=ROOT/'results/recurrent_language_audit.json'
    report=json.loads(rp.read_text());audit=json.loads(ap.read_text())
    assert audit['passed'] and audit['report_sha256']==digest(rp)
    checked=verify(report,audit);assert checked['agrees']
    wrong=copy.deepcopy(report);wrong['decision']['scoped_winner']='gru' if checked['independently_derived_winner']=='transformer' else 'transformer'
    detected=not verify(wrong,audit)['agrees'];assert detected
    result=dict(passed=True,report_sha256=digest(rp),audit_sha256=digest(ap),source_sha256=digest(Path(__file__)),
                wrong_winner_detected=detected,**checked)
    path=ROOT/'results/recurrent_language_decision_verification.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

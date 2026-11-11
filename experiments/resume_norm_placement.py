"""Verified recovery of interrupted factorial run; preserve original recipe."""
import json
from pathlib import Path
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'results/norm_placement.json'; report=json.loads(output.read_text())
    assert report['phase']=='training'
    sourcepath=ROOT/'experiments/norm_placement.py'
    for n,h in report['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    assert digest(ROOT/'docs/norm-placement-protocol.md')==report['protocol_sha256']
    assert digest(ROOT/'results/norm_placement_splits.json')==report['split_sha256']
    completed=[r for r in report['runs'] if r['status'] in ('trained','failed')]
    for r in completed:
        if r['status']=='trained': assert digest(ROOT/r['checkpoint'])==r['checkpoint_sha256'] and r['steps']==3000
        assert 'test' not in r
    interrupted=[r for r in report['runs'] if r['status']=='running']; assert len(interrupted)==1
    archive=ROOT/'results/norm_placement_interrupted.json'; assert not archive.exists(); archive.write_bytes(output.read_bytes())
    # Reuse the original, hash-verified training and test loops verbatim.
    # Only initialization is replaced with loading verified persisted state,
    # and completed conditions are skipped. The interrupted run restarts at 0.
    source=sourcepath.read_text()
    start=source.index('    assert not any(p.exists() for p in (out,sp,dp))')
    end=source.index("    for seed in (601,631,659,683,719):",start)
    setup='''    report=json.loads(out.read_text())
    report['resume']=dict(archive_sha256=digest(ROOT/'results/norm_placement_interrupted.json'),runner_sha256=digest(ROOT/'experiments/resume_norm_placement.py'),restarted=[dict(seed=r['seed'],placement=r['placement'],schedule=r['schedule'],last_saved_step=r['curve'][-1]['step']) for r in report['runs'] if r['status']=='running'])
    report['runs']=[r for r in report['runs'] if r['status'] in ('trained','failed')]
    done={(r['seed'],r['placement'],r['schedule']) for r in report['runs']}
    splits=json.loads(sp.read_text())
    data={n:(torch.tensor([x for x,y in rows]),torch.tensor([y for x,y in rows])) for n,rows in splits.items()}
    indices=torch.tensor(report['probe_indices']); px,py=(t[indices] for t in data['validation'])
    def save(): out.write_text(json.dumps(report,indent=2)+'\\n')
    directory=ROOT/'results/norm_placement_checkpoints'
    save()
'''
    source=source[:start]+setup+source[end:]
    marker="            for schedule in ('constant','warmup'):\n"
    assert source.count(marker)==1
    source=source.replace(marker,marker+"                if (seed,placement,schedule) in done: continue\n",1)
    namespace={'__name__':'norm_placement_recovery','__file__':str(sourcepath)}
    exec(compile(source,str(sourcepath),'exec'),namespace)
    namespace['main']()


if __name__=='__main__': main()

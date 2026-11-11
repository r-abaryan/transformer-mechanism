"""Independent scalar/Cartesian/document verification of source-marker evidence."""
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'results/copy_source_end_marker/count_recovery'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report=json.loads((DIRECTORY/'report.json').read_text(encoding='utf-8'))
    correction=json.loads((DIRECTORY/'count_correction.json').read_text(encoding='utf-8'))
    assert report['status']=='complete' and correction['passed'] and correction['report_sha256']==sha(DIRECTORY/'report.json')
    for dependencies in (report['design']['source_dependencies'],correction['dependencies']):
        assert all(sha(ROOT/p)==h for p,h in dependencies.items())
    d=report['design'];cases=report['cells']
    expected=set(itertools.product(d['seeds'],d['arms'],d['symbols'],map(tuple,d['source_length_pairs']),d['formats']))
    actual={(c['seed'],c['arm'],c['symbol'],tuple(c['source_lengths']),c['format']) for c in cases}
    assert len(expected)==len(actual)==len(cases)==384 and actual==expected
    max_error=0.
    for c in cases:
        rows=c['logits'];choices=[max(range(16),key=row.__getitem__) for row in rows]
        gap=max(abs(a-b) for a,b in zip(*rows));probabilities=[]
        for row in rows:
            maximum=max(row);exps=[math.exp(v-maximum) for v in row]
            probabilities.append(exps[2]/math.fsum(exps))
        margins=[row[2]-row[c['symbol']] for row in rows]
        assert c['prefix']==[1]+[c['symbol']]*c['source_lengths'][0] and c['labels']==[2,c['symbol']]
        assert choices==c['choices'] and c['both_correct']==(choices==c['labels'])
        assert c['collision']==(gap<=1e-9) and c['distinguishable']==(gap>1e-6)
        error=max(abs(gap-c['logit_difference']),*(abs(a-b) for a,b in zip(probabilities,c['eos_probabilities'])),
                  *(abs(a-b) for a,b in zip(margins,c['eos_minus_content_margins'])))
        assert error<=1e-12;max_error=max(max_error,error)
    for s in report['summaries']:
        selected=[c for c in cases if all(c[k]==s[k] for k in ('seed','arm','format'))]
        assert len(selected)==s['pairs']==16
        for key in ('collision','distinguishable','both_correct'):
            target={'collision':'collisions'}.get(key,key)
            assert sum(c[key] for c in selected)==s[target]
        assert sum(c['choices'][0]!=c['choices'][1] for c in selected)==s['different_choices']
        assert min(c['logit_difference'] for c in selected)==s['min_logit_difference']
        assert max(c['logit_difference'] for c in selected)==s['max_logit_difference']
    visible=[c for c in cases if c['arm']=='rope' and c['format']=='visible_end']
    assert report['marker_distinction']==all(c['distinguishable'] for c in visible)
    paths=['README.md','AGENTS.md','docs/research-plan.md','docs/assessments/research-checkpoint.md',
        'docs/assessments/failure-register.md','docs/assessments/copy-generalization-assessment.md',
        'docs/protocols/README.md','docs/guides/README.md','docs/topics/benchmarks/README.md',
        'docs/topics/positions/README.md','docs/topics/benchmarks/copy-source-end-marker.md',
        'docs/protocols/copy-source-end-marker.md','docs/protocols/copy-source-end-marker-count-correction.md']
    links=0
    for name in paths:
        p=ROOT/name
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if target.startswith('https://'):continue
            if 'source-end-marker' not in target and 'source_end_marker' not in target:continue
            assert (p.parent/target.split('#')[0]).resolve().exists(),(name,target);links+=1
    document=ROOT/'docs/topics/benchmarks/copy-source-end-marker.md'
    body=document.read_text(encoding='utf-8')
    for form,label in [('original','Original content'),('visible_end','Visible EOS marker'),
                       ('masked_end','EOS masked in both attentions'),('same_symbol_end','Same-symbol marker')]:
        vals=[]
        for arm in ('rope','sinusoidal'):
            selected=[c for c in cases if c['arm']==arm and c['format']==form]
            vals.extend([sum(c['distinguishable'] for c in selected),sum(c['both_correct'] for c in selected)])
        assert '| '+label+' | '+' | '.join(map(str,vals))+' |' in body
    for s in report['summaries']:
        if s['arm']=='rope' and s['format']=='visible_end':
            row=f"| {s['seed']} | {s['distinguishable']} | {s['different_choices']} | {s['both_correct']} | {s['min_logit_difference']:.6f} |"
            assert row in body
    result=dict(passed=True,cases=384,cartesian_coverage_verified=True,scalar_metrics_and_24_summaries_verified=True,
        max_scalar_error=max_error,document_table_rows_verified=7,new_local_links_verified=links,
        all_relevant_hashes_unchanged=True,report_sha256=sha(DIRECTORY/'report.json'),
        source_sha256=sha(Path(__file__)),document_sha256=sha(document))
    (DIRECTORY/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()

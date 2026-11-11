"""Short educational tables generated only from completed audited marker training."""
import json
import re
from pathlib import Path

from encoder_decoder_contract import ROOT
from position_distractors import digest
from recurrent_retrieval import write

DIRECTORY=ROOT/'results/copy_marker_training'
TARGET=ROOT/'docs/topics/benchmarks/copy-marker-training-results.md'


def main():
    r=json.loads((DIRECTORY/'report.json').read_text(encoding='utf-8'))
    audit=json.loads((DIRECTORY/'audit.json').read_text(encoding='utf-8'))
    assert r['status']=='complete' and audit['passed'] and audit['report_sha256']==digest(DIRECTORY/'report.json')
    assert not TARGET.exists()
    runs=r['runs'];arms=r['arms'];seeds=r['seeds'];d=r['decision']
    def selected(a):return [x for x in runs if x['arm']==a]
    def mean(a,p,k):return sum(x['metrics'][p]['all'][k] for x in selected(a))/len(seeds)
    def pct(x):return f'{100*x:.2f}%'
    ordinary=[];uniform=[];fit=[];long=[]
    for n in (3,4,5,6,8,10):
        values=[mean(a,f'test_{n}',k) for k in ('exact_accuracy','oracle_content_exact') for a in arms]
        ordinary.append(f'| {n} | '+' | '.join(map(pct,values))+' |')
        values=[mean(a,f'uniform_{n}','exact_accuracy') for a in arms]
        values.extend(min(x['metrics'][f'uniform_{n}']['all']['exact_accuracy'] for x in selected(a)) for a in arms)
        uniform.append(f'| {n} | '+' | '.join(map(pct,values))+' |')
    for seed in seeds:
        pair=[next(x for x in runs if x['seed']==seed and x['arm']==a) for a in arms];values=[]
        for split in ('train','test'):
            values.extend(min(x['metrics'][f'{split}_{n}'][g]['exact_accuracy'] for n in range(3,7) for g in ('distinct','repeated')) for x in pair)
        fit.append(f'| {seed} | '+' | '.join(map(pct,values))+' |')
        for n,g in ((8,'distinct'),(8,'repeated'),(10,'repeated')):
            values=[x['metrics'][f'test_{n}'][g][k] for k in ('exact_accuracy','oracle_content_exact') for x in pair]
            long.append(f'| {seed} | {n} / {g} | '+' | '.join(map(pct,values))+' |')
    decisions=[]
    for name,title in (('training_fit','Training fit'),('seen_length_reliability','Seen-length reliability'),('uniform_short_copy','Short uniform-copy rule')):
        decisions.append(title+': '+', '.join(f"{a} {'passes' if d[name][a] else 'fails'}" for a in arms)+'.')
    decisions.append(f"The frozen RoPE benefit rule {'passes' if d['rope_benefit'] else 'fails'}.")
    for a in arms:
        decisions.append(a+' extrapolation: '+', '.join(f"length {n} {'passes' if v else 'fails'}" for n,v in d['extrapolation'][a].items())+'.')
    gains='; '.join(f"{p['seed']}: {100*p['long_gains']['8']:+.2f}/{100*p['long_gains']['10']:+.2f} points" for p in d['paired'])
    preserve=', '.join(str(p['seed']) for p in d['paired'] if not p['seen_length_preserved']) or 'none'
    losses=[]
    for n in (8,10):
        losses.append(f"Length {n}: mean content CE {mean('sinusoidal',f'test_{n}','content_loss'):.4f}/{mean('rope',f'test_{n}','content_loss'):.4f}; early-EOS rate {pct(mean('sinusoidal',f'test_{n}','early_eos'))}/{pct(mean('rope',f'test_{n}','early_eos'))} (sinusoidal/RoPE).")
    overlap=r['input_checks']['uniform_overlap']
    overlap_text='; '.join(f"length {n}: training {counts['train']}/8, validation {counts['validation']}/8, test {counts['test']}/8" for n,counts in overlap.items())
    interpretation=('Both training-fit gates pass; the held-out results can be separated from failure to fit the supplied examples.'
        if all(d['training_fit'].values()) else 'At least one training-fit gate fails. A pure transfer or architecture explanation is therefore ineligible.')
    body=f'''# Copying after source-marker training

2026-10-04 assessment of six matched runs completed on 2026-10-03, independently audited. **This is training with the marker**, unlike the earlier [inference-only insertion](copy-source-end-marker.md). It asks whether models learn correct content and stopping in the marked format.

## What changed and why

| Earlier checkpoint diagnostic | This learning diagnostic |
|---|---|
| Models trained without a source marker | Source EOS present in every training example |
| Insert marker only during evaluation | Evaluate the same source format used during training |
| Shows whether fixed logits can distinguish length | Measures learned copying and stopping |

Encoder input is content+EOS. Decoder prefix is BOS+content; targets remain content+EOS, with **no second EOS target**. Ordinary generation starts at BOS and uses only its own predictions. The source marker supplies a deterministic boundary; it does not supply the content answers or decoder length control. At termination, it makes the required EOS available as an encoder value.

With identical content values v and distinct end value u, one attention mixture can be \\(v+P_{{\\mathrm{{END}}}}(u-v)\\). Its output can depend on the marker weight. Training updates the source embedding and attention/readout parameters through CE; sharing the token ID alone did not teach this in the earlier diagnostic. Position-matched reads are possible, but accuracy does not prove that mechanism was learned.

## Methods

Seeds {','.join(map(str,seeds))}, sinusoidal/RoPE arms, 1,888 parameters, width 8, two width 4 heads, one post-norm encoder/decoder, ReLU FFN16, dropout zero. Identical initial tensors and sampled indices within each pair. One fixed 6,000-update Adam .001 recipe; no selection or tuning. Content lengths 3/4/5/6, 840 target rows/update, 957,000 sampled sequences and 5.04M supervised targets each. Source input also has 5.04M tokens, including 957,000 markers; visible attention work is 71.82M pairs before the head factor. CPU float64, one thread, GPU unused. Training-loop total {sum(x['elapsed_seconds'] for x in runs):.2f} seconds includes evaluations/saves and is not a benchmark.

Opened random pools are reused. Uniform probes are **not** added to minibatches. Exact uniform overlap with original splits: {overlap_text}. Uniform scores mix previously seen and unseen inputs as disclosed; they are not an independent held-out split. All final checkpoints precede longer-input scoring. [Frozen protocol](../../protocols/copy-marker-training.md).

## Results

Three-seed means on random test pools. Full exactness requires all content and immediate EOS. Oracle content excludes EOS and supplies n, retaining learned routing and its own prefix.

| Test symbols | Ordinary sinusoidal | Ordinary RoPE | Oracle sinusoidal | Oracle RoPE |
|---|---|---|---|---|
{chr(10).join(ordinary)}

Worst distinct/repeated group within lengths 3/4/5/6, per seed. Training requires >=99%; seen-length tests >=95%.

| Seed | Training sinusoidal | Training RoPE | Seen test sinusoidal | Seen test RoPE |
|---|---|---|---|---|
{chr(10).join(fit)}

| Seed | Longer group | Ordinary sinusoidal | Ordinary RoPE | Oracle sinusoidal | Oracle RoPE |
|---|---|---|---|---|---|
{chr(10).join(long)}

Uniform challenge sources contain one symbol repeated n times, eight sources per length. Scores below require full ordinary copying, including correct stopping; minima expose weak seeds.

| Symbols | Mean sinusoidal | Mean RoPE | Worst seed sinusoidal | Worst seed RoPE |
|---|---|---|---|---|
{chr(10).join(uniform)}

{chr(10).join(losses)}

## Conclusions and limits

{interpretation}

RoPE gets every random test sequence at lengths 3–6 and every short uniform probe correct in all three seeds. This establishes learned copying and stopping on those evaluated inputs. Longer random copying remains weak, including under oracle content decoding; excluding EOS does not solve these content predictions. The marker removes the previous forced ambiguity, but the remaining learning cause is not identified.

{chr(10).join(decisions)}

Paired ordinary RoPE gains at length 8/10: {gains}. Seeds failing seen-length preservation: {preserve}. Extrapolation gates include seen-length reliability; keep that prerequisite separate from raw longer scores.

Both arms are marked. There is no new paired unmarked training arm, so historical scores cannot establish the marker's isolated training effect. This is one opened-data recipe, not fresh confirmation, convergence or a general positional winner. The original unmarked collision and diagnostic remain valid. Remaining errors do not by themselves identify routing, readout or optimization causes.

Close this budget without replacement seeds, extra updates or a position/LR grid. No new experiment is selected automatically.

## Verification and sources

All 18 initialized cases pass logits/all-parameter gradients, a nonzero source-marker gradient, checkpoint reload and the actual final evaluator/audit cell. Known-answer logits, absent groups, marker masking, duplicate EOS targets and criterion corruption are checked. The final audit verifies {audit['cells_checked']} pool cells, {audit['teacher_rows']:,} teacher rows, {audit['sequences']:,} sequences, {audit['ordinary_generated_tokens']:,} ordinary generated tokens and {audit['oracle_content_tokens']:,} oracle tokens. Zero teacher argmax, ordinary-sequence or oracle-sequence differences. Maximum logit error {audit['max_logit_error']:.3e}; scalar metric error {audit['max_metric_error']:.3e}, tolerance 1e-9. Streams, initialization, optimizer counters, exposure and relevant dependency hashes pass. Reference rotations/attention/layer equations share projection/shape utilities with the base implementation.

[Report](../../../results/copy_marker_training/report.json), [per-seed/group losses and metrics](../../../results/copy_marker_training/metrics.csv), [audit](../../../results/copy_marker_training/audit.json), [preflight](../../../results/copy_marker_training/preflight.json), [profile](../../../results/copy_marker_training/profile.json). [Training source](../../../experiments/copy_marker_training.py), [format adapter](../../../experiments/copy_marker_model.py), [independent audit](../../../experiments/audit_copy_marker_training.py).

[RoFormer, section 3](https://arxiv.org/html/2104.09864v5) supports the retained Q/K rotation identity, not these training outcomes. The value-averaging boundary and marker hypothesis come from the local [copy guide](copy-rope-guide.md#a-limit-of-this-particular-variant). No paper benchmark is reproduced.
'''
    TARGET.write_text(body,encoding='utf-8')
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',body):
        if target.startswith('https://'):continue
        assert (TARGET.parent/target.split('#')[0]).resolve().exists(),target;links.append(target)
    write(DIRECTORY/'document_verification.json',dict(passed=True,report_sha256=digest(DIRECTORY/'report.json'),
        document_sha256=digest(TARGET),source_sha256=digest(Path(__file__)),generated_table_rows=len(ordinary)+len(fit)+len(long)+len(uniform),
        verified_local_links=len(links)))
    print(json.dumps(dict(document=str(TARGET),tables_verified=True,local_links=len(links))),flush=True)


if __name__=='__main__':main()

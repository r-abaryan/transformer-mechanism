"""Build concise copy tables/figure from the completed, audited report."""
import json
import re
from pathlib import Path

from encoder_decoder_contract import ROOT
from position_distractors import digest
from recurrent_retrieval import write

DIRECTORY=ROOT/'results/copy_rope_training'
TARGET=ROOT/'docs/topics/benchmarks/copy-rope-results.md'
FIGURE=ROOT/'docs/figures/copy-rope-transfer.svg'


def main():
    report=json.loads((DIRECTORY/'report.json').read_text(encoding='utf-8'))
    audit=json.loads((DIRECTORY/'audit.json').read_text(encoding='utf-8'))
    witness=json.loads((DIRECTORY/'uniform_source_witness.json').read_text(encoding='utf-8'))
    assert report['status']=='complete' and audit['passed'] and audit['report_sha256']==digest(DIRECTORY/'report.json')
    assert witness['passed'] and witness['report_sha256']==digest(DIRECTORY/'report.json')
    assert not TARGET.exists() and not FIGURE.exists()
    runs=report['runs'];arms=report['arms'];seeds=report['seeds'];decision=report['decision']
    def selected(arm):return [r for r in runs if r['arm']==arm]
    def mean(arm,pool,key):return sum(r['metrics'][pool]['all'][key] for r in selected(arm))/3
    def pct(value):return f'{100*value:.2f}%'
    def paired(seed):return [next(r for r in runs if r['seed']==seed and r['arm']==a) for a in arms]
    means=[]
    for n in (3,4,5,6,8,10):
        vals=[mean(a,f'test_{n}',k) for k in ('exact_accuracy','oracle_content_exact') for a in arms]
        means.append(f"| {n} | "+' | '.join(map(pct,vals))+' |')
    fit=[]
    for seed in seeds:
        pair=paired(seed);vals=[]
        for split in ('train','test'):
            vals.extend(min(r['metrics'][f'{split}_{n}'][g]['exact_accuracy'] for n in range(3,7) for g in ('distinct','repeated')) for r in pair)
        fit.append(f'| {seed} | '+' | '.join(map(pct,vals))+' |')
    long=[]
    for seed in seeds:
        pair=paired(seed)
        for n,g in ((8,'distinct'),(8,'repeated'),(10,'repeated')):
            vals=[r['metrics'][f'test_{n}'][g][k] for k in ('exact_accuracy','oracle_content_exact') for r in pair]
            long.append(f'| {seed} | {n} / {g} | '+' | '.join(map(pct,vals))+' |')
    claims=[]
    if all(decision['training_fit'].values()):
        claims.append('Both training-fit gates pass. A longer-input gap can therefore be distinguished from failure to fit the supplied training sequences.')
    else:
        claims.append('At least one training-fit gate fails. The comparison does not isolate a pure length-transfer failure; optimization, capacity and the chosen recipe remain candidate explanations.')
    claims.append(f"The frozen RoPE benefit rule **{'passes' if decision['rope_benefit'] else 'fails'}**. It requires both fit gates, a 10-point ordinary exact gain at both longer lengths in every seed, and seen-length preservation.")
    claims.append('Seen-length reliability: '+', '.join(f"{a} {'passes' if decision['seen_length_reliability'][a] else 'fails'}" for a in arms)+'.')
    claims.append('RoPE extrapolation: '+', '.join(f"length {n} {'passes' if v else 'fails'}" for n,v in decision['rope_extrapolation'].items())+'. This includes the seen-length reliability prerequisite.')
    gain=[]
    for p in decision['paired']:
        gain.append(f"| {p['seed']} | {100*p['long_gains']['8']:+.2f} | {100*p['long_gains']['10']:+.2f} | {'yes' if p['seen_length_preserved'] else 'no'} |")
    elapsed=sum(r['elapsed_seconds'] for r in runs)
    losses=[]
    for n in (8,10):
        losses.append(f"Length {n}: mean teacher content CE {mean('sinusoidal',f'test_{n}','content_loss'):.4f}/{mean('rope',f'test_{n}','content_loss'):.4f}; ordinary early-EOS rate {pct(mean('sinusoidal',f'test_{n}','early_eos'))}/{pct(mean('rope',f'test_{n}','early_eos'))} (sinusoidal/RoPE).")
    body=f'''# RoPE versus sinusoidal encoder-decoder copying

2026-10-03. Six fresh matched runs, executed and independently checked. This tests positional representation on the already opened copy pools. It is distinct from decoder-only retrieval and is not fresh confirmation.

## Methods and math

Three paired seeds, 1,888 learned parameters, one width 8 post-norm encoder/decoder, two width 4 heads, ReLU FFN width 16, dropout zero. Same learned initialization, sampled indices, EOS objective and 6,000 Adam .001 updates per run. Training lengths 3/4/5/6, 840 targets/update: 957,000 sampled sequences and 5.04M supervised targets each. CPU float64, one thread; GPU unused. Recorded training-loop total {elapsed:.2f} seconds includes scheduled evaluation/saving and is not a speed benchmark.

Sinusoidal adds the original positional vectors. RoPE removes them and rotates projected Q/K in both self-attentions and cross-attention, with frequencies (1, .01). V is not directly rotated. For fixed Q/K, \\( (R_pq)^T(R_tk)=q^TR_{{t-p}}k \\). Source and decoder share origin zero; decoder row p predicts source p. This is a local copy convention, not forced alignment. Additive-position removal and Q/K rotation change together.

Ordinary full exactness requires the entire source followed by immediate EOS. Oracle content excludes EOS and supplies length, but uses learned attention and the model's own generated prefix. All 16 output classes otherwise remain available. See the [learning guide](copy-rope-guide.md) and [frozen protocol](../../protocols/copy-rope-training.md).

## Results

Percentages below are means across three seeds; subgroup/seed detail follows. Length 10 cannot have a distinct-symbol group.

| Test symbols | Ordinary sinusoidal | Ordinary RoPE | Oracle sinusoidal | Oracle RoPE |
|---|---|---|---|---|
{chr(10).join(means)}

![Ordinary full-copy means across lengths](../../figures/copy-rope-transfer.svg)

Worst distinct/repeated cell within lengths 3/4/5/6, separately per seed. These are minima, not means. Training requires 99%; seen-length test reliability requires 95%.

| Seed | Training sinusoidal | Training RoPE | Seen test sinusoidal | Seen test RoPE |
|---|---|---|---|---|
{chr(10).join(fit)}

| Seed | Longer group | Ordinary sinusoidal | Ordinary RoPE | Oracle sinusoidal | Oracle RoPE |
|---|---|---|---|---|---|
{chr(10).join(long)}

| Seed | Ordinary gain at 8, points | Gain at 10, points | Seen groups preserved within 5 points |
|---|---|---|---|
{chr(10).join(gain)}

{chr(10).join(losses)} Full training/validation/test losses and stopping groups are in the linked CSV. Early stopping rates describe behavior, not its unique cause.

## Observations, limits and closure

{chr(10).join(claims)}

RoPE's relative-position identity holds in the preflight, but it does not predict these trained accuracy scores. Adding tokens changes the attention denominator and available offsets. The trained alignment/readout mechanism and cause of the varied-content test failures are not identified. No universal RoPE winner or isolated cross-attention effect follows. Shared symbols/subsequences, opened pools and the small fixed recipe limit scope.

## A structural boundary, checked after training

For identical source symbols, the encoder starts with identical vectors. RoPE changes Q/K weights, but every V is identical, so normalized attention returns the same vector at every position. Residuals, normalization and the FFN preserve this equality. Cross-attention again averages identical values. At a common decoder prefix, source length therefore disappears from this variant's logits in ideal arithmetic.

Example: source `A A A` versus `A A A A`, decoder prefix `BOS A A A`. The correct next tokens are EOS and A, respectively. Identical logits cannot choose both. Source inputs here have no distinct end marker; this witness does not cover positional values or source formats that add such a marker.

A separately declared **posthoc** boundary check tests all eight symbols, source length pairs 3/4 and 6/8, and all three saved RoPE checkpoints: {witness['cells_checked']} paired probes, no training. Every pair has the same argmax and fails to get both conflicting labels correct. Maximum logit difference {witness['max_rope_logit_difference']:.3e}, below 1e-9. Explicit equations agree within {witness['max_reference_error']:.3e}. The sinusoidal control distinguishes all {witness['sinusoidal_distinguishable_cells']} pairs above 1e-6. This confirms a representation limit for uniform sources, not its contribution to aggregate errors on the original pools. Frozen quality criteria remain unchanged.

[Boundary design](../../../results/copy_rope_training/uniform_source_witness_design.json), [all probes](../../../results/copy_rope_training/uniform_source_witness.json), [checker](../../../experiments/check_copy_rope_uniform_source.py). This is a correctness/capability boundary within the comparison, not another training study. Do not tune the current representation as if its exact uniform-source ambiguity were an optimizer failure.

Close this six-run diagnostic at its declared budget. No automatic frequency/length-scaling search, loss-weight sweep, replacement seed or extra training is selected.

## Checks and evidence

Five reusable tests include 18 initialized model cases, all parameter gradients, a finite-difference direction, common-shift identity, causal/source/memory masks, wrong-rotation/frequency controls, criterion corruption, supplied logits and checkpoint reload into the actual final evaluator/audit cell. An environment checkpoint-access failure and duplicate test-report key were fixed before training, [recorded separately](../../../results/copy_rope_training/preflight-development.md); equations and tolerances were unchanged.

The final audit checks {audit['cells_checked']} pool cells, {audit['teacher_rows']:,} teacher rows, {audit['sequences']:,} sequences, {audit['oracle_content_tokens']:,} oracle content tokens and {audit['ordinary_generated_tokens']:,} ordinary generated tokens. Zero teacher argmax, ordinary-sequence or oracle-sequence differences. Maximum logit error {audit['max_logit_error']:.3e}; scalar metric error {audit['max_metric_error']:.3e}, against 1e-9 tolerance. Initial states, optimizer counts, streams, exposure, criteria, pretest snapshot and relevant dependency hashes pass. Native SDPA is checked against explicit rotations/exponential attention and separate normalization/FFN equations; projection/shape utilities are shared, so this is not a wholly disjoint implementation.

[Report](../../../results/copy_rope_training/report.json), [all per-group losses/metrics](../../../results/copy_rope_training/metrics.csv), [audit](../../../results/copy_rope_training/audit.json), [preflight](../../../results/copy_rope_training/preflight.json), [full-step profile](../../../results/copy_rope_training/profile.json). Producer: [training](../../../experiments/copy_rope_training.py), [model](../../../experiments/copy_rope_model.py), [independent audit](../../../experiments/audit_copy_rope_training.py), [reusable tests](../../../tests/test_copy_rope.py).

[Su et al., RoFormer, section 3](https://arxiv.org/html/2104.09864v5) supports the rotation identity. Its section 4.1.2 describes rotary self-attention for translation. Our small copy setup and shared-origin cross-attention differ; neither paper reproduction nor an exhaustive recent-literature survey is claimed.
'''
    # Documentation is generated only from completed evidence; linked local targets must exist.
    TARGET.write_text(body,encoding='utf-8')
    x=lambda n:80+(n-3)*80
    y=lambda v:270-220*v
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="740" height="340" viewBox="0 0 740 340">',
         '<rect width="740" height="340" fill="white"/>',
         '<g font-family="sans-serif" font-size="13" fill="#233047">',
         '<text x="50" y="24" font-size="18">Ordinary exact copying: three-seed means</text>',
         '<rect x="330" y="40" width="335" height="230" fill="#f2f4f8"/>',
         '<text x="367" y="44">Beyond training lengths</text>']
    for value in (0,.25,.5,.75,1):
        svg.append(f'<path d="M80 {y(value)} H665" stroke="#dce2e8"/><text x="30" y="{y(value)+4}">{value*100:.0f}%</text>')
    for n in (3,4,5,6,8,10):svg.append(f'<text x="{x(n)-4}" y="290">{n}</text>')
    svg.append('<text x="310" y="314">Source symbols</text>')
    for arm,color in (('sinusoidal','#2563eb'),('rope','#d97706')):
        points=' '.join(f'{x(n)},{y(mean(arm,f"test_{n}","exact_accuracy")):.4f}' for n in (3,4,5,6,8,10))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for n in (3,4,5,6,8,10):svg.append(f'<circle cx="{x(n)}" cy="{y(mean(arm,f"test_{n}","exact_accuracy")):.4f}" r="3" fill="{color}"/>')
    svg.extend(['<text x="80" y="334" fill="#2563eb">Sinusoidal</text><text x="190" y="334" fill="#d97706">RoPE</text>',
                '</g></svg>'])
    FIGURE.write_text('\n'.join(svg),encoding='utf-8')
    links=[]
    for link in re.findall(r'\]\(([^)]+)\)',body):
        if link.startswith('https://'):continue
        target=(TARGET.parent/link.split('#')[0]).resolve();assert target.exists(),str(target);links.append(link)
    write(DIRECTORY/'document_verification.json',dict(passed=True,report_sha256=digest(DIRECTORY/'report.json'),
        document_sha256=digest(TARGET),figure_sha256=digest(FIGURE),verified_local_links=len(links),
        generated_table_rows=len(means)+len(fit)+len(long)+len(gain),source_sha256=digest(Path(__file__))))
    print(json.dumps(dict(document=str(TARGET),verified_links=len(links))),flush=True)


if __name__=='__main__':main()

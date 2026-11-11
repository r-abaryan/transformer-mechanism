"""Saved-checkpoint boundary intervention; no new training or quality tuning."""
import json
import math
from pathlib import Path

import torch

from encoder_decoder_contract import ROOT, independent as sinusoidal_reference
from copy_rope_training import SEEDS, DIRECTORY as PARENT
from copy_rope_model import make_model, rope_reference_features
from encoder_decoder_copy import logits
from position_distractors import digest, pe
from recurrent_retrieval import write

DIRECTORY=ROOT/'results/copy_source_end_marker'
PROTOCOL=ROOT/'docs/protocols/copy-source-end-marker.md'
FORMATS=('original','visible_end','masked_end','same_symbol_end')


def forward(model, source, prefix, arm, padding, independent=False):
    tp=torch.zeros_like(prefix,dtype=torch.bool)
    if independent:
        s=model.source(source)*math.sqrt(8);t=model.target(prefix)*math.sqrt(8)
        if arm=='sinusoidal':
            s=s+pe(torch.arange(source.shape[1]),8);t=t+pe(torch.arange(prefix.shape[1]),8)
            return sinusoidal_reference(model,s,t,padding,tp)[:,-1]
        return rope_reference_features(model,s,t,padding,tp)[:,-1]
    s,t=model.features(source,prefix)
    return model(s,t,padding,tp)[:,-1]


def main():
    assert PROTOCOL.exists() and not DIRECTORY.exists()
    parent=json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    old=json.loads((PARENT/'uniform_source_witness.json').read_text(encoding='utf-8'))
    audit=json.loads((PARENT/'audit.json').read_text(encoding='utf-8'))
    assert parent['status']=='complete' and audit['passed'] and audit['report_sha256']==digest(PARENT/'report.json')
    assert old['passed'] and old['report_sha256']==digest(PARENT/'report.json')
    paths=[Path(__file__),PROTOCOL,PARENT/'report.json',PARENT/'audit.json',PARENT/'uniform_source_witness.json',
           *[ROOT/r['checkpoint'] for r in parent['runs']],
           *[ROOT/'experiments'/n for n in ('copy_rope_model.py','encoder_decoder_contract.py','encoder_decoder_copy.py',
               'copy_rope_training.py','audit_kv_cache_equivalence.py','position_distractors.py','recurrent_retrieval.py')]]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    design=dict(seeds=list(SEEDS),arms=parent['arms'],symbols=list(range(3,11)),source_length_pairs=[[3,4],[6,8]],
        formats=list(FORMATS),marker_id=2,training_updates=0,checkpoint_step=6000,
        collision_tolerance=1e-9,distinction_threshold=1e-6,source_dependencies=hashes)
    DIRECTORY.mkdir();write(DIRECTORY/'design.json',design)
    old_cases={(c['seed'],c['symbol'],tuple(c['source_lengths'])):c for c in old['cells']}
    cells=[];max_error=0.;max_mask_error=0.
    with torch.no_grad():
        for seed in SEEDS:
            for arm in parent['arms']:
                run=next(r for r in parent['runs'] if r['seed']==seed and r['arm']==arm)
                assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256']
                cp=torch.load(ROOT/run['checkpoint'],weights_only=True);assert cp['step']==6000
                model=make_model(arm);model.load_state_dict(cp['model']);originals={}
                for symbol in range(3,11):
                    for short,long in ((3,4),(6,8)):
                        prefix=torch.tensor([[1]+[symbol]*short])
                        for format in FORMATS:
                            zs=[];errors=[]
                            for n in (short,long):
                                row=[symbol]*n
                                if format!='original':row.append(symbol if format=='same_symbol_end' else 2)
                                source=torch.tensor([row]);padding=torch.zeros_like(source,dtype=torch.bool)
                                if format=='masked_end':padding[:,-1]=True
                                z=forward(model,source,prefix,arm,padding)
                                rz=forward(model,source,prefix,arm,padding,independent=True)
                                error=float((z-rz).abs().max());assert error<=1e-9
                                assert torch.equal(z.argmax(-1),rz.argmax(-1));errors.append(error);zs.append(z[0])
                            max_error=max(max_error,*errors);gap=float((zs[0]-zs[1]).abs().max())
                            choices=[int(z.argmax()) for z in zs]
                            probabilities=[float(z.softmax(-1)[2]) for z in zs]
                            margins=[float(z[2]-z[symbol]) for z in zs]
                            key=(symbol,short,long)
                            if format=='original':
                                originals[key]=zs
                                old_result=old_cases[seed,symbol,(short,long)]['results'][arm]
                                assert abs(gap-old_result['logit_difference'])<=1e-9 and choices==old_result['choices']
                            mask_error=None
                            if format=='masked_end':
                                mask_error=max(float((z-o).abs().max()) for z,o in zip(zs,originals[key]))
                                max_mask_error=max(max_mask_error,mask_error);assert mask_error<=1e-9
                            if arm=='rope' and format!='visible_end':
                                assert gap<=1e-9 and choices[0]==choices[1]
                            cells.append(dict(seed=seed,arm=arm,symbol=symbol,source_lengths=[short,long],format=format,
                                prefix=prefix[0].tolist(),labels=[2,symbol],logits=[z.tolist() for z in zs],
                                logit_difference=gap,collision=gap<=1e-9,distinguishable=gap>1e-6,
                                choices=choices,both_correct=choices==[2,symbol],eos_probabilities=probabilities,
                                eos_minus_content_margins=margins,reference_errors=errors,masked_original_error=mask_error))
    assert len(cells)==192 and all(digest(ROOT/p)==h for p,h in hashes.items())
    summaries=[]
    for seed in SEEDS:
        for arm in parent['arms']:
            for format in FORMATS:
                selected=[c for c in cells if c['seed']==seed and c['arm']==arm and c['format']==format]
                summaries.append(dict(seed=seed,arm=arm,format=format,pairs=len(selected),
                    collisions=sum(c['collision'] for c in selected),distinguishable=sum(c['distinguishable'] for c in selected),
                    different_choices=sum(c['choices'][0]!=c['choices'][1] for c in selected),
                    both_correct=sum(c['both_correct'] for c in selected),
                    min_logit_difference=min(c['logit_difference'] for c in selected),
                    max_logit_difference=max(c['logit_difference'] for c in selected)))
    visible=[c for c in cells if c['arm']=='rope' and c['format']=='visible_end']
    report=dict(status='complete',design=design,cells_checked=len(cells),source_prefix_examples=2*len(cells),
        marker_distinction=all(c['distinguishable'] for c in visible),max_reference_error=max_error,
        reference_argmax_changes=0,max_masked_original_error=max_mask_error,original_witness_verified=True,
        prior_hashes_unchanged=True,summaries=summaries,cells=cells)
    write(DIRECTORY/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('design','cells','summaries')}),flush=True)
    print(json.dumps([s for s in summaries if s['arm']=='rope' and s['format']=='visible_end']),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()

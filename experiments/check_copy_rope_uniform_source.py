"""Post-training structural witness; no tuning or change to frozen criteria."""
import json
from pathlib import Path

import torch

from copy_rope_training import ROOT, DIRECTORY, SEEDS
from copy_rope_model import make_model, reference
from encoder_decoder_copy import logits
from position_distractors import digest
from recurrent_retrieval import write


def main():
    target=DIRECTORY/'uniform_source_witness.json';assert not target.exists()
    report=json.loads((DIRECTORY/'report.json').read_text(encoding='utf-8'));assert report['status']=='complete'
    # Declare all probes before loading weights. Short source ends where the long source continues.
    design=dict(posthoc=True,training_updates=0,seeds=list(SEEDS),symbols=list(range(3,11)),
        source_length_pairs=[[3,4],[6,8]],prefix='BOS followed by short_length copies of symbol',
        tolerance=1e-9,labels='EOS for short source, symbol for long source',
        scope='Uniform content sources, this no-additive-position encoder-decoder variant; not aggregate-failure attribution')
    write(DIRECTORY/'uniform_source_witness_design.json',design)
    cells=[]
    with torch.no_grad():
        for seed in SEEDS:
            models={}
            for arm in ('sinusoidal','rope'):
                run=next(r for r in report['runs'] if r['seed']==seed and r['arm']==arm)
                assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256']
                model=make_model(arm);model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model']);models[arm]=model
            for symbol in range(3,11):
                for short,long in ((3,4),(6,8)):
                    prefix=torch.tensor([[1]+[symbol]*short]);sources=[torch.tensor([[symbol]*n]) for n in (short,long)]
                    results={}
                    for arm,model in models.items():
                        zs=[logits(model,s,prefix)[:,-1] for s in sources]
                        rs=[reference(model,s,prefix,arm)[:,-1] for s in sources]
                        reference_error=max(float((z-r).abs().max()) for z,r in zip(zs,rs));assert reference_error<=1e-9
                        gap=float((zs[0]-zs[1]).abs().max());choices=[int(z.argmax(-1)[0]) for z in zs]
                        results[arm]=dict(logit_difference=gap,reference_error=reference_error,choices=choices,
                            incompatible_labels=[2,symbol],both_correct=choices==[2,symbol])
                    assert results['rope']['logit_difference']<=design['tolerance']
                    assert results['rope']['choices'][0]==results['rope']['choices'][1] and not results['rope']['both_correct']
                    cells.append(dict(seed=seed,symbol=symbol,source_lengths=[short,long],results=results))
    negative=sum(c['results']['sinusoidal']['logit_difference']>1e-6 for c in cells);assert negative>0
    write(target,dict(passed=True,design=design,cells_checked=len(cells),
        max_rope_logit_difference=max(c['results']['rope']['logit_difference'] for c in cells),
        max_reference_error=max(c['results'][a]['reference_error'] for c in cells for a in ('sinusoidal','rope')),
        sinusoidal_distinguishable_cells=negative,rope_both_correct_cells=0,
        source_sha256=digest(Path(__file__)),report_sha256=digest(DIRECTORY/'report.json'),cells=cells))
    print(json.dumps(dict(cells=len(cells),sinusoidal_distinguishable=negative,
        max_rope_gap=max(c['results']['rope']['logit_difference'] for c in cells))),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()

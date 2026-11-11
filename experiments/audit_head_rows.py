"""Recompute stored metrics and error overlaps; verify artifact provenance."""
import json
from pathlib import Path
import torch
from head_ablation import ROOT, summarize
from position_distractors import digest


def main():
    torch.set_num_threads(1)
    rp=ROOT/'results/head_rows.json'; report=json.loads(rp.read_text())
    assert digest(ROOT/'experiments/head_rows.py')==report['source_sha256']
    assert digest(ROOT/'docs/head-row-protocol.md')==report['protocol_sha256']
    assert digest(ROOT/'results/head_ablation.json')==report['prior_sha256']
    inp=ROOT/'results/norm_extension_inputs.json'; assert digest(inp)==report['input_sha256']
    data=json.loads(inp.read_text()); ids=torch.tensor([x for x,y in data]); y=torch.tensor([y for x,y in data])
    dp=ROOT/'results/head_rows_scores.pt'; assert digest(dp)==report['scores_sha256']
    saved=torch.load(dp,weights_only=True); count=0
    assert len(saved)==len(report['runs'])==5
    for run,detail in zip(report['runs'],saved):
        assert run['seed']==detail['seed'] and len(run['conditions'])==24
        assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256']
        scores=detail['scores']; base=scores['ordinary']
        for name,c in run['conditions'].items():
            assert summarize(scores[name],y,base,ids)==c; count+=1
        for prefix,counts in run['value_all_error_overlap'].items():
            v=scores[prefix+'_values'].argmax(-1)!=y; a=scores[prefix+'_all'].argmax(-1)!=y
            assert counts==dict(both_wrong=int((v&a).sum()),values_only_wrong=int((v&~a).sum()),all_only_wrong=int((~v&a).sum()))
        for h in ('h0','h1','both'):
            assert (scores[f'b2_{h}_values']-base).abs().max()<1e-10
            assert (scores[f'b2_{h}_remaining']-base).abs().max()<1e-10
            assert (scores[f'b2_{h}_query']-scores[f'b2_{h}_all']).abs().max()<1e-10
    result=dict(passed=True,conditions=count,report_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    (ROOT/'results/head_rows_audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()

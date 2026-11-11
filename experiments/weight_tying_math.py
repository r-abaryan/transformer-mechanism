"""Analytical shared-gradient identity, with Decimal finite differences."""
import json
from decimal import Decimal, localcontext
from pathlib import Path
import torch
from torch.nn import functional as F
from position_distractors import digest

ROOT = Path(__file__).resolve().parents[1]


def decimal_loss(values, bridge, ids, targets):
    with localcontext() as ctx:
        ctx.prec = 60
        e = [[Decimal.from_float(v) for v in row] for row in values]
        b = [[Decimal.from_float(v) for v in row] for row in bridge]
        loss = Decimal(0)
        for token, target in zip(ids, targets):
            u = [sum(e[token][i]*b[i][j] for i in range(3)) for j in range(3)]
            h = [(Decimal(2)*v).exp() for v in u]
            h = [(v-1)/(v+1) for v in h]
            scores = [sum(h[j]*row[j] for j in range(3)) for row in e]
            peak = max(scores)
            loss += peak + sum((v-peak).exp() for v in scores).ln()-scores[target]
        return float(loss/len(ids))


def main():
    torch.set_num_threads(1)
    records = []
    ids = torch.tensor([0, 2, 0, 3]); targets = torch.tensor([1, 4, 3, 2])
    for seed in (7, 19, 31, 43, 59, 71):
        generator = torch.Generator().manual_seed(seed)
        e = (torch.randn(5, 3, generator=generator, dtype=torch.float64)*.3).requires_grad_()
        bridge = torch.randn(3, 3, generator=generator, dtype=torch.float64)*.4
        h = (e[ids]@bridge).tanh(); logits = h@e.T
        loss = F.cross_entropy(logits, targets)
        grad = torch.autograd.grad(loss, e)[0]
        residual = logits.softmax(-1).detach()
        residual[torch.arange(len(ids)), targets] -= 1
        residual /= len(ids)
        output_role = residual.T@h.detach()
        input_role = torch.zeros_like(e)
        back = ((residual@e.detach())*(1-h.detach().square()))@bridge.T
        input_role.index_add_(0, ids, back)
        analytical = input_role+output_role
        analytical_error = (grad-analytical).abs().max().item()
        assert analytical_error <= 1e-12
        # Duplicate vocabulary parameters, equal values, expose the two partials.
        inp=e.detach().clone().requires_grad_(); out=e.detach().clone().requires_grad_()
        untied_loss=F.cross_entropy((inp[ids]@bridge).tanh()@out.T,targets)
        gi,go=torch.autograd.grad(untied_loss,(inp,out))
        partial_error=max((gi-input_role).abs().max().item(),(go-output_role).abs().max().item())
        assert partial_error<=1e-12
        finite_error=0.; decimal_gradient_error=0.;step=1e-5
        for i in range(5):
            for j in range(3):
                plus=e.detach().clone();minus=e.detach().clone()
                plus[i,j]+=step;minus[i,j]-=step
                lp=F.cross_entropy((plus[ids]@bridge).tanh()@plus.T,targets).item()
                lm=F.cross_entropy((minus[ids]@bridge).tanh()@minus.T,targets).item()
                finite_error=max(finite_error,abs((lp-lm)/(2*step)-grad[i,j].item()))
                dp=decimal_loss(plus.tolist(),bridge.tolist(),ids.tolist(),targets.tolist())
                dm=decimal_loss(minus.tolist(),bridge.tolist(),ids.tolist(),targets.tolist())
                decimal_gradient_error=max(decimal_gradient_error,abs((dp-dm)/(2*step)-grad[i,j].item()))
        assert finite_error<=1e-8 and decimal_gradient_error<=1e-8
        decimal_error=abs(loss.item()-decimal_loss(e.detach().tolist(),bridge.tolist(),ids.tolist(),targets.tolist()))
        assert decimal_error<=1e-14
        wrong_input=(grad-input_role).abs().max().item()
        wrong_output=(grad-output_role).abs().max().item()
        assert min(wrong_input,wrong_output)>1e-4
        assert input_role[1].abs().max()==0 and output_role[1].abs().max()>1e-4
        records.append(dict(seed=seed,analytical_error=analytical_error,partial_error=partial_error,
                            finite_gradient_error=finite_error,decimal_gradient_error=decimal_gradient_error,
                            decimal_loss_error=decimal_error,omitted_output_error=wrong_input,
                            omitted_input_error=wrong_output,repeated_input_ids=ids.tolist()))
    result=dict(passed=True,cases=records,finite_difference_entries=90,source_sha256=digest(Path(__file__)),
                protocol_sha256=digest(ROOT/'docs/decisions/0006-wikitext2-weight-tying.md'))
    out=ROOT/'results/weight_tying_math.json';assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(records),entries=90,
                         max_gradient_error=max(r['analytical_error'] for r in records))),flush=True)


if __name__=='__main__':main()

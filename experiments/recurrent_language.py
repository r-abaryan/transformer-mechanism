"""Frozen WikiText-2 comparison using the existing Transformer/corpus pipeline."""
from wikitext2_training import ROOT, configure, model_new, forward_device, load_stream, tensor_hash
import copy
import hashlib
import json
import math
import time

import torch
from torch import nn
from torch.nn import functional as F
from position_distractors import digest

SEEDS = (2183, 2203, 2213)
ARMS = ('transformer', 'gru')


class LanguageGRU(nn.Module):
    def __init__(self, vocabulary):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, 16)
        self.gru = nn.GRU(16, 16, num_layers=2, batch_first=True, dropout=0)
        self.output = nn.Linear(16, vocabulary, bias=False)

    def forward(self, ids):
        hidden, _ = self.gru(self.embedding(ids))
        return self.output(hidden)


def pair(seed, vocabulary):
    torch.manual_seed(seed)
    transformer = model_new('mqa', vocabulary).float()
    torch.manual_seed(seed)
    gru = LanguageGRU(vocabulary)
    with torch.no_grad():
        gru.embedding.weight.copy_(transformer.embedding.weight)
        gru.output.weight.copy_(transformer.output.weight)
    assert [sum(p.numel() for p in m.parameters()) for m in (transformer, gru)] == [32*vocabulary+3936, 32*vocabulary+3264]
    return dict(transformer=transformer, gru=gru)


def forward(model, ids):
    return model(ids) if isinstance(model, LanguageGRU) else forward_device(model, ids)


def shared_hash(model):
    return tensor_hash(nn.ModuleDict(dict(embedding=model.embedding, output=model.output)))


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def batches(stream, context=64, subset=False):
    complete, tail = divmod(len(stream)-1, context)
    if subset:
        assert context == 64 and complete == 32634
        blocks = torch.arange(1024)*(complete-1)//1023
        assert len(blocks.unique()) == 1024
        tail = 0
    else:
        blocks = torch.arange(complete)
    for b in blocks.split(8):
        positions = b[:, None]*context+torch.arange(context)[None, :]
        yield stream[positions], stream[positions+1], torch.arange(context).expand(len(b), context), int(b[0])*context
    if tail:
        start = complete*context
        yield stream[start:start+tail][None], stream[start+1:start+tail+1][None], torch.arange(tail)[None], start


def summarize(losses, predictions, targets, positions):
    correct = predictions == targets
    def metric(mask):
        mean = losses[mask].double().mean().item()
        return dict(targets=int(mask.sum()), loss=mean, perplexity=math.exp(mean), accuracy=correct[mask].double().mean().item())
    result = metric(torch.ones_like(targets, dtype=torch.bool))
    masks = dict(early=positions<8, middle=(positions>=8)&(positions<32), late=positions>=32,
                 unknown=targets==0, row_end=targets==1, ordinary=targets>1)
    result['groups'] = {name: metric(mask) for name, mask in masks.items()}
    return result


def evaluate(model, stream, context=64, subset=False, retain=False):
    device = next(model.parameters()).device
    losses, predictions, targets, positions, fixtures = [], [], [], [], []
    with torch.no_grad():
        for index, (x, y, pos, start) in enumerate(batches(stream, context, subset)):
            logits = forward(model, x.to(device))
            losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten().to(device), reduction='none').cpu())
            predictions.append(logits.argmax(-1).flatten().cpu())
            targets.append(y.flatten())
            positions.append(pos.flatten())
            if retain and (index == 0 or x.shape[1] < context):
                fixtures.append(dict(start=start, logits=logits[:2].cpu()))
    vectors = dict(losses=torch.cat(losses), predictions=torch.cat(predictions), targets=torch.cat(targets), positions=torch.cat(positions))
    assert len(vectors['losses']) == (65536 if subset else len(stream)-1)
    assert bool(torch.isfinite(vectors['losses']).all())
    return summarize(**vectors), dict(**vectors, fixtures=fixtures) if retain else None


def unigram(train, stream, vocabulary, context=64, subset=False):
    probability = (torch.bincount(train, minlength=vocabulary).double()+1)/(len(train)+vocabulary)
    ys, ps = [], []
    for _, y, pos, _ in batches(stream, context, subset):
        ys.append(y.flatten()); ps.append(pos.flatten())
    y, positions = torch.cat(ys), torch.cat(ps)
    return summarize(-probability[y].log(), torch.full_like(y, int(probability.argmax())), y, positions)


def decision(runs, baselines):
    complete = len(runs) == 6 and all(r['status'] == 'trained' and r['step'] == 6000 for r in runs)
    beyond = complete and all(baselines[k]['loss']-r['metrics'][k]['loss'] >= .05 for r in runs for k in ('train_subset', 'validation', 'test64'))
    ratios = [math.exp(next(r for r in runs if (r['seed'], r['arm']) == (seed, 'gru'))['metrics']['validation']['loss']-
                       next(r for r in runs if (r['seed'], r['arm']) == (seed, 'transformer'))['metrics']['validation']['loss'])
              for seed in SEEDS] if complete else []
    geometric = math.exp(sum(math.log(v) for v in ratios)/3) if ratios else None
    winner = None
    if beyond and geometric <= .98 and all(v < 1 for v in ratios): winner = 'gru'
    if beyond and geometric >= 1/.98 and all(v > 1 for v in ratios): winner = 'transformer'
    return dict(completed=complete, learned_beyond_unigram=beyond, validation_gru_over_transformer=ratios,
                geometric_validation_ratio=geometric, scoped_winner=winner)


def main():
    configure()
    assert torch.cuda.is_available()
    rp = ROOT/'results/recurrent_language.json'
    assert not rp.exists(), 'Preserve existing attempts; no automatic restart.'
    encoding = json.loads((ROOT/'results/wikitext2_encoding.json').read_text())
    data_audit = json.loads((ROOT/'results/wikitext2_data_audit.json').read_text())
    assert data_audit['passed'] and data_audit['encoding_sha256'] == digest(ROOT/'results/wikitext2_encoding.json')
    vocabulary = encoding['vocabulary']
    train, validation = load_stream('train'), load_stream('validation')
    from audit_recurrent_language import preflight
    prov = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT/'experiments').glob('*.py')}
    for name in ('docs/protocols/recurrent-language.md', 'results/wikitext2_encoding.json', 'results/wikitext2_data_audit.json'):
        prov[name] = digest(ROOT/name)
    report = dict(phase='preflight', provenance=prov, torch=torch.__version__, cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name(0), preflight=preflight(train, vocabulary), runs=[],
                  baselines=dict(train_subset=unigram(train, train, vocabulary, subset=True), validation=unigram(train, validation, vocabulary)))
    write(rp, report)
    directory = ROOT/'results/recurrent_language_checkpoints'; directory.mkdir(exist_ok=True)
    train_gpu = train.cuda()
    for index, seed in enumerate(SEEDS):
        models = pair(seed, vocabulary)
        for arm in ARMS[::1 if index % 2 == 0 else -1]:
            model = models.pop(arm)
            run = dict(seed=seed, arm=arm, parameters=sum(p.numel() for p in model.parameters()),
                       initial_sha256=tensor_hash(model), shared_initial_sha256=shared_hash(model), status='running', curve=[], metrics={})
            model.cuda(); report['runs'].append(run)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            generator = torch.Generator().manual_seed(seed+1000)
            stream_hash = hashlib.sha256()
            run['curve'].append(dict(step=0, train_subset=evaluate(model, train, subset=True)[0], validation=evaluate(model, validation)[0]))
            tick = time.perf_counter(); window = []
            for step in range(1, 6001):
                starts = torch.randint(len(train)-64, (8,), generator=generator)
                stream_hash.update(json.dumps(starts.tolist()).encode())
                positions = starts.cuda()[:, None]+torch.arange(64, device='cuda')[None]
                x, y = train_gpu[positions], train_gpu[positions+1]
                optimizer.zero_grad()
                logits = forward(model, x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                finite = bool(torch.isfinite(loss))
                if finite:
                    loss.backward()
                    finite = all(p.grad is not None for p in model.parameters()) and bool(torch.stack([p.grad.isfinite().all() for p in model.parameters()]).all())
                if finite:
                    optimizer.step()
                    finite = bool(torch.stack([p.isfinite().all() for p in model.parameters()]).all())
                    window.append(loss.detach())
                if not finite: run.update(status='failed', failure_step=step)
                if step % 1000 == 0 or not finite:
                    point = dict(step=step, training_window_loss=torch.stack(window).mean().item() if window else None,
                                 train_subset=evaluate(model, train, subset=True)[0], validation=evaluate(model, validation)[0])
                    run['curve'].append(point); window=[]
                    print(json.dumps(dict(seed=seed, arm=arm, step=step, train_ppl=point['train_subset']['perplexity'], validation_ppl=point['validation']['perplexity'])), flush=True)
                if not finite: break
            if run['status'] != 'failed': run['status']='trained'
            torch.cuda.synchronize()
            cp = directory/f'{seed}_{arm}.pt'
            torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, optimizer=optimizer.state_dict(), generator=generator.get_state(), step=step), cp)
            run.update(step=step, targets=step*512, elapsed_seconds=time.perf_counter()-tick, stream_sha256=stream_hash.hexdigest(),
                       checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=digest(cp))
            if run['status']=='trained':
                last=run['curve'][-1]
                run['metrics']=dict(train_subset=last['train_subset'], validation=last['validation'])
            report['phase']='training'; write(rp, report)
            del model, optimizer, logits, loss
    report['phase']='all_attempts_finished_before_test'
    write(ROOT/'results/recurrent_language_pretest.json', report)
    test=load_stream('test')
    report['baselines'].update(test64=unigram(train, test, vocabulary), test128=unigram(train, test, vocabulary, context=128))
    for run in report['runs']:
        if run['status']!='trained':continue
        model=pair(run['seed'], vocabulary)[run['arm']].cuda()
        model.load_state_dict(torch.load(ROOT/run['checkpoint'], map_location='cpu', weights_only=True)['model'])
        details={}
        for key, seq, context, subset in [('train_subset',train,64,True), ('validation',validation,64,False), ('test64',test,64,False), ('test128',test,128,False)]:
            metric, details[key]=evaluate(model,seq,context,subset,True)
            if key in run['metrics']:assert metric==run['metrics'][key]
            run['metrics'][key]=metric
        sp=directory/f"{run['seed']}_{run['arm']}_scores.pt"; torch.save(details, sp)
        run.update(scores=str(sp.relative_to(ROOT)), scores_sha256=digest(sp)); write(rp, report)
        del model,details
    report.update(phase='complete', decision=decision(report['runs'],report['baselines']))
    write(rp,report);print(json.dumps(report['decision']),flush=True)


if __name__=='__main__':main()

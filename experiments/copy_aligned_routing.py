"""Saved-copy cross-attention routing probes with learned values retained."""
import csv
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from encoder_decoder_contract import ROOT, Model, attention, ffn
from encoder_decoder_copy import inputs, generate
from copy_stopping import intervene
from audit_kv_cache_equivalence import norm
from position_distractors import digest
from recurrent_retrieval import write

DIRECTORY=ROOT/'results/copy_aligned_routing'
PROTOCOL=ROOT/'docs/protocols/copy-aligned-routing.md'
MODES=('sham','aligned','shifted')


def route(weights,mode):
    if mode=='sham':return weights
    b,h,t,n=weights.shape
    p=weights.clone();rows=torch.arange(min(t,n))
    columns=rows if mode=='aligned' else (rows+1)%n
    p[:,:,:min(t,n)]=0
    p[:,:,rows,columns]=1
    return p


def patched(model,source,prefix,mode,reference=False):
    s,t=model.features(source,prefix)
    sp=torch.zeros_like(source,dtype=torch.bool);tp=torch.zeros_like(prefix,dtype=torch.bool)
    enc,dec=model.encoder,model.decoder
    if reference:
        memory=norm(s+attention(enc.self_attn,s,s,s,sp),enc.norm1)
        memory=norm(memory+ffn(enc,memory),enc.norm2)
        x=norm(t+attention(dec.self_attn,t,t,t,tp,causal=True),dec.norm1)
    else:
        memory=enc(s,src_key_padding_mask=sp)
        mask=torch.ones(len(prefix[0]),len(prefix[0]),dtype=torch.bool).triu(1)
        self_output=dec.self_attn(t,t,t,attn_mask=mask,need_weights=False)[0]
        x=dec.norm1(t+self_output)
    m=dec.multihead_attn;heads=m.num_heads;d=m.embed_dim//heads
    wq,wk,wv=m.in_proj_weight.chunk(3);bq,bk,bv=m.in_proj_bias.chunk(3)
    if reference:
        q=(x@wq.T+bq).reshape(len(x),len(prefix[0]),heads,d).transpose(1,2)
        k=(memory@wk.T+bk).reshape(len(x),len(source[0]),heads,d).transpose(1,2)
        v=(memory@wv.T+bv).reshape(len(x),len(source[0]),heads,d).transpose(1,2)
        raw=(q@k.transpose(-1,-2)/math.sqrt(d)).softmax(-1)
        # Independent selector applies one source value per query row.
        head=raw@v
        if mode!='sham':
            for j in range(min(len(prefix[0]),len(source[0]))):
                column=j if mode=='aligned' else (j+1)%len(source[0])
                head[:,:,j]=v[:,:,column]
        cross=head.transpose(1,2).reshape_as(x)@m.out_proj.weight.T+m.out_proj.bias
        x=norm(x+cross,dec.norm2);x=norm(x+ffn(dec,x),dec.norm3)
        z=x@model.output.weight.T
    else:
        _,raw=m(x,memory,memory,need_weights=True,average_attn_weights=False)
        v=F.linear(memory,wv,bv).reshape(len(x),len(source[0]),heads,d).transpose(1,2)
        cross=(route(raw,mode)@v).transpose(1,2).reshape_as(x)
        x=dec.norm2(x+m.out_proj(cross));x=dec.norm3(x+dec.linear2(F.relu(dec.linear1(x))))
        z=model.output(x)
    return z,raw


def reference_generate(model,source,mode):
    n=source.shape[1];prefix=torch.ones(len(source),1,dtype=torch.long)
    outputs=[[] for _ in source];active=[True]*len(source)
    for step in range(n+3):
        z=patched(model,source,prefix,mode,True)[0][:,-1]
        allowed=[k for k in range(16) if k!=2 or step>=n]
        choices=[max(allowed,key=lambda k:float(row[k])) for row in z]
        for i,k in enumerate(choices):
            if active[i]:outputs[i].append(k);active[i]=k!=2
        prefix=torch.cat((prefix,torch.tensor(choices)[:,None]),1)
        if not any(active):break
    return outputs


def metrics(z,source,outputs,weights):
    n=source.shape[1];_,labels=inputs(source)
    without=z[:,:n].clone();without[:,:,2]=-torch.inf
    result={}
    distinct=torch.tensor([len(set(row))==n for row in source.tolist()])
    for group,mask in (('all',torch.ones(len(source),dtype=torch.bool)),('distinct',distinct),('repeated',~distinct)):
        ids=torch.where(mask)[0].tolist();scores=z[mask];y=labels[mask];good=scores.argmax(-1)==y
        restricted=without[mask].argmax(-1)==source[mask]
        content=sum(outputs[i][:n]==source[i].tolist() for i in ids)
        full=sum(outputs[i]==source[i].tolist()+[2] for i in ids)
        ce=float(F.cross_entropy(scores.flatten(0,1),y.flatten()))
        scalar_losses=[]
        for row,targets in zip(scores.tolist(),y.tolist()):
            for values,target in zip(row,targets):
                maximum=max(values);scalar_losses.append(maximum+math.log(math.fsum(math.exp(v-maximum) for v in values))-values[target])
        assert abs(ce-math.fsum(scalar_losses)/len(scalar_losses))<=1e-9
        positions=[]
        for j in range(n):
            positions.append(dict(position=j+1,teacher_correct=int(good[:,j].sum()),
                teacher_correct_no_eos=int(restricted[:,j].sum()),eos_top=int((scores[:,j].argmax(-1)==2).sum()),
                mean_eos_margin=float((scores[:,j,2]-scores[:,j].gather(1,y[:,j,None]).squeeze(1)).mean()),
                native_aligned_mass=float(weights[mask,:,j,j].mean())))
        result[group]=dict(count=len(ids),teacher_content_accuracy=float(good[:,:n].double().mean()),
            teacher_content_no_eos_accuracy=float(restricted.double().mean()),teacher_content_no_eos_exact=float(restricted.all(1).double().mean()),
            loss=ce,generated_content_exact=content/len(ids),generated_full_exact=full/len(ids),positions=positions)
        # Scalar accuracy/count checks use an independent selection reduction.
        correct=restricted_correct=restricted_exact=0
        for i in ids:
            chosen=[]
            for j in range(n):
                row=z[i,j].tolist();correct+=max(range(16),key=lambda k:row[k])==int(source[i,j])
                picked=max([k for k in range(16) if k!=2],key=lambda k:row[k])
                chosen.append(picked);restricted_correct+=picked==int(source[i,j])
            restricted_exact+=chosen==source[i].tolist()
        assert correct/(len(ids)*n)==result[group]['teacher_content_accuracy']
        assert restricted_correct/(len(ids)*n)==result[group]['teacher_content_no_eos_accuracy']
        assert restricted_exact/len(ids)==result[group]['teacher_content_no_eos_exact']
    return result


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert PROTOCOL.exists() and not DIRECTORY.exists()
    old=json.loads((ROOT/'results/encoder_decoder_copy.json').read_text(encoding='utf-8'))
    stop=json.loads((ROOT/'results/copy_stopping.json').read_text(encoding='utf-8'))
    pools=json.loads((ROOT/'results/encoder_decoder_copy_data.json').read_text(encoding='utf-8'))['pools']
    paths=[PROTOCOL,Path(__file__)]+[ROOT/p for p in ('experiments/encoder_decoder_contract.py','experiments/encoder_decoder_copy.py',
        'experiments/copy_stopping.py','experiments/audit_kv_cache_equivalence.py','experiments/position_distractors.py',
        'results/encoder_decoder_copy.json','results/encoder_decoder_copy_data.json','results/encoder_decoder_copy_audit.json','results/copy_stopping.json')]
    paths+=[ROOT/run[k] for run in old['runs'] for k in ('checkpoint','scores')]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    assert digest(ROOT/'results/encoder_decoder_copy_data.json')==old['input_sha256']
    assert {r['seed'] for r in old['runs']}=={2371,2377,2381}
    # Supplied position-coded values distinguish correct and off-by-one selection.
    fixture=torch.arange(4,dtype=torch.float64).reshape(1,1,4,1)
    weights=torch.full((1,1,5,4),.25,dtype=torch.float64)
    selected=route(weights,'aligned')@fixture;wrong=route(weights,'shifted')@fixture
    assert torch.equal(selected[0,0,:4,0],torch.arange(4,dtype=torch.float64))
    assert float((selected-wrong).abs().max())>1
    assert torch.equal(route(weights,'aligned')[:,:,4],weights[:,:,4])
    DIRECTORY.mkdir();records=[];saved={};max_error=0.;generated_tokens=0;teacher_targets=0
    report=dict(status='running',device='cpu',dtype='float64',training_updates=0,source_dependencies=hashes,
                selector_error_control=True,terminal_row_preserved=True,cells=records)
    write(DIRECTORY/'report.json',report)
    for run in old['runs']:
        assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256'] and digest(ROOT/run['scores'])==run['scores_sha256']
        model=Model().eval();model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        archive=torch.load(ROOT/run['scores'],weights_only=True);saved[run['seed']]={}
        for pool,rows in pools.items():
            source=torch.tensor(rows);prefix,labels=inputs(source);conditions={};entries=[]
            with torch.no_grad():
                for mode in MODES:
                    z,w=patched(model,source,prefix,mode);rz,rw=patched(model,source,prefix,mode,True)
                    error=max(float((z-rz).abs().max()),float((w-rw).abs().max()))
                    assert error<=1e-9 and torch.equal(z.argmax(-1),rz.argmax(-1));max_error=max(max_error,error)
                    outputs=intervene(lambda a,b:patched(model,a,b,mode)[0][:,-1],source)
                    other=reference_generate(model,source,mode)
                    assert outputs==other
                    if mode=='sham':
                        error=float((z-archive[pool]['logits']).abs().max());assert error<=1e-9
                        assert outputs==stop['generated'][f"{run['seed']}/{pool}"];max_error=max(max_error,error)
                    generated_tokens+=sum(map(len,outputs));teacher_targets+=labels.numel()
                    met=metrics(z,source,outputs,w)
                    records.append(dict(seed=run['seed'],pool=pool,mode=mode,groups=met))
                    conditions[mode]=dict(logits=z,outputs=outputs,native_weights=w)
                    entries.append(dict(mode=mode,content=met['all']['generated_content_exact']))
                effect=float((conditions['aligned']['logits']-conditions['shifted']['logits']).abs().max());assert effect>1e-6
                saved[run['seed']][pool]=conditions
            print(json.dumps(dict(seed=run['seed'],pool=pool,conditions=entries)),flush=True)
            write(DIRECTORY/'report.json',report)
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    torch.save(saved,DIRECTORY/'evidence.pt')
    with (DIRECTORY/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        flat=[dict(seed=c['seed'],pool=c['pool'],mode=c['mode'],group=g,**{k:v for k,v in m.items() if k!='positions'}) for c in records for g,m in c['groups'].items()]
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    report.update(status='complete',prior_hashes_unchanged=True,evidence_sha256=digest(DIRECTORY/'evidence.pt'),
        metrics_sha256=digest(DIRECTORY/'metrics.csv'),audit=dict(passed=True,cells=len(records),teacher_targets=teacher_targets,
            generated_tokens=generated_tokens,generated_sequence_differences=0,teacher_argmax_changes=0,max_logit_weight_error=max_error,
            archived_teacher_and_blocked_generation_match=True))
    write(DIRECTORY/'report.json',report)
    print(json.dumps(report['audit']))


if __name__=='__main__':main()

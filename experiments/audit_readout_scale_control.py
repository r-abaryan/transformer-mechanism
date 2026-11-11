"""Scalar scale/loss reconstruction and independent controlled-model forwards."""
import json
import math

import torch
import audit_feedback_loss_decomposition as prior
import readout_scale_control as study


def norm(state):
    rows=[w+[b] for w,b in zip(state['output.weight'].tolist(),state['output.bias'].tolist())]
    means=[math.fsum(row[j] for row in rows)/len(rows) for j in range(len(rows[0]))]
    result=math.sqrt(math.fsum((row[j]-means[j])**2 for row in rows for j in range(len(means))))
    assert math.isfinite(result) and result>0,'Invalid centered head norm'
    return result


def verify_head(old,final,controlled,scale_record):
    a,b=norm(old),norm(final);s=a/b
    error=prior.compare(scale_record,dict(old_norm=a,final_norm=b,scale=s,temperature=1/s))
    assert final.keys()==controlled.keys()
    for name in final:
        if not name.startswith('output.'):
            assert torch.equal(final[name],controlled[name]),'Memory changed'
        else:
            v=final[name];rows=v.tolist()
            if v.ndim==1:
                m=math.fsum(rows)/len(rows);expected=[m+s*(x-m) for x in rows]
            else:
                means=[math.fsum(r[j] for r in rows)/len(rows) for j in range(len(rows[0]))]
                expected=[[means[j]+s*(x-means[j]) for j,x in enumerate(r)] for r in rows]
            difference=float((controlled[name]-torch.tensor(expected,dtype=v.dtype)).abs().max())
            assert difference<=1e-9,'Wrong head intervention';error=max(error,difference)
    return error


def reconstruct(old,final,control):
    for c in (final,control):
        assert torch.equal(old['labels'],c['labels']) and torch.equal(old['repeat'],c['repeat']),'Unaligned labels'
        assert old['logits'].shape==c['logits'].shape
    values=[]
    for za,zb,zc,y,r in zip(old['logits'].reshape(-1,6).tolist(),final['logits'].reshape(-1,6).tolist(),
                          control['logits'].reshape(-1,6).tolist(),old['labels'].flatten().tolist(),old['repeat'].flatten().tolist()):
        a,b,c=[prior.scalar(z,y) for z in (za,zb,zc)]
        assert b['prediction']==c['prediction'],'Changed winner'
        t=('correct_both' if a['correct'] else 'rescued') if b['correct'] else ('newly_wrong' if a['correct'] else 'wrong_both')
        values.append(dict(old=a,final=b,control=c,repeat=r,transition=t,removed=b['ce']-c['ce']))
    groups={}
    for group in ('all','unique','repeat'):
        scope=[v for v in values if group=='all' or v['repeat']==(group=='repeat')];n=len(scope);parts={}
        for name in ('correct_both','wrong_both','rescued','newly_wrong'):
            selected=[v for v in scope if v['transition']==name];count=len(selected);means={}
            for endpoint in ('final','control'):
                for f in ('ce','true_probability','winner_confidence'):
                    means[endpoint+'_'+f]=math.fsum(v[endpoint][f] for v in selected)/count if count else None
            parts[name]=dict(count=count,**means,ce_removed_contribution=math.fsum(v['removed'] for v in selected)/n if n else 0.,
                loss_increased=sum(v['removed'] < -1e-12 for v in selected),loss_decreased=sum(v['removed']>1e-12 for v in selected))
        means={f'{e}_ce':math.fsum(v[e]['ce'] for v in scope)/n if n else None for e in ('old','final','control')}
        removed=means['final_ce']-means['control_ce'] if n else None
        excess=means['final_ce']-means['old_ce'] if n else None
        groups[group]=dict(count=n,accuracy=sum(v['final']['correct'] for v in scope)/n if n else None,**means,
            ce_removed=removed,excess_removed_fraction=removed/excess if n and excess>1e-12 else None,
            prediction_changes=0,transitions=parts)
    return groups


def audit_cell(old,final,control,rows,record):
    _,_,labels,repeat=prior.previous.independent.inputs(rows,'all_queries')
    assert torch.equal(old['labels'],labels) and torch.equal(old['repeat'],repeat),'Wrong labels/order'
    error=prior.compare(record['groups'],reconstruct(old,final,control))
    metrics=prior.previous.metrics(control['logits'],labels,repeat)
    prior.previous.check_metrics(control['metrics'],metrics)
    assert record['complete_maps']==metrics['complete_maps']==final['metrics']['complete_maps'],'Changed maps'
    return error


def main():
    torch.set_num_threads(1)
    root=study.ROOT;out=study.OUT;replay=study.training.replay
    report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text())
    replay.check_files(lock['files'])
    assert report['status']=='complete' and report['training_updates']==0
    assert report['lock_sha256']==replay.digest(out/'lock.json')
    assert report['scores_sha256']==replay.digest(out/'scores.pt')
    assert report['scales_sha256']==replay.digest(out/'scales.json')
    assert report['scales']==json.loads((out/'scales.json').read_text())
    pairs,old_report,final_report=study.previous.inputs();scores=torch.load(out/'scores.pt',weights_only=True)
    ids=[(s,f'{split}_{m}') for s in (3461,3463) for split in ('train','validation','test') for m in (2,8,16,32)]
    assert [(c['seed'],c['pool']) for c in report['cells']]==ids
    assert [(c['seed'],c['pool']) for c in scores]==ids and len(pairs)==len(scores)==24
    errors=[];forward=[]
    for (a,b,rows),c,record in zip(pairs,scores,report['cells']):
        assert (a['seed'],a['pool'])==(b['seed'],b['pool'])==(c['seed'],c['pool'])==(record['seed'],record['pool'])
        assert record['old_checkpoint']==a['checkpoint'] and record['final_checkpoint']==b['checkpoint']
        assert record['controlled_checkpoint']==c['checkpoint']
        for cell,source in ((a,old_report),(b,final_report)):
            entry=next(x for x in source['cells'] if (x['seed'],x['arm'],x['step'],x['pool'])==(cell['seed'],cell['arm'],cell['step'],cell['pool']))
            assert entry=={k:cell[k] for k in ('seed','arm','step','pool','checkpoint','metrics')}
        cp=torch.load(root/c['checkpoint'],weights_only=True)
        assert cp['step']==12000 and report['checkpoint_hashes'][str(c['seed'])]==replay.digest(root/c['checkpoint'])
        errors.append(verify_head(torch.load(root/a['checkpoint'],weights_only=True)['model'],
            torch.load(root/b['checkpoint'],weights_only=True)['model'],cp['model'],report['scales'][str(c['seed'])]))
        errors.append(audit_cell(a,b,c,rows,record))
        forward.append(prior.previous.audit_cell(root/c['checkpoint'],rows,c)['max_logit_error'])
    assert report['answers']==sum(len(rows)*a['labels'].shape[1] for a,b,rows in pairs)==155904
    assert report['cells_count']==24 and report['max_affine_logit_error']<=1e-9
    replay.check_files(lock['files'])
    result=dict(passed=True,report_sha256=replay.digest(out/'report.json'),cells=24,answers=155904,
        training_updates=0,changed_winners=0,max_scalar_error=max(errors),max_matrix_logit_error=max(forward),
        protected_files=len(lock['files']),original_forward_audits_reused_by_hash=True)
    path=out/'audit.json';assert not path.exists();replay.write(path,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()

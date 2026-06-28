#!/usr/bin/env python3
"""Rebuild missing v17 phase2/hybrid token-fusion outputs for full8 coverage.

Uses saved 16A rendered 2D feature maps and saved full-benchmark Gaussian
language logits/codebooks. The policies are fixed/global; no question-aware or
scene-adaptive routing is used.
"""
from __future__ import annotations

import argparse, json, os, sys, time, gc, logging
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
LLAVA = ROOT / 'LLaVA-NeXT'
sys.path.insert(0, str(LLAVA))
from llava.model.builder import load_pretrained_model  # type: ignore
from llava.mm_utils import get_model_name_from_path  # type: ignore
from verify_reconstruction_quality import answer_question  # type: ignore

DEFAULT_16A = Path('/mnt/data/wangyz/exp_results/historicalAgent/eval_results/9999/full_benchmark_16a')
DEFAULT_CKPT = Path('/mnt/data/wangyz/exp_results/historicalAgent/output/9999/full_benchmark_16a')
DEFAULT_PHASE2 = Path('/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_phase2_fusion')
DEFAULT_HYBRID = Path('/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_hybrid_scan')
SCALES = [('level_0','Small',0),('level_1','Medium',1),('level_2','Large',2)]
PHASE2_METHODS = ['block','3d_first','2d_first','level_interleave','token_interleave']
HYBRID_CONFIGS = [(t,r) for t in (512,768,1024) for r in (30,50,70)]


def load_qas(root: Path, scene: str) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for p in sorted((root/scene).glob('**/analysis_no_rag/*_qa_results.json')):
        data=json.loads(p.read_text())
        out[str(data.get('image_name') or p.name.removesuffix('_qa_results.json'))]=data.get('questions',[])
    return out


def ckpt_path(root: Path, scene: str, level: int) -> Path:
    base=root/scene
    pats=[f'{scene}_9999_16A_{scene}_{level}', f'{scene}_9999_16A_{scene}_{level}_{level}', f'{scene}_9999_16A_{scene}_{level}_*']
    for pat in pats:
        hits=sorted(base.glob(pat+'/chkpnt10000.pth')) if '*' in pat else [base/pat/'chkpnt10000.pth']
        for h in hits:
            if h.exists(): return h
    raise FileNotFoundError(f'no ckpt for {scene} level {level} under {base}')


def load_level_ckpt(path: Path):
    model,_it=torch.load(path,map_location='cpu',weights_only=False)
    xyz=model[1].detach().float(); opacity=model[6].detach().float().reshape(-1); logits=model[7].detach().float(); codebook=model[8].detach().float()[0]
    return xyz, opacity, logits, codebook


def fps(signature: torch.Tensor, k: int) -> torch.Tensor:
    n=signature.shape[0]
    if k>=n: return torch.arange(n)
    sig=F.normalize(signature.float(),dim=1,eps=1e-6)
    chosen=torch.empty(k,dtype=torch.long)
    centroid=sig.mean(0,keepdim=True)
    dist=torch.cdist(sig, centroid).squeeze(1)
    idx=int(torch.argmax(dist)); chosen[0]=idx
    min_d=torch.cdist(sig, sig[idx:idx+1]).squeeze(1)
    for i in range(1,k):
        idx=int(torch.argmax(min_d)); chosen[i]=idx
        d=torch.cdist(sig, sig[idx:idx+1]).squeeze(1)
        min_d=torch.minimum(min_d,d)
    return chosen


def sample_3d(xyz, opacity, logits, codebook, k:int, strategy='farthest_feature', pool_factor=30, topk=4):
    n=logits.shape[0]
    k=min(k,n)
    op=torch.sigmoid(opacity)
    pool=min(n, max(k, k*pool_factor))
    if strategy=='opacity_topk':
        idx=torch.topk(op,k).indices
    elif strategy=='opacity_weighted':
        cand=torch.topk(op,pool).indices
        pos=torch.linspace(0, len(cand)-1, k).long()
        idx=cand[pos]
    else:
        cand=torch.topk(op,pool).indices
        if strategy=='farthest_3d':
            local=fps(xyz[cand], k); idx=cand[local]
        else:
            # feature/hybrid: use sparse code probabilities as cheap signature
            probs=torch.softmax(logits[cand],dim=1)
            vals,inds=torch.topk(probs, min(topk, probs.shape[1]), dim=1)
            sig=torch.zeros_like(probs); sig.scatter_(1,inds,vals); sig=sig/(sig.sum(1,keepdim=True)+1e-9)
            if strategy=='hybrid':
                k1=k//2; i1=cand[fps(xyz[cand], k1)] if k1 else torch.empty(0,dtype=torch.long)
                i2=cand[fps(sig, k-k1)]
                idx=torch.unique(torch.cat([i1,i2]))[:k]
                if idx.numel()<k: idx=torch.cat([idx,cand[:k-idx.numel()]])
            else:
                idx=cand[fps(sig,k)]
    probs=torch.softmax(logits[idx],dim=1)
    vals,inds=torch.topk(probs, min(topk, probs.shape[1]), dim=1)
    sparse=torch.zeros_like(probs); sparse.scatter_(1,inds,vals); sparse=sparse/(sparse.sum(1,keepdim=True)+1e-9)
    feats=sparse @ codebook
    return feats.contiguous(), idx.cpu().tolist()


def sample_2d(path: Path, k:int):
    t=torch.load(path,map_location='cpu')
    if t.dim()==4: fmap=t[0]
    else: fmap=t
    h,w,c=fmap.shape
    flat=fmap.reshape(-1,c)
    n=flat.shape[0]; k=min(k,n)
    # deterministic coverage + feature-norm ordering within even grid candidates
    cand_n=min(n, max(k*20,k))
    cand=torch.linspace(0,n-1,cand_n).long()
    cand_feat=flat[cand]
    if cand_feat.shape[0] > k:
        local=fps(cand_feat[:, ::max(1,c//64)], k)
        idx=cand[local]
    else:
        idx=cand
    feats=flat[idx].contiguous()
    meta={'source':str(path),'h':h,'w':w,'c':c,'candidates':n,'selected':int(feats.shape[0])}
    del t,fmap,flat,cand_feat
    gc.collect()
    return feats, meta


def feature_map_path(root: Path, scene: str, image: str, level: int) -> Path:
    p=root/scene/f'9999_16A_{scene}'/f'level{level}'/scene/image/f'feature_map_{image}.pt'
    if p.exists(): return p
    hits=list((root/scene).glob(f'**/level{level}/{scene}/{image}/feature_map_{image}.pt'))
    if hits: return hits[0]
    raise FileNotFoundError(p)


def fuse_tokens(tokens3d, tokens2d, method):
    if method=='3d_first': return torch.cat(tokens3d+tokens2d,0)
    if method=='2d_first': return torch.cat(tokens2d+tokens3d,0)
    if method=='level_interleave':
        parts=[]
        for a,b in zip(tokens3d,tokens2d): parts += [a,b]
        return torch.cat(parts,0)
    if method=='token_interleave':
        a=torch.cat(tokens3d,0); b=torch.cat(tokens2d,0); m=min(len(a),len(b));
        inter=torch.stack([a[:m],b[:m]],1).reshape(-1,a.shape[1])
        return torch.cat([inter,a[m:],b[m:]],0)
    return torch.cat(tokens3d+tokens2d,0)


def run_family(scene, image, qas, model, tokenizer, args, ckpts, family, label):
    if family=='phase2':
        outdir=args.phase2_out/scene/label; outfile=outdir/f'{image}_v17_p2_qa.json'; total=512; ratio=30
    else:
        outdir=args.hybrid_out/scene/label; outfile=outdir/f'{image}_v17_hybrid_qa_results.json'; total=int(label.split('_')[0][1:]); ratio=int(label.split('_')[1][1:])
    if outfile.exists() and not args.overwrite: return 'skip'
    outdir.mkdir(parents=True,exist_ok=True)
    n3d=max(3, round(total*ratio/100)); n2d=max(3,total-n3d)
    per3=[n3d//3, n3d//3, n3d-n3d//3*2]
    per2=[n2d//3, n2d//3, n2d-n2d//3*2]
    t0=time.time(); tokens3d=[]; tokens2d=[]; meta3d={}; meta2d={}
    for (lk,sn,lvl),k3,k2 in zip(SCALES,per3,per2):
        xyz,op,logits,code=ckpts[lvl]
        f3,idx=sample_3d(xyz,op,logits,code,k3,'farthest_feature',args.pool_factor,args.topk)
        tokens3d.append(f3); meta3d[lk]={'total_gaussians':int(logits.shape[0]),'sampled_count':int(f3.shape[0]),'strategy':'farthest_feature','topk':args.topk,'pool_factor':args.pool_factor,'sampled_global_indices':idx}
        f2,meta=sample_2d(feature_map_path(args.sixteen_a_root,scene,image,lvl),k2)
        tokens2d.append(f2); meta2d[sn]=meta
    fused=fuse_tokens(tokens3d,tokens2d,label if family=='phase2' else 'block')
    image_size=(int(meta2d['Large']['w']), int(meta2d['Large']['h']))
    t3d=time.time()-t0; qa_out=[]; tqa=time.time()
    for q in qas:
        ans=answer_question(model, tokenizer, fused, q.get('question',''), image_size=image_size, max_new_tokens=args.max_new_tokens)
        qa_out.append({'question_index':q.get('question_index'),'question':q.get('question',''),'expected':q.get('expected',''),'final_answer':ans,'rgb_answer':q.get('rgb_answer','')})
    payload={'image_name':image,'token_meta':{'3d':meta3d,'2d':meta2d,'fusion':label if family=='phase2' else None,'total':int(fused.shape[0])},'timing_3d_s':round(t3d,4),'questions':qa_out,'timing_qa_s':round(time.time()-tqa,4),'rebuild_note':'rebuilt missing full8 coverage from saved 2D feature maps and Gaussian language logits/codebooks'}
    outfile.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    del fused,tokens3d,tokens2d
    gc.collect(); torch.cuda.empty_cache()
    return 'done'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='+', default=['brandenburg_gate','notre_dame_front_facade'])
    ap.add_argument('--families', nargs='+', choices=['phase2','hybrid'], default=['phase2','hybrid'])
    ap.add_argument('--sixteen-a-root', type=Path, default=DEFAULT_16A)
    ap.add_argument('--ckpt-root', type=Path, default=DEFAULT_CKPT)
    ap.add_argument('--phase2-out', type=Path, default=DEFAULT_PHASE2)
    ap.add_argument('--hybrid-out', type=Path, default=DEFAULT_HYBRID)
    ap.add_argument('--model-path', default='lmms-lab/llava-onevision-qwen2-7b-ov')
    ap.add_argument('--topk', type=int, default=4); ap.add_argument('--pool-factor', type=int, default=30)
    ap.add_argument('--max-new-tokens', type=int, default=128)
    ap.add_argument('--overwrite', action='store_true')
    args=ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Loading VLM')
    name=get_model_name_from_path(args.model_path)
    tokenizer, model, _image_processor, _max_length = load_pretrained_model(args.model_path, None, name, device_map='cuda:0')
    model.eval()
    for scene in args.scenes:
        qas=load_qas(args.sixteen_a_root,scene)
        logging.info('Scene %s: %d images, %d questions', scene, len(qas), sum(len(v) for v in qas.values()))
        ckpts={lvl:load_level_ckpt(ckpt_path(args.ckpt_root,scene,lvl)) for lvl in [0,1,2]}
        labels=[]
        if 'phase2' in args.families: labels += [('phase2',m) for m in PHASE2_METHODS]
        if 'hybrid' in args.families: labels += [('hybrid',f'T{t}_R{r}') for t,r in HYBRID_CONFIGS]
        for family,label in labels:
            logging.info('== %s %s ==', family, label)
            for i,(image,qs) in enumerate(sorted(qas.items()),1):
                st=run_family(scene,image,qs,model,tokenizer,args,ckpts,family,label)
                logging.info('%s %s %s [%d/%d] %s %dQ', family,label,image,i,len(qas),st,len(qs))
        del ckpts; gc.collect()
    return 0
if __name__=='__main__': raise SystemExit(main())

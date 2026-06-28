#!/usr/bin/env python3
"""Score phase2/hybrid token-fusion outputs after full8 rebuild."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
from collections import defaultdict
from statistics import mean
from typing import Any
from compare_v17_2d3d import objective_correct
from score_16a_judgelm import allow_trusted_legacy_torch_load_for_judgelm

DEFAULT_PHASE2=Path('/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_phase2_fusion')
DEFAULT_HYBRID=Path('/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_hybrid_scan')
DEFAULT_META=Path('/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_3d_full_1061q_metrics/question_metrics.csv')
DEFAULT_LEGACY_LLAVA=Path('/home/wangyz/project/0working/Landmark-GS_12H_baseline_20260513/LLaVA-NeXT')
DEFAULT_JUDGELM_ROOT=Path('/home/wangyz/project/2past_project/JudgeLM-main')
DEFAULT_MODEL_PATH=Path('/home/wangyz/.cache/huggingface/hub/models--BAAI--JudgeLM-7B-v1.0/snapshots/dfbebe054b24c946d76bfc85c977b0d68a8be913')

def read_csv(p:Path):
    if not p.exists(): return []
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write_csv(p:Path, rows:list[dict[str,Any]], fields:list[str]):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); [w.writerow({k:r.get(k,'') for k in fields}) for r in rows]
def load_meta(p:Path):
    idx={}
    for r in read_csv(p): idx.setdefault((r.get('scene',''),r.get('image',''),str(r.get('question_index',''))),r)
    return idx

def iter_family(root:Path, family:str):
    if family=='phase2':
        for p in sorted(root.glob('*/*/*_v17_p2_qa.json')):
            data=json.loads(p.read_text()); parts=p.relative_to(root).parts
            yield parts[0], parts[1], str(data.get('image_name') or p.name.split('_v17')[0]), data, str(p)
    else:
        for p in sorted(root.glob('*/*/*_v17_hybrid_qa_results.json')):
            data=json.loads(p.read_text()); parts=p.relative_to(root).parts
            yield parts[0], parts[1], str(data.get('image_name') or p.name.split('_v17')[0]), data, str(p)

def existing(path:Path):
    out={}
    for r in read_csv(path):
        out[(r.get('family',''),r.get('label','') or r.get('strategy','') or r.get('config',''),r.get('scene',''),r.get('image',''),str(r.get('question_index','')))] = r
    return out

def build_rows(root:Path, family:str, meta:dict, old:dict):
    rows=[]
    for scene,label,image,data,source in iter_family(root,family):
        for q in data.get('questions',[]):
            qidx=str(q.get('question_index',''))
            m=meta.get((scene,image,qidx),{})
            et=m.get('eval_type') or ('judgelm' if m.get('metric_class')=='judgelm' else 'objective')
            subtype=m.get('objective_subtype') or m.get('metric_class') or ''
            expected=m.get('expected') or q.get('expected') or ''
            ans=q.get('final_answer') or ''
            rgb=q.get('rgb_answer') or m.get('rgb_answer') or ''
            r={'family':family,'label':label,'scene':scene,'image':image,'question_index':qidx,'question':m.get('question') or q.get('question') or '', 'expected':expected,'final_answer':ans,'rgb_answer':rgb,'metric_class':m.get('metric_class',''),'eval_type':et,'objective_subtype':subtype,'source_file':source}
            if et=='objective':
                r['final_correct']=objective_correct(ans,expected,subtype); r['rgb_correct']=objective_correct(rgb,expected,subtype) if rgb else ''
            else:
                o=old.get((family,label,scene,image,qidx))
                if o:
                    for k in ['score','reason','rgb_score','rgb_reason','raw_judgement','rgb_raw_judgement']:
                        if o.get(k) not in (None,''): r[k]=o.get(k)
            rows.append(r)
    return rows

def summarize(rows):
    by=defaultdict(list)
    for r in rows: by[(r['family'],r['label'])].append(r)
    out=[]
    for (fam,label),vals in sorted(by.items()):
        obj=[r for r in vals if r.get('eval_type')=='objective']
        judge=[r for r in vals if r.get('eval_type')=='judgelm' and r.get('score') not in ('',None)]
        rgbj=[r for r in judge if r.get('rgb_score') not in ('',None)]
        scenes={r['scene'] for r in vals}; imgs={(r['scene'],r['image']) for r in vals}
        out.append({'family':fam,'label':label,'scene_count':len(scenes),'image_count':len(imgs),'question_count':len(vals),'objective_count':len(obj),'judge_count':len(judge),'objective_accuracy':mean([bool(r.get('final_correct')) for r in obj]) if obj else '', 'rgb_objective_accuracy':mean([bool(r.get('rgb_correct')) for r in obj if r.get('rgb_correct') not in ('',None)]) if obj else '', 'judgelm_score':mean([float(r['score']) for r in judge]) if judge else '', 'rgb_judgelm_score':mean([float(r['rgb_score']) for r in rgbj]) if rgbj else '', 'low_validity_count':sum(1 for r in judge if float(r.get('score',0) or 0)<6), 'complete_full8':len(scenes)==8 and len(judge)>=297})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase2-root',type=Path,default=DEFAULT_PHASE2); ap.add_argument('--hybrid-root',type=Path,default=DEFAULT_HYBRID); ap.add_argument('--v17-qmetrics',type=Path,default=DEFAULT_META); ap.add_argument('--with-judgelm',action='store_true'); ap.add_argument('--families',nargs='+',default=['phase2','hybrid']); ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--legacy-llava',type=Path,default=DEFAULT_LEGACY_LLAVA); ap.add_argument('--judgelm-root',type=Path,default=DEFAULT_JUDGELM_ROOT); ap.add_argument('--judgelm-model-path',type=Path,default=DEFAULT_MODEL_PATH); ap.add_argument('--judgelm-model-id',default='JudgeLM-7B-v1.0'); ap.add_argument('--max-new-tokens',type=int,default=256); ap.add_argument('--num-gpus-per-model',type=int,default=1); ap.add_argument('--max-gpu-memory',default=None); ap.add_argument('--temperature',type=float,default=0.0); ap.add_argument('--fast-eval',type=int,default=1)
    args=ap.parse_args(); meta=load_meta(args.v17_qmetrics); all_rows=[]
    for family,root in [('phase2',args.phase2_root),('hybrid',args.hybrid_root)]:
        if family not in args.families: continue
        mdir=root/'metrics'; old=existing(mdir/'question_metrics.csv'); rows=build_rows(root,family,meta,old); all_rows+=rows
    open_rows=[r for r in all_rows if r.get('eval_type')=='judgelm']
    if args.limit: open_rows=open_rows[:args.limit]
    if args.with_judgelm:
        sys.path.insert(0,str(args.legacy_llava)); from compare_rag_results import JudgeLMSingleAnswerJudge  # type: ignore
        allow_trusted_legacy_torch_load_for_judgelm()
        judge=JudgeLMSingleAnswerJudge(judgelm_root=str(args.judgelm_root),model_path=str(args.judgelm_model_path),model_id=args.judgelm_model_id,max_new_tokens=args.max_new_tokens,num_gpus_per_model=args.num_gpus_per_model,max_gpu_memory=args.max_gpu_memory,temperature=args.temperature,if_fast_eval=args.fast_eval,cache_path=str(args.phase2_root/'metrics'/'judgelm_cache.jsonl'))
        for i,r in enumerate(open_rows,1):
            if r.get('score') in ('',None):
                print(f"JudgeLM [{i}/{len(open_rows)}] {r['family']} {r['label']} {r['scene']} {r['image']} q{r['question_index']}", flush=True)
                j=judge.judge_answer(image_name=r['image'],scale=f"{r['family']}-{r['label']}",question_index=int(r['question_index']),question=r['question'],expected=r['expected'],candidate_answer=r['final_answer'],candidate_tag=f"{r['family']}_{r['label']}")
                r['score']=float(j['candidate_score']); r['reason']=j.get('reason',''); r['raw_judgement']=j.get('raw_judgement','')
            if r.get('rgb_answer') and r.get('rgb_score') in ('',None):
                j=judge.judge_answer(image_name=r['image'],scale='rgb-reference',question_index=int(r['question_index']),question=r['question'],expected=r['expected'],candidate_answer=r['rgb_answer'],candidate_tag='rgb_reference')
                r['rgb_score']=float(j['candidate_score']); r['rgb_reason']=j.get('reason',''); r['rgb_raw_judgement']=j.get('raw_judgement','')
            if i%25==0: write_all(args, all_rows)
    write_all(args, all_rows); print(json.dumps({'summary':summarize(all_rows)},ensure_ascii=False,indent=2)); return 0

def write_all(args, all_rows):
    fields=['family','label','scene','image','question_index','question','expected','final_answer','rgb_answer','metric_class','eval_type','objective_subtype','final_correct','rgb_correct','score','reason','rgb_score','rgb_reason','source_file']
    for family,root in [('phase2',args.phase2_root),('hybrid',args.hybrid_root)]:
        if family not in args.families: continue
        rows=[r for r in all_rows if r['family']==family]; mdir=root/'metrics'; mdir.mkdir(parents=True,exist_ok=True)
        write_csv(mdir/'question_metrics.csv',rows,fields)
        summ=[r for r in summarize(rows) if r['family']==family]
        if family=='phase2':
            out=[{'strategy':r['label'], **{k:v for k,v in r.items() if k not in ('family','label')}} for r in summ]
            write_csv(mdir/'fusion_comparison.csv',out,['strategy','scene_count','image_count','question_count','objective_count','judge_count','objective_accuracy','rgb_objective_accuracy','judgelm_score','rgb_judgelm_score','low_validity_count','complete_full8'])
        else:
            out=[{'config':r['label'], **{k:v for k,v in r.items() if k not in ('family','label')}} for r in summ]
            write_csv(mdir/'config_comparison.csv',out,['config','scene_count','image_count','question_count','objective_count','judge_count','objective_accuracy','rgb_objective_accuracy','judgelm_score','rgb_judgelm_score','low_validity_count','complete_full8'])
if __name__=='__main__': raise SystemExit(main())

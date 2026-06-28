#!/usr/bin/env python3
"""Score full-8 v17 2D/3D answer-fusion candidates with Objective + JudgeLM."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from compare_v17_2d3d import objective_correct
from score_16a_judgelm import allow_trusted_legacy_torch_load_for_judgelm

DEFAULT_RUN_ROOT = Path('/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_answer_fusion_full8')
DEFAULT_V17_QMETRICS = Path('/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_3d_full_1061q_metrics/question_metrics.csv')
DEFAULT_LEGACY_LLAVA = Path('/home/wangyz/project/0working/Landmark-GS_12H_baseline_20260513/LLaVA-NeXT')
DEFAULT_JUDGELM_ROOT = Path('/home/wangyz/project/2past_project/JudgeLM-main')
DEFAULT_MODEL_PATH = Path('/home/wangyz/.cache/huggingface/hub/models--BAAI--JudgeLM-7B-v1.0/snapshots/dfbebe054b24c946d76bfc85c977b0d68a8be913')


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def load_meta(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    idx: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in read_csv(path):
        key = (row.get('scene',''), row.get('image',''), str(row.get('question_index','')))
        idx.setdefault(key, row)
    return idx


def iter_outputs(root: Path):
    for path in sorted(root.glob('*/*/*_answer_fusion_qa.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        scene = data.get('scene') or path.relative_to(root).parts[0]
        strategy = data.get('strategy') or path.relative_to(root).parts[1]
        image = str(data.get('image_name') or path.name.removesuffix('_answer_fusion_qa.json'))
        for q in data.get('questions', []):
            yield scene, strategy, image, str(q.get('question_index','')), q, str(path)


def load_existing_judgelm(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    out = {}
    try:
        for row in read_csv(path):
            out[(row['scene'], row['strategy'], row['image'], str(row['question_index']))] = row
    except Exception:
        return {}
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = defaultdict(list)
    for r in rows:
        by[r['strategy']].append(r)
    out = []
    for strategy, vals in sorted(by.items()):
        obj = [r for r in vals if r.get('eval_type') == 'objective']
        judge = [r for r in vals if r.get('eval_type') == 'judgelm' and r.get('score') not in ('', None)]
        rgb_obj = [r for r in obj if r.get('rgb_correct') not in ('', None)]
        rgb_judge = [r for r in judge if r.get('rgb_score') not in ('', None)]
        scenes = sorted({r['scene'] for r in vals})
        out.append({
            'strategy': strategy,
            'scene_count': len(scenes),
            'question_count': len(vals),
            'objective_count': len(obj),
            'objective_accuracy': mean([bool(r['final_correct']) for r in obj]) if obj else '',
            'rgb_objective_accuracy': mean([bool(r['rgb_correct']) for r in rgb_obj]) if rgb_obj else '',
            'image_count': len({(r['scene'], r['image']) for r in vals}),
            'judge_count': len(judge),
            'judgelm_count': len(judge),
            'judgelm_score': mean([float(r['score']) for r in judge]) if judge else '',
            'rgb_judgelm_score': mean([float(r['rgb_score']) for r in rgb_judge]) if rgb_judge else '',
            'low_validity_count': sum(1 for r in judge if float(r.get('score', 0) or 0) < 6),
            'complete_full8': len(scenes) == 8,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument('--v17-qmetrics', type=Path, default=DEFAULT_V17_QMETRICS)
    ap.add_argument('--with-judgelm', action='store_true')
    ap.add_argument('--legacy-llava', type=Path, default=DEFAULT_LEGACY_LLAVA)
    ap.add_argument('--judgelm-root', type=Path, default=DEFAULT_JUDGELM_ROOT)
    ap.add_argument('--judgelm-model-path', type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument('--judgelm-model-id', default='JudgeLM-7B-v1.0')
    ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--fast-eval', type=int, default=1)
    ap.add_argument('--num-gpus-per-model', type=int, default=1)
    ap.add_argument('--max-gpu-memory', default=None)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    metrics_dir = args.run_root / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(args.v17_qmetrics)
    existing_judge = load_existing_judgelm(metrics_dir / 'judgelm_question_metrics.csv')
    rows: list[dict[str, Any]] = []
    open_rows: list[dict[str, Any]] = []

    for scene, strategy, image, qidx, q, source in iter_outputs(args.run_root):
        m = meta.get((scene, image, qidx), {})
        eval_type = m.get('eval_type') or ('judgelm' if m.get('metric_class') == 'judgelm' else 'objective')
        subtype = m.get('objective_subtype') or m.get('metric_class') or ''
        expected = m.get('expected') or q.get('expected') or ''
        final_answer = q.get('final_answer') or ''
        rgb_answer = q.get('rgb_answer') or m.get('rgb_answer') or ''
        row: dict[str, Any] = {
            'index': 'v17_answer_fusion_full8',
            'scene': scene,
            'strategy': strategy,
            'image': image,
            'question_index': qidx,
            'question': m.get('question') or q.get('question') or '',
            'expected': expected,
            'final_answer': final_answer,
            'answer_2d': q.get('answer_2d') or '',
            'answer_3d': q.get('answer_3d') or '',
            'fusion_rule': q.get('fusion_rule') or '',
            'rgb_answer': rgb_answer,
            'metric_class': m.get('metric_class') or '',
            'eval_type': eval_type,
            'objective_subtype': subtype,
            'source_file': source,
        }
        if eval_type == 'objective':
            row['final_correct'] = objective_correct(final_answer, expected, subtype)
            row['rgb_correct'] = objective_correct(rgb_answer, expected, subtype) if rgb_answer else ''
        else:
            old = existing_judge.get((scene, strategy, image, qidx))
            if old and old.get('score') not in ('', None):
                row['score'] = old.get('score')
                row['reason'] = old.get('reason','')
                row['rgb_score'] = old.get('rgb_score','')
                row['rgb_reason'] = old.get('rgb_reason','')
            open_rows.append(row)
        rows.append(row)

    if args.limit:
        open_rows = open_rows[:args.limit]
    if args.with_judgelm:
        sys.path.insert(0, str(args.legacy_llava))
        from compare_rag_results import JudgeLMSingleAnswerJudge  # type: ignore
        allow_trusted_legacy_torch_load_for_judgelm()
        judge = JudgeLMSingleAnswerJudge(
            judgelm_root=str(args.judgelm_root),
            model_path=str(args.judgelm_model_path),
            model_id=args.judgelm_model_id,
            max_new_tokens=args.max_new_tokens,
            num_gpus_per_model=args.num_gpus_per_model,
            max_gpu_memory=args.max_gpu_memory,
            temperature=args.temperature,
            if_fast_eval=args.fast_eval,
            cache_path=str(metrics_dir / 'judgelm_cache.jsonl'),
        )
        for i, row in enumerate(open_rows, 1):
            if row.get('score') in ('', None):
                print(f"JudgeLM [{i}/{len(open_rows)}] {row['strategy']} {row['scene']} {row['image']} q{row['question_index']}", flush=True)
                j = judge.judge_answer(
                    image_name=row['image'],
                    scale=row['strategy'],
                    question_index=int(row['question_index']),
                    question=row['question'],
                    expected=row['expected'],
                    candidate_answer=row['final_answer'],
                    candidate_tag=row['strategy'],
                )
                row['score'] = float(j['candidate_score'])
                row['reason'] = j.get('reason','')
                row['raw_judgement'] = j.get('raw_judgement','')
            if row.get('rgb_answer') and row.get('rgb_score') in ('', None):
                j = judge.judge_answer(
                    image_name=row['image'],
                    scale='rgb-reference',
                    question_index=int(row['question_index']),
                    question=row['question'],
                    expected=row['expected'],
                    candidate_answer=row['rgb_answer'],
                    candidate_tag='rgb_reference',
                )
                row['rgb_score'] = float(j['candidate_score'])
                row['rgb_reason'] = j.get('reason','')
                row['rgb_raw_judgement'] = j.get('raw_judgement','')
            if i % 10 == 0:
                write_csv(metrics_dir / 'question_metrics.csv', rows, QUESTION_FIELDS)
                write_csv(metrics_dir / 'judgelm_question_metrics.csv', [r for r in rows if r.get('eval_type') == 'judgelm'], JUDGE_FIELDS)
                write_csv(metrics_dir / 'strategy_metrics.csv', summarize(rows), STRATEGY_FIELDS)

    write_csv(metrics_dir / 'question_metrics.csv', rows, QUESTION_FIELDS)
    write_csv(metrics_dir / 'judgelm_question_metrics.csv', [r for r in rows if r.get('eval_type') == 'judgelm'], JUDGE_FIELDS)
    strategy_rows = summarize(rows)
    write_csv(metrics_dir / 'strategy_metrics.csv', strategy_rows, STRATEGY_FIELDS)
    payload = {'strategy_metrics': strategy_rows, 'question_count': len(rows)}
    (metrics_dir / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


QUESTION_FIELDS = ['index','scene','strategy','image','question_index','question','expected','final_answer','answer_2d','answer_3d','fusion_rule','rgb_answer','metric_class','eval_type','objective_subtype','final_correct','rgb_correct','score','reason','rgb_score','rgb_reason','source_file']
JUDGE_FIELDS = ['scene','strategy','image','question_index','question','expected','final_answer','answer_2d','answer_3d','fusion_rule','score','reason','rgb_answer','rgb_score','rgb_reason','source_file']
STRATEGY_FIELDS = ['strategy','scene_count','image_count','question_count','objective_count','judge_count','objective_accuracy','rgb_objective_accuracy','judgelm_count','judgelm_score','rgb_judgelm_score','low_validity_count','complete_full8']

if __name__ == '__main__':
    raise SystemExit(main())

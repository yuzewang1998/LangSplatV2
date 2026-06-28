#!/usr/bin/env python3
"""Build recoverable full-8 2D/3D answer-fusion candidates.

This runner does not use question-aware or scene-adaptive routing. It combines
existing full-8 16A 2D answers with existing full-8 v17 3D answers using fixed,
global policies so missing phase2 token-fusion scenes can be rerun/rebuilt into
complete benchmark artifacts.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_16A_ROOT = Path('/mnt/data/wangyz/exp_results/historicalAgent/eval_results/9999/full_benchmark_16a')
DEFAULT_3D_ROOT = Path('/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_3d_full_1061q')
DEFAULT_OUT = Path('/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_answer_fusion_full8')
DEFAULT_3D_STRATEGY = 'farthest_feature'


def norm(value: str | None) -> str:
    value = (value or '').strip().lower()
    value = re.sub(r'[^a-z0-9]+', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def load_16a(root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    idx: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(root.rglob('*_qa_results.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        rel = path.relative_to(root).parts
        if not rel:
            continue
        scene = rel[0]
        image = str(data.get('image_name') or path.name.removesuffix('_qa_results.json'))
        for q in data.get('questions', []):
            qidx = str(q.get('question_index', ''))
            rendered = q.get('rendered_selected') or {}
            gt = q.get('gt_selected') or {}
            idx[(scene, image, qidx)] = {
                'scene': scene,
                'image': image,
                'question_index': qidx,
                'question': q.get('question', ''),
                'expected': q.get('expected', ''),
                'answer_2d': rendered.get('answer') or '',
                'answer_2d_scale': rendered.get('scale') or '',
                'answer_gt': gt.get('answer') or '',
                'answer_gt_scale': gt.get('scale') or '',
                'rgb_answer': q.get('rgb_answer') or '',
                'source_16a': str(path),
            }
    return idx


def load_3d(root: Path, strategy: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    idx: dict[tuple[str, str, str], dict[str, Any]] = {}
    pattern = f'*/strategy_{strategy}/*/*_v17_qa_results.json'
    for path in sorted(root.glob(pattern)):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        parts = path.relative_to(root).parts
        if len(parts) < 4:
            continue
        scene = parts[0]
        image = str(data.get('image_name') or path.name.removesuffix('_v17_qa_results.json'))
        for q in data.get('questions', []):
            qidx = str(q.get('question_index', ''))
            idx[(scene, image, qidx)] = {
                'scene': scene,
                'image': image,
                'question_index': qidx,
                'question_3d': q.get('question', ''),
                'expected_3d': q.get('expected', ''),
                'answer_3d': q.get('final_answer') or '',
                'source_3d': str(path),
                'token_meta': data.get('token_meta') or {},
            }
    return idx


def choose_answer(strategy: str, answer_2d: str, answer_3d: str) -> tuple[str, str]:
    a2 = answer_2d or ''
    a3 = answer_3d or ''
    if norm(a2) and norm(a2) == norm(a3):
        return a2, '2d_3d_agree'
    if strategy == 'answer_fusion_2d_priority':
        return (a2 or a3), 'fixed_2d_priority' if a2 else 'fallback_3d_empty_2d'
    if strategy == 'answer_fusion_3d_priority':
        return (a3 or a2), 'fixed_3d_priority' if a3 else 'fallback_2d_empty_3d'
    raise ValueError(strategy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sixteen-a-root', type=Path, default=DEFAULT_16A_ROOT)
    ap.add_argument('--three-d-root', type=Path, default=DEFAULT_3D_ROOT)
    ap.add_argument('--three-d-strategy', default=DEFAULT_3D_STRATEGY)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    two_d = load_16a(args.sixteen_a_root)
    three_d = load_3d(args.three_d_root, args.three_d_strategy)
    keys = sorted(set(two_d) & set(three_d), key=lambda k: (k[0], k[1], int(k[2] or 0)))
    strategies = ['answer_fusion_2d_priority', 'answer_fusion_3d_priority']
    counts = defaultdict(int)
    args.out.mkdir(parents=True, exist_ok=True)

    for strategy in strategies:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for key in keys:
            r2 = two_d[key]
            r3 = three_d[key]
            final, rule = choose_answer(strategy, r2['answer_2d'], r3['answer_3d'])
            q = {
                'question_index': int(key[2]) if str(key[2]).isdigit() else key[2],
                'question': r2.get('question') or r3.get('question_3d') or '',
                'expected': r2.get('expected') or r3.get('expected_3d') or '',
                'final_answer': final,
                'rgb_answer': r2.get('rgb_answer') or '',
                'answer_2d': r2.get('answer_2d') or '',
                'answer_2d_scale': r2.get('answer_2d_scale') or '',
                'answer_3d': r3.get('answer_3d') or '',
                'fusion_rule': rule,
                'source_16a': r2.get('source_16a') or '',
                'source_3d': r3.get('source_3d') or '',
            }
            grouped[(key[0], key[1])].append(q)
        for (scene, image), questions in grouped.items():
            outdir = args.out / scene / strategy
            outdir.mkdir(parents=True, exist_ok=True)
            payload = {
                'image_name': image,
                'scene': scene,
                'strategy': strategy,
                'method_note': 'fixed global answer-level 2D/3D fusion; no question-aware or scene-adaptive routing',
                'base_3d_strategy': args.three_d_strategy,
                'questions': sorted(questions, key=lambda q: int(q['question_index']) if str(q['question_index']).isdigit() else 0),
            }
            (outdir / f'{image}_answer_fusion_qa.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            counts[strategy] += len(questions)

    summary = {
        'sixteen_a_root': str(args.sixteen_a_root),
        'three_d_root': str(args.three_d_root),
        'three_d_strategy': args.three_d_strategy,
        'overlap_question_count': len(keys),
        'strategies': dict(counts),
        'scenes': sorted({k[0] for k in keys}),
        'images': len({(k[0], k[1]) for k in keys}),
    }
    (args.out / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

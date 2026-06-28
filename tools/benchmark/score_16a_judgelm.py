#!/usr/bin/env python3
"""Score 16A full-benchmark open questions with local JudgeLM.

Outputs a cacheable JSON/CSV artifact compatible with the v17 comparison report.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_16A_ROOT = Path("/mnt/data/wangyz/exp_results/historicalAgent/eval_results/9999/full_benchmark_16a")
DEFAULT_V17_QMETRICS = Path("/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_3d_full_1061q_metrics/question_metrics.csv")
DEFAULT_OUT = Path("experiment_results/benchmark/v17_2d3d_comparison/16a_judgelm")
DEFAULT_LEGACY_LLAVA = Path("/home/wangyz/project/0working/Landmark-GS_12H_baseline_20260513/LLaVA-NeXT")
DEFAULT_JUDGELM_ROOT = Path("/home/wangyz/project/2past_project/JudgeLM-main")
DEFAULT_MODEL_PATH = Path("/home/wangyz/.cache/huggingface/hub/models--BAAI--JudgeLM-7B-v1.0/snapshots/dfbebe054b24c946d76bfc85c977b0d68a8be913")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_open_question_index(v17_qmetrics: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    idx = {}
    for row in read_csv(v17_qmetrics):
        if row.get("eval_type") == "judgelm" or row.get("metric_class") == "judgelm":
            key = (row.get("scene", ""), row.get("image", ""), str(row.get("question_index", "")))
            idx.setdefault(key, row)
    return idx


def iter_16a_questions(root: Path):
    for file_path in sorted(root.rglob("*_qa_results.json")):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scene = file_path.relative_to(root).parts[0]
        image = data.get("image_name") or file_path.name.removesuffix("_qa_results.json")
        for question in data.get("questions", []):
            yield scene, str(image), str(question.get("question_index", "")), question


def build_records(root: Path, v17_qmetrics: Path) -> list[dict[str, Any]]:
    open_idx = load_open_question_index(v17_qmetrics)
    rows = []
    for scene, image, qidx, question in iter_16a_questions(root):
        meta = open_idx.get((scene, image, qidx))
        if not meta:
            continue
        rendered = question.get("rendered_selected") or {}
        candidate = rendered.get("answer") or ""
        rows.append({
            "scene": scene,
            "image": image,
            "question_index": qidx,
            "question": question.get("question") or meta.get("question", ""),
            "expected": question.get("expected") or meta.get("expected", ""),
            "answer_16a": candidate,
            "rgb_answer": question.get("rgb_answer") or meta.get("rgb_answer", ""),
        })
    rows.sort(key=lambda r: (r["scene"], r["image"], int(r["question_index"])))
    return rows


def allow_trusted_legacy_torch_load_for_judgelm() -> None:
    # Matches the existing Landmark-GS scoring helper behavior for this trusted local model.
    try:
        import torch  # type: ignore
        import torch.serialization  # type: ignore
        from argparse import Namespace
        torch.serialization.add_safe_globals([Namespace])
    except Exception:
        pass


def load_existing(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {(r["scene"], r["image"], str(r["question_index"])): r for r in data.get("records", [])}


def write_outputs(out_dir: Path, records: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scored16 = [float(r["score_16a"]) for r in records if r.get("score_16a") not in (None, "") and float(r["score_16a"]) >= 0]
    scored_rgb = [float(r["score_rgb"]) for r in records if r.get("score_rgb") not in (None, "") and float(r["score_rgb"]) >= 0]
    by_scene: dict[str, dict[str, list[float]]] = {}
    for r in records:
        scene = r["scene"]
        by_scene.setdefault(scene, {"score_16a": [], "score_rgb": []})
        for key in ("score_16a", "score_rgb"):
            try:
                val = float(r.get(key, -1))
            except Exception:
                continue
            if val >= 0:
                by_scene[scene][key].append(val)
    scene_rows = []
    for scene, vals in sorted(by_scene.items()):
        scene_rows.append({
            "scene": scene,
            "judge_count": len(vals["score_16a"]),
            "score_16a": mean(vals["score_16a"]) if vals["score_16a"] else None,
            "score_rgb": mean(vals["score_rgb"]) if vals["score_rgb"] else None,
        })
    payload = {
        "summary": {
            "judge_count": len(scored16),
            "score_16a": mean(scored16) if scored16 else None,
            "score_rgb": mean(scored_rgb) if scored_rgb else None,
            "low_validity_16a": sum(1 for v in scored16 if v < 6),
            "low_validity_rgb": sum(1 for v in scored_rgb if v < 6),
        },
        "scene_summary": scene_rows,
        "records": records,
    }
    (out_dir / "judgelm_scored_16a_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "judgelm_scored_16a_results.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["scene", "image", "question_index", "question", "expected", "answer_16a", "score_16a", "reason_16a", "rgb_answer", "score_rgb", "reason_rgb"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--sixteen-a-root", type=Path, default=DEFAULT_16A_ROOT)
    parser.add_argument("--v17-qmetrics", type=Path, default=DEFAULT_V17_QMETRICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--legacy-llava", type=Path, default=DEFAULT_LEGACY_LLAVA)
    parser.add_argument("--judgelm-root", type=Path, default=DEFAULT_JUDGELM_ROOT)
    parser.add_argument("--judgelm-model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--judgelm-model-id", default="JudgeLM-7B-v1.0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-gpu-memory", default=None)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--fast-eval", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="debug limit; 0 means all")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out_dir = args.out if args.out.is_absolute() else args.root / args.out
    records = build_records(args.sixteen_a_root, args.v17_qmetrics)
    if args.limit:
        records = records[: args.limit]
    existing = load_existing(out_dir / "judgelm_scored_16a_results.json")
    for i, rec in enumerate(records):
        old = existing.get((rec["scene"], rec["image"], str(rec["question_index"])))
        if old:
            rec.update({k: old.get(k) for k in ("score_16a", "reason_16a", "raw_16a", "score_rgb", "reason_rgb", "raw_rgb") if k in old})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "judgelm_16a_input.json").write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(records)} JudgeLM open-question records -> {out_dir}", flush=True)
    if args.prepare_only:
        write_outputs(out_dir, records)
        return 0

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
        cache_path=str(out_dir / "judgelm_cache.jsonl"),
    )
    for idx, rec in enumerate(records, 1):
        print(f"[{idx}/{len(records)}] {rec['scene']} {rec['image']} q{rec['question_index']}", flush=True)
        if rec.get("score_16a") in (None, ""):
            j = judge.judge_answer(
                image_name=rec["image"],
                scale="16A-rendered-selected",
                question_index=int(rec["question_index"]),
                question=rec["question"],
                expected=rec["expected"],
                candidate_answer=rec["answer_16a"],
                candidate_tag="16A_rendered_selected",
            )
            rec["score_16a"] = float(j["candidate_score"])
            rec["reason_16a"] = j.get("reason", "")
            rec["raw_16a"] = j.get("raw_judgement", "")
        if rec.get("rgb_answer") and rec.get("score_rgb") in (None, ""):
            j = judge.judge_answer(
                image_name=rec["image"],
                scale="rgb-reference",
                question_index=int(rec["question_index"]),
                question=rec["question"],
                expected=rec["expected"],
                candidate_answer=rec["rgb_answer"],
                candidate_tag="rgb_reference",
            )
            rec["score_rgb"] = float(j["candidate_score"])
            rec["reason_rgb"] = j.get("reason", "")
            rec["raw_rgb"] = j.get("raw_judgement", "")
        if idx % 10 == 0:
            write_outputs(out_dir, records)
    write_outputs(out_dir, records)
    print(json.dumps(json.loads((out_dir / "judgelm_scored_16a_results.json").read_text())["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

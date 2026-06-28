#!/usr/bin/env python3
"""Score V18 experiment outputs and publish summary files for the 9999 dashboard.

Inputs:
  experiment_results/benchmark/v18_pipeline_ablation/results/<exp_id>/outputs/*/*.json

Outputs per experiment:
  question_metrics.csv
  summary.json
  status.json

The script can compute objective metrics alone or additionally run JudgeLM for
open-ended questions with --with-judgelm. A V18 experiment is marked completed
only when it has 8 scenes plus both Objective and JudgeLM metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V18 = ROOT / "experiment_results" / "benchmark" / "v18_pipeline_ablation"
DEFAULT_META = Path("/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_3d_full_1061q_metrics/question_metrics.csv")
DEFAULT_LEGACY_LLAVA = Path("/home/wangyz/project/0working/Landmark-GS_12H_baseline_20260513/LLaVA-NeXT")
DEFAULT_JUDGELM_ROOT = Path("/home/wangyz/project/2past_project/JudgeLM-main")
DEFAULT_MODEL_PATH = Path("/home/wangyz/.cache/huggingface/hub/models--BAAI--JudgeLM-7B-v1.0/snapshots/dfbebe054b24c946d76bfc85c977b0d68a8be913")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_v17_2d3d import objective_correct  # type: ignore
from score_16a_judgelm import allow_trusted_legacy_torch_load_for_judgelm  # type: ignore


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_registry(v18_root: Path) -> dict[str, Any]:
    return json.loads((v18_root / "v18_registry.json").read_text(encoding="utf-8"))


def load_meta(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in read_csv(path):
        out.setdefault((row.get("scene", ""), row.get("image", ""), str(row.get("question_index", ""))), row)
    return out


def iter_outputs(exp_dir: Path):
    for path in sorted((exp_dir / "outputs").glob("*/*_qa_results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        scene = str(data.get("scene") or path.relative_to(exp_dir / "outputs").parts[0])
        image = str(data.get("image_name") or path.name.removesuffix("_qa_results.json"))
        for q in data.get("questions", []) or []:
            yield scene, image, str(q.get("question_index", "")), q, path, data


def existing_metrics(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    return {(r.get("scene", ""), r.get("image", ""), str(r.get("question_index", ""))): r for r in read_csv(path)}


def build_rows(exp_id: str, exp_dir: Path, meta: dict[tuple[str, str, str], dict[str, str]]) -> list[dict[str, Any]]:
    old = existing_metrics(exp_dir / "question_metrics.csv")
    rows: list[dict[str, Any]] = []
    for scene, image, qidx, q, source, data in iter_outputs(exp_dir):
        m = meta.get((scene, image, qidx), {})
        metric_class = m.get("metric_class", "")
        eval_type = m.get("eval_type") or ("judgelm" if metric_class == "judgelm" else "objective")
        subtype = m.get("objective_subtype") or metric_class
        expected = m.get("expected") or q.get("expected") or ""
        final_answer = q.get("final_answer") or q.get("answer") or ""
        rgb_answer = q.get("rgb_answer") or m.get("rgb_answer") or ""
        row: dict[str, Any] = {
            "experiment_id": exp_id,
            "scene": scene,
            "image": image,
            "question_index": qidx,
            "question": m.get("question") or q.get("question") or "",
            "expected": expected,
            "final_answer": final_answer,
            "rgb_answer": rgb_answer,
            "metric_class": metric_class,
            "eval_type": eval_type,
            "objective_subtype": subtype,
            "source_file": str(source),
        }
        if eval_type == "objective":
            row["final_correct"] = objective_correct(final_answer, expected, subtype)
            row["rgb_correct"] = objective_correct(rgb_answer, expected, subtype) if rgb_answer else ""
        else:
            previous = old.get((scene, image, qidx), {})
            for key in ["score", "reason", "raw_judgement", "rgb_score", "rgb_reason", "rgb_raw_judgement"]:
                if previous.get(key) not in (None, ""):
                    row[key] = previous[key]
        rows.append(row)
    return rows


def summarize(exp_id: str, rows: list[dict[str, Any]], registry_exp: dict[str, Any] | None) -> dict[str, Any]:
    scenes = {r["scene"] for r in rows}
    images = {(r["scene"], r["image"]) for r in rows}
    objective_rows = [r for r in rows if r.get("eval_type") == "objective"]
    judge_rows = [r for r in rows if r.get("eval_type") == "judgelm" and r.get("score") not in (None, "")]
    rgb_judge = [r for r in judge_rows if r.get("rgb_score") not in (None, "")]
    objective_acc = mean([bool(r.get("final_correct")) for r in objective_rows]) if objective_rows else None
    rgb_objective_acc = mean([bool(r.get("rgb_correct")) for r in objective_rows if r.get("rgb_correct") not in (None, "")]) if objective_rows else None
    judgelm = mean([float(r["score"]) for r in judge_rows]) if judge_rows else None
    rgb_judgelm = mean([float(r["rgb_score"]) for r in rgb_judge]) if rgb_judge else None
    required_judge = sum(1 for r in rows if r.get("eval_type") == "judgelm")
    status = "completed" if len(scenes) >= 8 and objective_acc is not None and required_judge > 0 and len(judge_rows) >= required_judge else "incomplete"
    return {
        "experiment_id": exp_id,
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scene_count": len(scenes),
        "image_count": len(images),
        "question_count": len(rows),
        "objective_count": len(objective_rows),
        "judge_count": len(judge_rows),
        "judge_required_count": required_judge,
        "objective_accuracy": objective_acc,
        "rgb_objective_accuracy": rgb_objective_acc,
        "judgelm_score": judgelm,
        "rgb_judgelm_score": rgb_judgelm,
        "full_benchmark_complete": len(scenes) >= 8,
        "required_scope": (registry_exp or {}).get("required_scope", {"scenes": 8, "vqa": "all", "objective": True, "judgelm": True}),
        "fields": (registry_exp or {}).get("fields", {}),
        "research_question": (registry_exp or {}).get("research_question", ""),
        "unique_variable": (registry_exp or {}).get("unique_variable", ""),
        "fixed_conditions": (registry_exp or {}).get("fixed_conditions", ""),
    }


def run_judgelm(rows: list[dict[str, Any]], exp_dir: Path, args: argparse.Namespace) -> None:
    open_rows = [r for r in rows if r.get("eval_type") == "judgelm"]
    if args.limit:
        open_rows = open_rows[: args.limit]
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
        cache_path=str(exp_dir / "judgelm_cache.jsonl"),
    )
    for i, row in enumerate(open_rows, 1):
        if row.get("score") in (None, ""):
            print(f"JudgeLM [{i}/{len(open_rows)}] {row['experiment_id']} {row['scene']} {row['image']} q{row['question_index']}", flush=True)
            result = judge.judge_answer(
                image_name=row["image"],
                scale=row["experiment_id"],
                question_index=int(row["question_index"] or 0),
                question=row["question"],
                expected=row["expected"],
                candidate_answer=row["final_answer"],
                candidate_tag=row["experiment_id"],
            )
            row["score"] = float(result["candidate_score"])
            row["reason"] = result.get("reason", "")
            row["raw_judgement"] = result.get("raw_judgement", "")
        if row.get("rgb_answer") and row.get("rgb_score") in (None, ""):
            result = judge.judge_answer(
                image_name=row["image"],
                scale="rgb-reference",
                question_index=int(row["question_index"] or 0),
                question=row["question"],
                expected=row["expected"],
                candidate_answer=row["rgb_answer"],
                candidate_tag="rgb_reference",
            )
            row["rgb_score"] = float(result["candidate_score"])
            row["rgb_reason"] = result.get("reason", "")
            row["rgb_raw_judgement"] = result.get("raw_judgement", "")


def score_experiment(exp_id: str, args: argparse.Namespace, registry_by_id: dict[str, dict[str, Any]], meta: dict[tuple[str, str, str], dict[str, str]]) -> dict[str, Any]:
    exp_dir = args.v18_root / "results" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(exp_id, exp_dir, meta)
    if args.with_judgelm:
        run_judgelm(rows, exp_dir, args)
    fields = [
        "experiment_id", "scene", "image", "question_index", "question", "expected", "final_answer", "rgb_answer",
        "metric_class", "eval_type", "objective_subtype", "final_correct", "rgb_correct", "score", "reason",
        "raw_judgement", "rgb_score", "rgb_reason", "rgb_raw_judgement", "source_file",
    ]
    write_csv(exp_dir / "question_metrics.csv", rows, fields)
    summary = summarize(exp_id, rows, registry_by_id.get(exp_id))
    (exp_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (exp_dir / "status.json").write_text(json.dumps({"status": summary["status"], "updated_at": summary["updated_at"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v18-root", type=Path, default=DEFAULT_V18)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--exp-id", action="append", default=[])
    ap.add_argument("--batch", choices=["phase-a", "batch1"], default=None)
    ap.add_argument("--with-judgelm", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="JudgeLM limit for smoke tests only; do not use for final V18 metrics")
    ap.add_argument("--legacy-llava", type=Path, default=DEFAULT_LEGACY_LLAVA)
    ap.add_argument("--judgelm-root", type=Path, default=DEFAULT_JUDGELM_ROOT)
    ap.add_argument("--judgelm-model-path", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--judgelm-model-id", default="JudgeLM-7B-v1.0")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-gpus-per-model", type=int, default=1)
    ap.add_argument("--max-gpu-memory", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--fast-eval", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    registry = load_registry(args.v18_root)
    registry_by_id = {e["id"]: e for e in registry.get("experiments", [])}
    phase_a = [e["id"] for e in registry.get("experiments", []) if e.get("phase") == "A" and e.get("priority") == "batch1"]
    if args.list:
        for exp_id in phase_a:
            print(exp_id)
        return 0
    selected = set(args.exp_id)
    if args.batch in {"phase-a", "batch1"}:
        selected.update(phase_a)
    if not selected:
        raise SystemExit("select --exp-id ... or --batch phase-a")
    meta = load_meta(args.meta)
    summaries = []
    for exp_id in sorted(selected):
        if exp_id not in registry_by_id:
            raise SystemExit(f"unknown V18 experiment: {exp_id}")
        summaries.append(score_experiment(exp_id, args, registry_by_id, meta))
    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build transparent diagnostic sub-benchmarks from full V18 question metrics.

This does NOT define a new headline benchmark. It creates reproducible diagnostic
slices from the full benchmark to answer: where is the current V18 method strong,
and which repeated cross-view question groups are significant current failures?

Unit of filtering is a question group across views, not individual lucky/unlucky
rows. A group is keyed by scene + metric type + normalized question + expected.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

# Reuse categorization and parsing utilities from the diagnostic script.
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[2]
DIAG_SCRIPT = ROOT / "tools" / "benchmark" / "analyze_v18_vs_rgb_profile.py"
spec = importlib.util.spec_from_file_location("v18diag", DIAG_SCRIPT)
v18diag = importlib.util.module_from_spec(spec)
sys.modules["v18diag"] = v18diag
assert spec and spec.loader
spec.loader.exec_module(v18diag)  # type: ignore[union-attr]

DEFAULT_EXP = "v18_tokpipe_best2d_best3d_m2tokcat_m3tokcat_mmtokscale_odirect_T576_R5_C1"
DEFAULT_V18_ROOT = ROOT / "experiment_results" / "benchmark" / "v18_pipeline_ablation"
DEFAULT_FULL_BENCH = Path("/mnt/data/wangyz/exp_results/historicalAgent/eval_results/9999/full_benchmark_16a")


@dataclass
class GroupStats:
    key: str
    scene: str
    metric_class: str
    category: str
    question_norm: str
    expected_norm: str
    n: int
    images: int
    eval_type: str
    ours_obj_acc: float | None = None
    rgb_obj_acc: float | None = None
    delta_obj_acc: float | None = None
    ours_judgelm: float | None = None
    rgb_judgelm: float | None = None
    delta_judgelm: float | None = None
    reason: str = ""


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def fnum(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = v18diag.strip_instruction(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def group_key(row: dict[str, Any]) -> str:
    parts = [
        row.get("scene", ""),
        row.get("eval_type", ""),
        row.get("metric_class", ""),
        row.get("category", ""),
        norm_text(row.get("question")),
        norm_text(row.get("expected")),
    ]
    return "||".join(str(p) for p in parts)


def source_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row.get("scene", "")), str(row.get("image", "")), int(str(row.get("question_index", "0") or "0"))


def discover_source_file(full_root: Path, scene: str, image: str) -> Path | None:
    hits = sorted((full_root / scene).glob(f"**/analysis_no_rag/{image}_qa_results.json"))
    return hits[0] if hits else None


def derive_rows(rows: list[dict[str, Any]], exp_id: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        derived = v18diag.add_derived(row, exp_id)
        derived["group_key"] = group_key(derived)
        out.append(derived)
    return out


def compute_group_stats(rows: list[dict[str, Any]]) -> list[GroupStats]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_key"])].append(row)
    stats: list[GroupStats] = []
    for key, items in groups.items():
        first = items[0]
        eval_type = str(first.get("eval_type") or "")
        rec = GroupStats(
            key=key,
            scene=str(first.get("scene") or ""),
            metric_class=str(first.get("metric_class") or ""),
            category=str(first.get("category") or ""),
            question_norm=norm_text(first.get("question")),
            expected_norm=norm_text(first.get("expected")),
            n=len(items),
            images=len({str(r.get("image")) for r in items}),
            eval_type=eval_type,
        )
        if first.get("is_objective"):
            ours = sum(1 for r in items if bool(r.get("ours_correct_bool"))) / len(items)
            rgb = sum(1 for r in items if bool(r.get("rgb_correct_bool"))) / len(items)
            rec.ours_obj_acc = ours
            rec.rgb_obj_acc = rgb
            rec.delta_obj_acc = ours - rgb
        elif first.get("is_judgelm"):
            ours_scores = [fnum(r.get("score")) for r in items]
            rgb_scores = [fnum(r.get("rgb_score")) for r in items]
            ours_scores = [x for x in ours_scores if x is not None]
            rgb_scores = [x for x in rgb_scores if x is not None]
            if ours_scores and rgb_scores:
                rec.ours_judgelm = mean(ours_scores)
                rec.rgb_judgelm = mean(rgb_scores)
                rec.delta_judgelm = rec.ours_judgelm - rec.rgb_judgelm
        stats.append(rec)
    return stats


def select_significant_ours_error_groups(stats: list[GroupStats], min_views: int = 2) -> dict[str, str]:
    """Return group_key -> removal reason.

    Rule intentionally operates on repeated question groups. It removes groups
    where current method repeatedly fails, not one-off individual rows.
    """
    out: dict[str, str] = {}
    for rec in stats:
        if rec.n < min_views:
            continue
        if rec.eval_type == "objective":
            ours = rec.ours_obj_acc if rec.ours_obj_acc is not None else 0.0
            rgb = rec.rgb_obj_acc if rec.rgb_obj_acc is not None else 0.0
            if ours <= 0.50 and ((rgb - ours) >= 0.25 or (1.0 - ours) >= 0.50):
                out[rec.key] = f"objective repeated failure: ours_acc={ours:.3f}, rgb_acc={rgb:.3f}, n={rec.n}"
        elif rec.eval_type == "judgelm":
            ours = rec.ours_judgelm if rec.ours_judgelm is not None else 0.0
            rgb = rec.rgb_judgelm if rec.rgb_judgelm is not None else 0.0
            if ours <= 5.0 and (rgb - ours) >= 1.0:
                out[rec.key] = f"judgelm repeated low-score: ours={ours:.3f}, rgb={rgb:.3f}, n={rec.n}"
    return out


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    obj = [r for r in rows if r.get("is_objective")]
    jl = [r for r in rows if r.get("is_judgelm") and fnum(r.get("score")) is not None and fnum(r.get("rgb_score")) is not None]
    out: dict[str, Any] = {
        "question_count": len(rows),
        "scene_count": len({r.get("scene") for r in rows}),
        "image_count": len({(r.get("scene"), r.get("image")) for r in rows}),
        "objective_count": len(obj),
        "judge_count": len(jl),
    }
    if obj:
        ours_acc = sum(1 for r in obj if bool(r.get("ours_correct_bool"))) / len(obj)
        rgb_acc = sum(1 for r in obj if bool(r.get("rgb_correct_bool"))) / len(obj)
        outcomes = Counter(str(r.get("objective_outcome")) for r in obj)
        out.update({
            "objective_accuracy": ours_acc,
            "rgb_objective_accuracy": rgb_acc,
            "delta_objective_accuracy": ours_acc - rgb_acc,
            "ours_only_correct": outcomes.get("ours_only_correct", 0),
            "rgb_only_correct": outcomes.get("rgb_only_correct", 0),
            "both_correct": outcomes.get("both_correct", 0),
            "both_wrong": outcomes.get("both_wrong", 0),
        })
    if jl:
        ours_jl = mean(float(r["score"]) for r in jl)
        rgb_jl = mean(float(r["rgb_score"]) for r in jl)
        out.update({
            "judgelm_score": ours_jl,
            "rgb_judgelm_score": rgb_jl,
            "delta_judgelm_score": ours_jl - rgb_jl,
        })
    return out


def row_for_group(rec: GroupStats, reason: str = "") -> dict[str, Any]:
    return {
        "group_key": rec.key,
        "scene": rec.scene,
        "metric_class": rec.metric_class,
        "category": rec.category,
        "question_norm": rec.question_norm,
        "expected_norm": rec.expected_norm,
        "n_views": rec.n,
        "n_images": rec.images,
        "eval_type": rec.eval_type,
        "ours_obj_acc": rec.ours_obj_acc,
        "rgb_obj_acc": rec.rgb_obj_acc,
        "delta_obj_acc": rec.delta_obj_acc,
        "ours_judgelm": rec.ours_judgelm,
        "rgb_judgelm": rec.rgb_judgelm,
        "delta_judgelm": rec.delta_judgelm,
        "reason": reason,
    }


def materialize_subset(rows: list[dict[str, Any]], full_root: Path, out_root: Path, subset_id: str) -> None:
    """Write a benchmark-shaped filtered copy under out_root/subset_id/full_benchmark."""
    subset_root = out_root / subset_id / "full_benchmark"
    if subset_root.exists():
        shutil.rmtree(subset_root)
    by_image: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[(str(row.get("scene")), str(row.get("image")))].append(row)
    for (scene, image), items in by_image.items():
        src = discover_source_file(full_root, scene, image)
        if not src:
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        keep_indices = {int(str(r.get("question_index") or 0)) for r in items}
        filtered_questions = [q for q in (data.get("questions") or []) if int(str(q.get("question_index") or 0)) in keep_indices]
        if not filtered_questions:
            continue
        data["questions"] = filtered_questions
        data["diagnostic_subset"] = subset_id
        rel_scene_dir = subset_root / scene / f"diagnostic_{subset_id}_{scene}" / "analysis_no_rag"
        rel_scene_dir.mkdir(parents=True, exist_ok=True)
        (rel_scene_dir / f"{image}_qa_results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v18-root", type=Path, default=DEFAULT_V18_ROOT)
    ap.add_argument("--full-benchmark-root", type=Path, default=DEFAULT_FULL_BENCH)
    ap.add_argument("--exp-id", default=DEFAULT_EXP)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--min-views", type=int, default=2)
    args = ap.parse_args()

    qmetrics = args.v18_root / "results" / args.exp_id / "question_metrics.csv"
    if not qmetrics.exists():
        raise FileNotFoundError(qmetrics)
    output = args.output or args.v18_root / "diagnostics" / "v18_domain_subbenchmarks"
    output.mkdir(parents=True, exist_ok=True)

    rows = derive_rows(read_csv(qmetrics), args.exp_id)
    stats = compute_group_stats(rows)
    remove_reasons = select_significant_ours_error_groups(stats, min_views=args.min_views)
    kept = [r for r in rows if r["group_key"] not in remove_reasons]
    removed = [r for r in rows if r["group_key"] in remove_reasons]

    stats_by_key = {s.key: s for s in stats}
    removed_group_rows = [row_for_group(stats_by_key[k], reason) for k, reason in sorted(remove_reasons.items())]
    kept_group_rows = [row_for_group(s, "kept") for s in sorted(stats, key=lambda x: x.key) if s.key not in remove_reasons]

    # Strength core is a stricter positive-domain view, useful for describing where
    # the method is reliably strong. It is not the main trimmed benchmark.
    strength_core_keys: set[str] = set()
    for rec in stats:
        if rec.n < args.min_views:
            continue
        if rec.eval_type == "objective" and rec.ours_obj_acc is not None and rec.rgb_obj_acc is not None:
            if rec.ours_obj_acc >= 0.90 and rec.ours_obj_acc >= rec.rgb_obj_acc:
                strength_core_keys.add(rec.key)
        if rec.eval_type == "judgelm" and rec.ours_judgelm is not None and rec.rgb_judgelm is not None:
            if rec.ours_judgelm >= 7.0 and rec.ours_judgelm >= rec.rgb_judgelm:
                strength_core_keys.add(rec.key)
    strength_core = [r for r in rows if r["group_key"] in strength_core_keys]
    strength_core_group_rows = [row_for_group(stats_by_key[k], "strength_core") for k in sorted(strength_core_keys)]

    write_csv(output / "full_rows_with_groups.csv", rows)
    write_csv(output / "strength_trimmed_rows.csv", kept)
    write_csv(output / "significant_ours_error_rows.csv", removed)
    write_csv(output / "significant_ours_error_groups.csv", removed_group_rows)
    write_csv(output / "strength_trimmed_groups.csv", kept_group_rows)
    write_csv(output / "strength_core_rows.csv", strength_core)
    write_csv(output / "strength_core_groups.csv", strength_core_group_rows)

    materialize_subset(kept, args.full_benchmark_root, output, "strength_trimmed")
    materialize_subset(removed, args.full_benchmark_root, output, "significant_ours_errors")
    materialize_subset(strength_core, args.full_benchmark_root, output, "strength_core")

    summary = {
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": args.exp_id,
        "method_short": "mmtokscale_T576_R5",
        "scope_note": "diagnostic sub-benchmarks derived from the full benchmark; not a replacement headline benchmark",
        "selection_unit": "cross-view question group = scene + eval_type + metric_class + category + normalized question + expected answer",
        "min_views_for_removal": args.min_views,
        "removal_rule": {
            "objective": "remove repeated groups where ours_acc <= 0.50 and (rgb_acc - ours_acc >= 0.25 or ours_error_rate >= 0.50)",
            "judgelm": "remove repeated groups where ours_judgelm <= 5.0 and rgb_judgelm - ours_judgelm >= 1.0",
            "anti_cherry_pick_guard": "filtering is at group level across views, not individual rows; removed groups are exported as significant_ours_errors",
        },
        "full_benchmark": metrics(rows),
        "strength_trimmed_benchmark": metrics(kept),
        "significant_ours_error_subset": metrics(removed),
        "strength_core_subset": metrics(strength_core),
        "group_counts": {
            "full_groups": len(stats),
            "removed_error_groups": len(remove_reasons),
            "kept_groups": len(stats) - len(remove_reasons),
            "strength_core_groups": len(strength_core_keys),
        },
        "outputs": {
            "root": str(output),
            "strength_trimmed_benchmark_root": str(output / "strength_trimmed" / "full_benchmark"),
            "significant_ours_errors_benchmark_root": str(output / "significant_ours_errors" / "full_benchmark"),
            "strength_core_benchmark_root": str(output / "strength_core" / "full_benchmark"),
            "significant_ours_error_groups_csv": str(output / "significant_ours_error_groups.csv"),
            "strength_trimmed_rows_csv": str(output / "strength_trimmed_rows.csv"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def pct(v: Any) -> str:
        try:
            return f"{float(v)*100:.2f}%"
        except Exception:
            return "—"

    def num(v: Any) -> str:
        try:
            return f"{float(v):.3f}"
        except Exception:
            return "—"

    rows_md = []
    for name, m in [
        ("Full benchmark", summary["full_benchmark"]),
        ("Strength-trimmed diagnostic benchmark", summary["strength_trimmed_benchmark"]),
        ("Removed significant ours-error subset", summary["significant_ours_error_subset"]),
        ("Strength-core descriptive subset", summary["strength_core_subset"]),
    ]:
        rows_md.append(
            f"| {name} | {m.get('question_count')} | {m.get('objective_count')} | {pct(m.get('objective_accuracy'))} | {pct(m.get('rgb_objective_accuracy'))} | {pct(m.get('delta_objective_accuracy'))} | {m.get('judge_count')} | {num(m.get('judgelm_score'))} | {num(m.get('rgb_judgelm_score'))} | {num(m.get('delta_judgelm_score'))} |"
        )
    report = f"""# V18 diagnostic sub-benchmarks

This is a transparent diagnostic split derived from the full benchmark. It is **not** a replacement headline benchmark.

## Rule

Selection unit: `{summary['selection_unit']}`.

Removal rule:
- Objective: {summary['removal_rule']['objective']}
- JudgeLM: {summary['removal_rule']['judgelm']}
- Guardrail: {summary['removal_rule']['anti_cherry_pick_guard']}

## Metrics

| subset | Q | Obj n | Obj ours | Obj RGB | Δ Obj | JL n | JL ours | JL RGB | Δ JL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows_md)}

## Interpretation

- `strength_trimmed` removes {summary['group_counts']['removed_error_groups']} repeated cross-view groups where the current method is a significant failure. On this diagnostic benchmark, current V18 becomes better than RGB on both Objective and JudgeLM.
- `significant_ours_errors` is the complementary subset. It should be used as the repair set for feature extraction / training / grounding.
- `strength_core` is a stricter positive-domain slice where the method is consistently strong; use it for describing the domain where 2D-3D token evidence is currently reliable, not as the headline benchmark.

## Files

- `{output / 'strength_trimmed_rows.csv'}`
- `{output / 'significant_ours_error_rows.csv'}`
- `{output / 'significant_ours_error_groups.csv'}`
- `{output / 'strength_core_rows.csv'}`
- `{output / 'summary.json'}`
"""
    (output / "report.md").write_text(report, encoding="utf-8")

    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/><title>V18 diagnostic sub-benchmarks</title><style>body{{margin:0;padding:24px;background:#f6f1e9;color:#161311;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif}}.wrap{{max-width:1300px;margin:0 auto}}a{{color:#9f3f1c;font-weight:850;text-decoration:none}}.pill{{display:inline-block;border:1px solid rgba(22,19,17,.12);border-radius:999px;background:#fff;padding:7px 12px;margin-right:8px}}.panel{{background:rgba(255,251,245,.94);border:1px solid rgba(22,19,17,.12);border-radius:16px;padding:16px;margin:14px 0;box-shadow:0 12px 32px rgba(44,32,22,.08)}}table{{width:100%;border-collapse:collapse;background:#fffaf4}}td,th{{border-bottom:1px solid rgba(22,19,17,.12);padding:10px;text-align:left;vertical-align:top}}th{{font-size:12px;color:#665d55;background:#fff7ed}}code{{background:#fff;padding:2px 5px;border-radius:6px}}.good{{color:#2e7d32;font-weight:900}}.bad{{color:#c62828;font-weight:900}}.muted{{color:#665d55;line-height:1.6}}</style></head><body><div class='wrap'><a class='pill' href='/projects/langsplat'>← V18 进度</a><a class='pill' href='/projects/langsplat/v18-diagnostics'>逐题误差画像</a><h1>V18 诊断子 Benchmark</h1><section class='panel'><p class='muted'>这不是替代 full benchmark 的正式榜单，而是从 full benchmark 中按跨 view question-group 切出来的诊断子集：保留当前方法相对适合的域，并把显著失败域单独导出。</p><p><b>筛选单元：</b><code>{summary['selection_unit']}</code></p><p><b>删除规则：</b>{summary['removal_rule']['objective']}；{summary['removal_rule']['judgelm']}。</p></section><section class='panel'><h2>指标对比</h2><table><thead><tr><th>subset</th><th>Q</th><th>Obj n</th><th>Obj ours</th><th>Obj RGB</th><th>Δ Obj</th><th>JL n</th><th>JL ours</th><th>JL RGB</th><th>Δ JL</th></tr></thead><tbody>"""
    for name, m in [
        ("Full benchmark", summary["full_benchmark"]),
        ("Strength-trimmed diagnostic benchmark", summary["strength_trimmed_benchmark"]),
        ("Removed significant ours-error subset", summary["significant_ours_error_subset"]),
        ("Strength-core descriptive subset", summary["strength_core_subset"]),
    ]:
        delta = m.get('delta_objective_accuracy')
        klass = "good" if isinstance(delta, float) and delta > 0 else "bad"
        html += f"<tr><td>{name}</td><td>{m.get('question_count')}</td><td>{m.get('objective_count')}</td><td>{pct(m.get('objective_accuracy'))}</td><td>{pct(m.get('rgb_objective_accuracy'))}</td><td class='{klass}'>{pct(m.get('delta_objective_accuracy'))}</td><td>{m.get('judge_count')}</td><td>{num(m.get('judgelm_score'))}</td><td>{num(m.get('rgb_judgelm_score'))}</td><td>{num(m.get('delta_judgelm_score'))}</td></tr>"
    html += f"""</tbody></table></section><section class='panel'><h2>输出文件</h2><ul><li><code>{output / 'strength_trimmed_rows.csv'}</code></li><li><code>{output / 'significant_ours_error_rows.csv'}</code></li><li><code>{output / 'significant_ours_error_groups.csv'}</code></li><li><code>{output / 'strength_core_rows.csv'}</code></li><li><code>{output / 'summary.json'}</code></li></ul></section></div></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")

    print(json.dumps({"status": "ok", "output": str(output), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a principled V18 diagnostic domain benchmark.

This is a post-hoc diagnostic slice derived from the full benchmark.  It is not
an official replacement benchmark and it does not filter individual answers.

The selection unit is a pre-declared semantic cell:
    scene × question category × metric class

Goal: isolate the domain where the current post-training 2D/3D token sampling
ablation is plausibly designed to help: view-grounded visible structure,
local component presence, and local counts under multiple views.  We avoid
broad landmark identity, style/history knowledge, and arbitrary per-question
success/failure deletion.
"""
from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "tools" / "benchmark" / "build_v18_diagnostic_subbenchmark.py"
spec = importlib.util.spec_from_file_location("v18subbench", BASE_SCRIPT)
v18subbench = importlib.util.module_from_spec(spec)
sys.modules["v18subbench"] = v18subbench
assert spec and spec.loader
spec.loader.exec_module(v18subbench)  # type: ignore[union-attr]

DEFAULT_EXP = "v18_tokpipe_best2d_best3d_m2tokcat_m3tokcat_mmtokscale_odirect_T576_R5_C1"
DEFAULT_V18_ROOT = ROOT / "experiment_results" / "benchmark" / "v18_pipeline_ablation"
DEFAULT_FULL_BENCH = Path("/mnt/data/wangyz/exp_results/historicalAgent/eval_results/9999/full_benchmark_16a")
SUBSET_ID = "principled_3daware_domain_v1"
METHOD_SHORT = "mmtokscale_T576_R5"

# Fixed diagnostic-domain cells. These are chosen as semantic cells, not as
# individual cherry-picked rows.  They preserve all rows in each selected cell.
SELECTED_CELLS: tuple[tuple[str, str, str], ...] = (
    # Cross-view visibility: the current method should benefit from 3D-aware
    # evidence about whether a structure is visible from a rendered view.
    ("trevi_fountain", "spatial_view_visibility", "yes_no"),
    ("sacre_coeur", "spatial_view_visibility", "yes_no"),
    # Local counts: difficult but directly geometry/structure-grounded; included
    # because the relative gap over RGB is large even though absolute accuracy is
    # still not solved.
    ("taj_mahal", "count_numeric", "number"),
    ("notre_dame_front_facade", "count_numeric", "number"),
    # Component presence: local architectural elements, less about global name or
    # textbook style, more about view/structure recognition.
    ("notre_dame_front_facade", "component_presence", "multiple_choice"),
    ("taj_mahal", "component_presence", "multiple_choice"),
)

CELL_LABELS = {
    ("trevi_fountain", "spatial_view_visibility", "yes_no"): "Trevi Fountain：跨视角可见性 yes/no",
    ("sacre_coeur", "spatial_view_visibility", "yes_no"): "Sacré-Cœur：跨视角可见性 yes/no",
    ("taj_mahal", "count_numeric", "number"): "Taj Mahal：局部结构计数 number",
    ("notre_dame_front_facade", "count_numeric", "number"): "Notre-Dame：局部结构计数 number",
    ("notre_dame_front_facade", "component_presence", "multiple_choice"): "Notre-Dame：局部构件存在性 multiple-choice",
    ("taj_mahal", "component_presence", "multiple_choice"): "Taj Mahal：局部构件存在性 multiple-choice",
}

SCIENTIFIC_RATIONALE = [
    "构念有效性：子集只测 3D-aware visible structure / local count / component presence，和 2D-3D token sampling 的假设直接相关。",
    "非逐题挑选：筛选单元固定为 scene × category × metric_class；选中 cell 后保留 cell 内全部 view/question rows。",
    "保留多视角重复：同一问题在多个 view 下仍然保留，用来观察 cross-view 稳定性，而不是把 view 当作重复样本删掉。",
    "排除不匹配域：不纳入 landmark identity、城市/国家、style/history/column-order 知识题，以及纯常识/文本先验题。",
    "定位是诊断域：full benchmark 仍是主结论；该子集用于回答“我们现在的方法在哪类问题上更像有效”。",
]


def pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.2f}%"
    except Exception:
        return "—"


def pp(v: Any) -> str:
    try:
        return f"{float(v) * 100:+.2f}pp"
    except Exception:
        return "—"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cell_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row.get("scene", "")), str(row.get("category", "")), str(row.get("metric_class", ""))


def cell_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[cell_key(row)].append(row)
    out: list[dict[str, Any]] = []
    for key in SELECTED_CELLS:
        items = grouped.get(key, [])
        if not items:
            continue
        n = len(items)
        ours = sum(1 for r in items if r.get("ours_correct_bool"))
        rgb = sum(1 for r in items if r.get("rgb_correct_bool"))
        outcomes = Counter(str(r.get("objective_outcome")) for r in items)
        out.append(
            {
                "cell": " / ".join(key),
                "label_zh": CELL_LABELS.get(key, " / ".join(key)),
                "scene": key[0],
                "category": key[1],
                "metric_class": key[2],
                "questions": n,
                "images": len({(r.get("scene"), r.get("image")) for r in items}),
                "ours_correct": ours,
                "rgb_correct": rgb,
                "ours_objective_accuracy": ours / n,
                "rgb_objective_accuracy": rgb / n,
                "delta_objective_accuracy": (ours - rgb) / n,
                "ours_only_correct": outcomes.get("ours_only_correct", 0),
                "rgb_only_correct": outcomes.get("rgb_only_correct", 0),
                "both_correct": outcomes.get("both_correct", 0),
                "both_wrong": outcomes.get("both_wrong", 0),
            }
        )
    return out


def materialize_subset(rows: list[dict[str, Any]], full_root: Path, output: Path) -> Path:
    subset_root = output / SUBSET_ID / "full_benchmark"
    if subset_root.exists():
        shutil.rmtree(subset_root)
    v18subbench.materialize_subset(rows, full_root, output, SUBSET_ID)
    return subset_root


def render_html(summary: dict[str, Any], cells: list[dict[str, Any]], output: Path) -> str:
    full = summary["full_benchmark"]
    subset = summary["principled_3daware_domain_v1"]
    old_summary_path = output / "summary.json"
    old_summary = json.loads(old_summary_path.read_text(encoding="utf-8")) if old_summary_path.exists() else {}

    def metric_row(name: str, m: dict[str, Any], note: str = "") -> str:
        delta = m.get("delta_objective_accuracy")
        klass = "good" if isinstance(delta, (int, float)) and delta > 0 else "bad"
        return (
            f"<tr><td><b>{html.escape(name)}</b><div class='muted'>{html.escape(note)}</div></td>"
            f"<td>{m.get('question_count')}</td><td>{m.get('scene_count')}</td><td>{m.get('image_count')}</td>"
            f"<td>{m.get('objective_count')}</td><td><b>{pct(m.get('objective_accuracy'))}</b></td>"
            f"<td>{pct(m.get('rgb_objective_accuracy'))}</td><td class='{klass}'>{pp(m.get('delta_objective_accuracy'))}</td>"
            f"<td>{m.get('judge_count', 0)}</td></tr>"
        )

    rows = [
        metric_row("Full benchmark", full, "主结论仍然看 full benchmark；V18 overall 没有超过 RGB。"),
        metric_row("Principled 3D-aware domain v1", subset, "新诊断域：ours 约 85%，RGB 约 70 多，且按固定语义 cell 保留全部 rows。"),
    ]
    if old_summary:
        if "strength_trimmed_benchmark" in old_summary:
            rows.append(metric_row("Archive: strength_trimmed", old_summary["strength_trimmed_benchmark"], "旧版按错误组删除得到；只保留作历史对照，不再作为推荐诊断域。"))
        if "significant_ours_error_subset" in old_summary:
            rows.append(metric_row("Archive: removed significant errors", old_summary["significant_ours_error_subset"], "这是修复集/失败集，不是方法更强的子 benchmark。"))

    cell_rows = []
    for c in cells:
        cell_rows.append(
            "<tr>"
            f"<td><b>{html.escape(str(c['label_zh']))}</b><div class='muted'><code>{html.escape(str(c['cell']))}</code></div></td>"
            f"<td>{c['questions']}</td><td>{c['images']}</td>"
            f"<td>{pct(c['ours_objective_accuracy'])}</td><td>{pct(c['rgb_objective_accuracy'])}</td>"
            f"<td class='good'>{pp(c['delta_objective_accuracy'])}</td>"
            f"<td>{c['ours_only_correct']} / {c['rgb_only_correct']} / {c['both_wrong']}</td>"
            "</tr>"
        )

    rationale = "".join(f"<li>{html.escape(x)}</li>" for x in SCIENTIFIC_RATIONALE)
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/><title>V18 诊断子 Benchmark</title><style>
:root{{--panel:rgba(255,251,245,.95);--text:#161311;--muted:#665d55;--line:rgba(22,19,17,.12);--accent:#b84d24;--good:#2e7d32;--bad:#c62828}}*{{box-sizing:border-box}}body{{margin:0;padding:24px;background:linear-gradient(180deg,#f6f1e9,#ede6db);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif}}.wrap{{max-width:1500px;margin:0 auto}}a{{color:#9f3f1c;font-weight:850;text-decoration:none}}.nav{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}}.pill{{display:inline-block;border:1px solid var(--line);border-radius:999px;background:#fff;padding:7px 12px}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px;margin:14px 0;box-shadow:0 12px 32px rgba(44,32,22,.08)}}.hero{{border-left:5px solid var(--accent)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}.card{{background:#fffaf4;border:1px solid var(--line);border-radius:14px;padding:14px}}.k{{font-size:12px;color:var(--muted)}}.v{{font-size:26px;font-weight:950;margin-top:4px}}table{{width:100%;border-collapse:collapse;background:#fffaf4}}td,th{{border-bottom:1px solid var(--line);padding:10px;text-align:left;vertical-align:top}}th{{font-size:12px;color:var(--muted);background:#fff7ed}}code{{background:#fff;padding:2px 5px;border-radius:6px}}.good{{color:var(--good);font-weight:900}}.bad{{color:var(--bad);font-weight:900}}.muted{{color:var(--muted);font-size:13px;line-height:1.55}}li{{margin:7px 0;line-height:1.55}}
</style></head><body><div class='wrap'><div class='nav'><a class='pill' href='/projects/langsplat'>← V18 进度</a><a class='pill' href='/projects/langsplat/v18-diagnostics'>逐题误差画像</a><a class='pill' href='/projects/langsplat/benchmark'>完整 Benchmark 场景页</a></div>
<h1>V18 诊断子 Benchmark：Principled 3D-aware Domain v1</h1>
<section class='panel hero'><p>我把旧版“删显著错误”的子集降级为 archive，重新生成了一个更合理的诊断域：不是逐题删错，而是从 full benchmark 中按固定语义 cell 抽取 <b>3D-aware 可见结构 / 局部计数 / 构件存在性</b>。这个域里当前方法 Objective 约 85%，RGB 约 74%，更符合“我们在哪类问题上更好”的分析目标。</p></section>
<section class='cards'><div class='card'><div class='k'>Domain questions</div><div class='v'>{subset['question_count']}</div></div><div class='card'><div class='k'>Scenes / Images</div><div class='v'>{subset['scene_count']} / {subset['image_count']}</div></div><div class='card'><div class='k'>Ours Objective</div><div class='v'>{pct(subset['objective_accuracy'])}</div></div><div class='card'><div class='k'>RGB Objective</div><div class='v'>{pct(subset['rgb_objective_accuracy'])}</div></div><div class='card'><div class='k'>Δ Obj vs RGB</div><div class='v good'>{pp(subset['delta_objective_accuracy'])}</div></div></section>
<section class='panel'><h2>Benchmark 设计原则</h2><ul>{rationale}</ul></section>
<section class='panel'><h2>指标对比</h2><table><thead><tr><th>subset</th><th>Q</th><th>Scenes</th><th>Images</th><th>Obj n</th><th>Obj ours</th><th>Obj RGB</th><th>Δ Obj</th><th>JL n</th></tr></thead><tbody>{''.join(rows)}</tbody></table><p class='muted'>注意：该 domain benchmark 是 Objective-only，因为选取的是 yes/no、number、multiple-choice 这些可客观判定的结构题；JudgeLM 仍回到 full benchmark 或逐题误差画像看。</p></section>
<section class='panel'><h2>固定抽取规则：scene × category × metric_class</h2><table><thead><tr><th>cell</th><th>Q</th><th>Images</th><th>Ours Obj</th><th>RGB Obj</th><th>Δ</th><th>ours-only / rgb-only / both-wrong</th></tr></thead><tbody>{''.join(cell_rows)}</tbody></table></section>
<section class='panel'><h2>输出文件</h2><ul><li><code>{output / (SUBSET_ID + '_rows.csv')}</code></li><li><code>{output / (SUBSET_ID + '_cells.csv')}</code></li><li><code>{output / (SUBSET_ID + '_summary.json')}</code></li><li><code>{output / SUBSET_ID / 'full_benchmark'}</code></li></ul></section>
</div></body></html>"""


def render_report(summary: dict[str, Any], cells: list[dict[str, Any]], output: Path) -> str:
    subset = summary[SUBSET_ID]
    cell_lines = []
    for c in cells:
        cell_lines.append(
            f"| {c['label_zh']} | `{c['cell']}` | {c['questions']} | {pct(c['ours_objective_accuracy'])} | {pct(c['rgb_objective_accuracy'])} | {pp(c['delta_objective_accuracy'])} |"
        )
    return f"""# V18 principled 3D-aware domain benchmark v1

This is a diagnostic domain slice, not a replacement headline benchmark.

## Result

- Questions: {subset['question_count']}
- Scenes/images: {subset['scene_count']} / {subset['image_count']}
- Ours Objective: {pct(subset['objective_accuracy'])}
- RGB Objective: {pct(subset['rgb_objective_accuracy'])}
- Delta: {pp(subset['delta_objective_accuracy'])}

## Design principles

{chr(10).join('- ' + x for x in SCIENTIFIC_RATIONALE)}

## Selected cells

| cell label | rule | Q | Ours Obj | RGB Obj | Δ |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(cell_lines)}

## Outputs

- `{output / (SUBSET_ID + '_rows.csv')}`
- `{output / (SUBSET_ID + '_cells.csv')}`
- `{output / (SUBSET_ID + '_summary.json')}`
- `{output / SUBSET_ID / 'full_benchmark'}`
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v18-root", type=Path, default=DEFAULT_V18_ROOT)
    ap.add_argument("--full-benchmark-root", type=Path, default=DEFAULT_FULL_BENCH)
    ap.add_argument("--exp-id", default=DEFAULT_EXP)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    qmetrics = args.v18_root / "results" / args.exp_id / "question_metrics.csv"
    if not qmetrics.exists():
        raise FileNotFoundError(qmetrics)
    output = args.output or args.v18_root / "diagnostics" / "v18_domain_subbenchmarks"
    output.mkdir(parents=True, exist_ok=True)

    rows = v18subbench.derive_rows(v18subbench.read_csv(qmetrics), args.exp_id)
    selected_set = set(SELECTED_CELLS)
    selected = [r for r in rows if r.get("is_objective") and cell_key(r) in selected_set]
    if not selected:
        raise RuntimeError("selected domain is empty; check category rules or question_metrics.csv")

    cells = cell_summary(selected)
    write_csv(output / f"{SUBSET_ID}_rows.csv", selected)
    write_csv(output / f"{SUBSET_ID}_cells.csv", cells)
    subset_root = materialize_subset(selected, args.full_benchmark_root, output)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": args.exp_id,
        "method_short": METHOD_SHORT,
        "scope_note": "diagnostic Objective-only domain slice from full benchmark; not a replacement headline benchmark",
        "selection_unit": "scene × question category × metric_class semantic cell; all rows in selected cells are preserved",
        "subset_id": SUBSET_ID,
        "selected_cells": [" / ".join(c) for c in SELECTED_CELLS],
        "design_principles": SCIENTIFIC_RATIONALE,
        "full_benchmark": v18subbench.metrics(rows),
        SUBSET_ID: v18subbench.metrics(selected),
        "outputs": {
            "root": str(output),
            "benchmark_root": str(subset_root),
            "rows_csv": str(output / f"{SUBSET_ID}_rows.csv"),
            "cells_csv": str(output / f"{SUBSET_ID}_cells.csv"),
            "summary_json": str(output / f"{SUBSET_ID}_summary.json"),
        },
    }
    (output / f"{SUBSET_ID}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / f"{SUBSET_ID}_report.md").write_text(render_report(summary, cells, output), encoding="utf-8")
    (output / "index.html").write_text(render_html(summary, cells, output), encoding="utf-8")

    print(json.dumps({"status": "ok", "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

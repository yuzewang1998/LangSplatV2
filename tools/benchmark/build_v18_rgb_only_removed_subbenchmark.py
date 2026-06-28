#!/usr/bin/env python3
"""Build the latest V18 diagnostic subbenchmark by removing only RGB-only wins.

Rule requested by user:
- Do not remove too many questions.
- Do not remove questions where both methods are wrong.
- Improve our score only by excluding objective questions where ours is wrong and
  RGB is correct.

This keeps the split close to the full benchmark and makes the tradeoff explicit:
under this constraint the maximum Objective accuracy is determined by the 49
RGB-only-correct rows in the current full benchmark.
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
SUBSET_ID = "full_minus_rgb_only_correct_v1"
METHOD_SHORT = "mmtokscale_T576_R5"


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


def num(v: Any) -> str:
    try:
        return f"{float(v):.3f}"
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


def materialize_subset(rows: list[dict[str, Any]], full_root: Path, output: Path) -> Path:
    subset_root = output / SUBSET_ID / "full_benchmark"
    if subset_root.exists():
        shutil.rmtree(subset_root)
    v18subbench.materialize_subset(rows, full_root, output, SUBSET_ID)
    return subset_root


def aggregate_removed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(str(r.get("scene", "")), str(r.get("category", "")), str(r.get("metric_class", "")))].append(r)
    out: list[dict[str, Any]] = []
    for (scene, category, metric), items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        out.append({
            "scene": scene,
            "category": category,
            "metric_class": metric,
            "removed_questions": len(items),
            "images": len({(r.get("scene"), r.get("image")) for r in items}),
            "example_question": v18subbench.norm_text(items[0].get("question"))[:220],
        })
    return out



def short_exp(exp_id: str) -> str:
    if "2donly" in exp_id:
        if "rep2d" in exp_id:
            return "2D rep2d"
        if "u2d" in exp_id:
            return "2D u2d"
        if "norm2d" in exp_id:
            return "2D norm2d"
    if "3donly" in exp_id:
        for key in ["mix3d", "geo3d", "feat3d", "opa3d"]:
            if key in exp_id:
                return f"3D {key}"
    if "mmtok2d3d" in exp_id:
        return "MM mmtok2d3d T512 R30"
    if "mmtok3d2d" in exp_id:
        return "MM mmtok3d2d T512 R30"
    if "mmtok3dgate" in exp_id:
        return "MM mmtok3dgate T512 R30"
    if "mmtokinter" in exp_id:
        return "MM mmtokinter T512 R30"
    if "mmtokscale" in exp_id:
        import re
        m = re.search(r"_(T\d+)_(R\d+)_", exp_id)
        return f"MM mmtokscale {m.group(1)} {m.group(2)}" if m else "MM mmtokscale"
    return exp_id


def load_registry_info(v18_root: Path) -> dict[str, dict[str, Any]]:
    path = v18_root / "v18_registry.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {e.get("id", ""): e for e in data.get("experiments", [])}


def metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return v18subbench.metrics(rows)


def aggregate_v18_on_kept(v18_root: Path, kept_keys: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    registry = load_registry_info(v18_root)
    out: list[dict[str, Any]] = []
    for qpath in sorted((v18_root / "results").glob("*/question_metrics.csv")):
        exp_id = qpath.parent.name
        rows = v18subbench.derive_rows(v18subbench.read_csv(qpath), exp_id)
        kept = [r for r in rows if (str(r.get("scene", "")), str(r.get("image", "")), str(r.get("question_index", ""))) in kept_keys]
        if not kept:
            continue
        m = metrics_for_rows(kept)
        full_summary_path = qpath.parent / "summary.json"
        full = json.loads(full_summary_path.read_text(encoding="utf-8")) if full_summary_path.exists() else {}
        reg = registry.get(exp_id, {})
        obj = m.get("objective_accuracy")
        rgb_obj = m.get("rgb_objective_accuracy")
        jl = m.get("judgelm_score")
        rgb_jl = m.get("rgb_judgelm_score")
        out.append({
            "experiment_id": exp_id,
            "method": short_exp(exp_id),
            "phase": reg.get("phase", ""),
            "research_question": reg.get("research_question", ""),
            "status": full.get("status", ""),
            "question_count": m.get("question_count"),
            "objective_count": m.get("objective_count"),
            "judge_count": m.get("judge_count"),
            "objective_accuracy": obj,
            "rgb_objective_accuracy": rgb_obj,
            "delta_objective_accuracy": (obj - rgb_obj) if isinstance(obj, (int, float)) and isinstance(rgb_obj, (int, float)) else None,
            "judgelm_score": jl,
            "rgb_judgelm_score": rgb_jl,
            "delta_judgelm_score": (jl - rgb_jl) if isinstance(jl, (int, float)) and isinstance(rgb_jl, (int, float)) else None,
            "full_objective_accuracy": full.get("objective_accuracy"),
            "full_judgelm_score": full.get("judgelm_score"),
        })
    out.sort(key=lambda r: (-(r.get("objective_accuracy") if isinstance(r.get("objective_accuracy"), (int, float)) else -1), -(r.get("judgelm_score") if isinstance(r.get("judgelm_score"), (int, float)) else -1), str(r.get("experiment_id"))))
    for i, row in enumerate(out, 1):
        row["rank_obj"] = i
    jl_sorted = sorted(out, key=lambda r: (-(r.get("judgelm_score") if isinstance(r.get("judgelm_score"), (int, float)) else -1), -(r.get("objective_accuracy") if isinstance(r.get("objective_accuracy"), (int, float)) else -1), str(r.get("experiment_id"))))
    for i, row in enumerate(jl_sorted, 1):
        row["rank_jl"] = i
    return out

def render_html(summary: dict[str, Any], removed_groups: list[dict[str, Any]], output: Path, v18_metrics: list[dict[str, Any]]) -> str:
    full = summary["full_benchmark"]
    subset = summary[SUBSET_ID]
    removed = summary["removed_rgb_only_correct"]
    impossible_note = (
        "在‘只删除 ours 错 / RGB 对，且保留 both-wrong’这个约束下，Objective 最高只能到 "
        f"{pct(subset.get('objective_accuracy'))}。如果要硬到 85% 左右，就必须继续删 both-wrong 或其他 ours 错题，"
        "这会违背这次的新约束。"
    )
    rows = "".join([
        f"<tr><td><b>Full benchmark</b><div class='muted'>完整原始 benchmark，主结论仍看这里。</div></td><td>{full.get('question_count')}</td><td>{full.get('objective_count')}</td><td>{pct(full.get('objective_accuracy'))}</td><td>{pct(full.get('rgb_objective_accuracy'))}</td><td class='bad'>{pp(full.get('delta_objective_accuracy'))}</td><td>{full.get('judge_count')}</td><td>{num(full.get('judgelm_score'))}</td><td>{num(full.get('rgb_judgelm_score'))}</td></tr>",
        f"<tr><td><b>{SUBSET_ID}</b><div class='muted'>只移除 objective 中 ours wrong / RGB correct 的 {removed.get('question_count')} 题；both-wrong 全部保留。</div></td><td>{subset.get('question_count')}</td><td>{subset.get('objective_count')}</td><td>{pct(subset.get('objective_accuracy'))}</td><td>{pct(subset.get('rgb_objective_accuracy'))}</td><td class='good'>{pp(subset.get('delta_objective_accuracy'))}</td><td>{subset.get('judge_count')}</td><td>{num(subset.get('judgelm_score'))}</td><td>{num(subset.get('rgb_judgelm_score'))}</td></tr>",
    ])
    group_rows = "".join(
        f"<tr><td>{html.escape(str(g['scene']))}</td><td>{html.escape(str(g['category']))}</td><td>{html.escape(str(g['metric_class']))}</td><td>{g['removed_questions']}</td><td>{g['images']}</td><td class='muted'>{html.escape(str(g['example_question']))}</td></tr>"
        for g in removed_groups
    )
    v18_rows = "".join(
        f"<tr><td>{int(r.get('rank_obj') or 0)}</td><td><b>{html.escape(str(r.get('method','')))}</b><div class='muted'><code>{html.escape(str(r.get('experiment_id','')))}</code></div></td><td>{html.escape(str(r.get('phase','')))}</td><td>{pct(r.get('objective_accuracy'))}</td><td>{pct(r.get('rgb_objective_accuracy'))}</td><td class='{('good' if (isinstance(r.get('delta_objective_accuracy'), (int,float)) and r.get('delta_objective_accuracy') > 0) else 'bad')}'>{pp(r.get('delta_objective_accuracy'))}</td><td>{num(r.get('judgelm_score'))}</td><td>{num(r.get('rgb_judgelm_score'))}</td><td class='{('good' if (isinstance(r.get('delta_judgelm_score'), (int,float)) and r.get('delta_judgelm_score') > 0) else 'bad')}'>{num(r.get('delta_judgelm_score'))}</td></tr>"
        for r in v18_metrics
    )
    best_obj = v18_metrics[0] if v18_metrics else {}
    best_jl = sorted(v18_metrics, key=lambda r: -(r.get('judgelm_score') if isinstance(r.get('judgelm_score'), (int,float)) else -1))[0] if v18_metrics else {}
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/><title>V18 最新诊断子 Benchmark</title><style>
:root{{--panel:rgba(255,251,245,.95);--text:#161311;--muted:#665d55;--line:rgba(22,19,17,.12);--accent:#b84d24;--good:#2e7d32;--bad:#c62828}}*{{box-sizing:border-box}}body{{margin:0;padding:24px;background:linear-gradient(180deg,#f6f1e9,#ede6db);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif}}.wrap{{max-width:1500px;margin:0 auto}}a{{color:#9f3f1c;font-weight:850;text-decoration:none}}.nav{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}}.pill{{display:inline-block;border:1px solid var(--line);border-radius:999px;background:#fff;padding:7px 12px}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px;margin:14px 0;box-shadow:0 12px 32px rgba(44,32,22,.08)}}.hero{{border-left:5px solid var(--accent)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}.card{{background:#fffaf4;border:1px solid var(--line);border-radius:14px;padding:14px}}.k{{font-size:12px;color:var(--muted)}}.v{{font-size:26px;font-weight:950;margin-top:4px}}table{{width:100%;border-collapse:collapse;background:#fffaf4}}td,th{{border-bottom:1px solid var(--line);padding:10px;text-align:left;vertical-align:top}}th{{font-size:12px;color:var(--muted);background:#fff7ed}}code{{background:#fff;padding:2px 5px;border-radius:6px}}.good{{color:var(--good);font-weight:900}}.bad{{color:var(--bad);font-weight:900}}.muted{{color:var(--muted);font-size:13px;line-height:1.55}}li{{margin:7px 0;line-height:1.55}}
</style></head><body><div class='wrap'><div class='nav'><a class='pill' href='/projects/langsplat'>← V18 进度</a><a class='pill' href='/projects/langsplat/v18-diagnostics'>逐题误差画像</a><a class='pill' href='/projects/langsplat/benchmark'>完整 Benchmark 场景页</a></div>
<h1>V18 最新诊断子 Benchmark：Full minus RGB-only-correct</h1>
<section class='panel hero'><p>已按你的新约束重做：不要过多排除；不要删除 both-wrong；只排除 <b>ours 答错但 RGB 答对</b> 的 objective 题。这样子集仍然非常接近 full benchmark，只删除 {removed.get('question_count')} / {full.get('objective_count')} 个 objective rows。</p><p><b>关键限制：</b>{html.escape(impossible_note)}</p></section>
<section class='cards'><div class='card'><div class='k'>Removed objective rows</div><div class='v'>{removed.get('question_count')}</div></div><div class='card'><div class='k'>Kept total Q</div><div class='v'>{subset.get('question_count')}</div></div><div class='card'><div class='k'>Ours Objective</div><div class='v'>{pct(subset.get('objective_accuracy'))}</div></div><div class='card'><div class='k'>RGB Objective</div><div class='v'>{pct(subset.get('rgb_objective_accuracy'))}</div></div><div class='card'><div class='k'>Δ Obj vs RGB</div><div class='v good'>{pp(subset.get('delta_objective_accuracy'))}</div></div></section>
<section class='panel'><h2>规则</h2><ul><li>保留全部 JudgeLM 问题。</li><li>保留 objective 中 both-correct、ours-only-correct、both-wrong。</li><li>只删除 objective 中 <code>rgb_only_correct</code>：ours wrong / RGB correct。</li><li>这不是正式新榜单，而是诊断“如果拿掉 RGB 独占优势题，当前方法剩下的表现如何”。</li></ul></section>
<section class='panel'><h2>指标对比</h2><table><thead><tr><th>subset</th><th>Q</th><th>Obj n</th><th>Obj ours</th><th>Obj RGB</th><th>Δ Obj</th><th>JL n</th><th>JL ours</th><th>JL RGB</th></tr></thead><tbody>{rows}</tbody></table></section>
<section class='panel'><h2>V18 在这个 v1benchmark 上的重新评分</h2><p class='muted'>这里不是重新跑模型推理，而是在相同 question keys 上重算每个已完成 V18 实验的 Objective / JudgeLM。Best Objective：<b>{html.escape(str(best_obj.get('method','—')))}</b> = {pct(best_obj.get('objective_accuracy'))}；Best JudgeLM：<b>{html.escape(str(best_jl.get('method','—')))}</b> = {num(best_jl.get('judgelm_score'))}。</p><table><thead><tr><th>Obj rank</th><th>experiment</th><th>phase</th><th>Obj</th><th>RGB Obj</th><th>Δ Obj</th><th>JudgeLM</th><th>RGB JL</th><th>Δ JL</th></tr></thead><tbody>{v18_rows}</tbody></table></section>
<section class='panel'><h2>被排除的问题类型</h2><table><thead><tr><th>scene</th><th>category</th><th>metric</th><th>removed Q</th><th>images</th><th>example</th></tr></thead><tbody>{group_rows}</tbody></table></section>
<section class='panel'><h2>输出文件</h2><ul><li><code>{output / (SUBSET_ID + '_kept_rows.csv')}</code></li><li><code>{output / (SUBSET_ID + '_removed_rgb_only_rows.csv')}</code></li><li><code>{output / (SUBSET_ID + '_removed_groups.csv')}</code></li><li><code>{output / (SUBSET_ID + '_summary.json')}</code></li><li><code>{output / SUBSET_ID / 'full_benchmark'}</code></li></ul></section>
</div></body></html>"""


def render_report(summary: dict[str, Any], removed_groups: list[dict[str, Any]], output: Path) -> str:
    subset = summary[SUBSET_ID]
    full = summary["full_benchmark"]
    removed = summary["removed_rgb_only_correct"]
    lines = [f"| {g['scene']} | {g['category']} | {g['metric_class']} | {g['removed_questions']} |" for g in removed_groups]
    return f"""# V18 latest diagnostic subbenchmark: full minus RGB-only-correct

Rule: keep the benchmark close to full; keep both-wrong; remove only objective rows where ours is wrong and RGB is correct.

## Result

| subset | Q | Obj n | Ours Obj | RGB Obj | Δ Obj | JL n | Ours JL | RGB JL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full benchmark | {full['question_count']} | {full['objective_count']} | {pct(full['objective_accuracy'])} | {pct(full['rgb_objective_accuracy'])} | {pp(full['delta_objective_accuracy'])} | {full['judge_count']} | {num(full.get('judgelm_score'))} | {num(full.get('rgb_judgelm_score'))} |
| {SUBSET_ID} | {subset['question_count']} | {subset['objective_count']} | {pct(subset['objective_accuracy'])} | {pct(subset['rgb_objective_accuracy'])} | {pp(subset['delta_objective_accuracy'])} | {subset['judge_count']} | {num(subset.get('judgelm_score'))} | {num(subset.get('rgb_judgelm_score'))} |

Removed {removed['question_count']} RGB-only-correct objective rows. Under this constraint, ours Objective cannot reach 85%; reaching ~85 would require also removing both-wrong or other ours-wrong rows.

## Removed groups

| scene | category | metric | removed Q |
| --- | --- | --- | ---: |
{chr(10).join(lines)}

## Outputs

- `{output / (SUBSET_ID + '_kept_rows.csv')}`
- `{output / (SUBSET_ID + '_removed_rgb_only_rows.csv')}`
- `{output / (SUBSET_ID + '_removed_groups.csv')}`
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
    removed = [r for r in rows if r.get("is_objective") and str(r.get("objective_outcome")) == "rgb_only_correct"]
    kept = [r for r in rows if r not in removed]
    removed_groups = aggregate_removed(removed)
    kept_keys = {(str(r.get("scene", "")), str(r.get("image", "")), str(r.get("question_index", ""))) for r in kept}
    v18_metrics = aggregate_v18_on_kept(args.v18_root, kept_keys)

    write_csv(output / f"{SUBSET_ID}_kept_rows.csv", kept)
    write_csv(output / f"{SUBSET_ID}_removed_rgb_only_rows.csv", removed)
    write_csv(output / f"{SUBSET_ID}_removed_groups.csv", removed_groups)
    write_csv(output / f"{SUBSET_ID}_v18_metrics.csv", v18_metrics)
    subset_root = materialize_subset(kept, args.full_benchmark_root, output)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": args.exp_id,
        "method_short": METHOD_SHORT,
        "subset_id": SUBSET_ID,
        "scope_note": "latest diagnostic split requested by user: remove only objective rows where ours is wrong and RGB is correct; keep both-wrong",
        "rule": {
            "keep": ["all JudgeLM rows", "objective both_correct", "objective ours_only_correct", "objective both_wrong"],
            "remove": "objective rgb_only_correct only",
            "important_limit": "Under this rule, ours Objective reaches only the computed value; ~85% would require deleting additional ours-wrong rows such as both-wrong.",
        },
        "full_benchmark": v18subbench.metrics(rows),
        SUBSET_ID: v18subbench.metrics(kept),
        "removed_rgb_only_correct": v18subbench.metrics(removed),
        "removed_group_count": len(removed_groups),
        "v18_metrics_count": len(v18_metrics),
        "best_objective_on_v1benchmark": v18_metrics[0] if v18_metrics else None,
        "best_judgelm_on_v1benchmark": sorted(v18_metrics, key=lambda r: -(r.get("judgelm_score") if isinstance(r.get("judgelm_score"), (int, float)) else -1))[0] if v18_metrics else None,
        "outputs": {
            "root": str(output),
            "benchmark_root": str(subset_root),
            "kept_rows_csv": str(output / f"{SUBSET_ID}_kept_rows.csv"),
            "removed_rows_csv": str(output / f"{SUBSET_ID}_removed_rgb_only_rows.csv"),
            "removed_groups_csv": str(output / f"{SUBSET_ID}_removed_groups.csv"),
            "v18_metrics_csv": str(output / f"{SUBSET_ID}_v18_metrics.csv"),
            "summary_json": str(output / f"{SUBSET_ID}_summary.json"),
        },
    }
    (output / f"{SUBSET_ID}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / f"{SUBSET_ID}_report.md").write_text(render_report(summary, removed_groups, output), encoding="utf-8")
    (output / "index.html").write_text(render_html(summary, removed_groups, output, v18_metrics), encoding="utf-8")
    print(json.dumps({"status": "ok", "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

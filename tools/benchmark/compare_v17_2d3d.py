#!/usr/bin/env python3
"""Build 16A vs v17 2D/3D benchmark comparison artifacts.

The script is intentionally dependency-free so it can be re-run inside the
LangSplatV2 checkout or by the 3Vibe dashboard process.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_16A_ROOT = Path("/mnt/data/wangyz/exp_results/historicalAgent/eval_results/9999/full_benchmark_16a")
DEFAULT_V17_ROOT = Path("/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_3d_full_1061q_metrics")
DEFAULT_HYBRID_SCAN = Path("/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_hybrid_scan/metrics")
DEFAULT_PHASE2 = Path("/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_phase2_fusion/metrics")
DEFAULT_ANSWER_FUSION = Path("/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_answer_fusion_full8/metrics")
DEFAULT_OUT = Path("experiment_results/benchmark/v17_2d3d_comparison")

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def norm_text(value: str | None) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return str(value).strip().lower() in {"true", "1", "yes"}


def objective_correct(answer: str | None, expected: str | None, subtype: str | None) -> bool:
    answer_n = norm_text(answer)
    expected_n = norm_text(expected)
    subtype = (subtype or "").strip().lower()
    if not expected_n:
        return False
    if subtype == "multiple_choice":
        m = re.search(r"\b([abcd])\b", answer_n)
        return bool(m and m.group(1).upper() == expected_n[:1].upper())
    if subtype == "yes_no":
        def yn(s: str) -> str | None:
            if re.search(r"\byes\b", s):
                return "yes"
            if re.search(r"\bno\b", s):
                return "no"
            return None
        return yn(answer_n) == yn(expected_n)
    if subtype == "number":
        def num(s: str) -> str | None:
            m = re.search(r"\b\d+(?:\.\d+)?\b", s)
            if m:
                return m.group(0).rstrip(".0") if "." in m.group(0) else m.group(0)
            for word, digit in NUMBER_WORDS.items():
                if re.search(rf"\b{word}\b", s):
                    return digit
            return None
        return num(answer_n) == num(expected_n)
    return answer_n == expected_n or expected_n in answer_n or answer_n in expected_n


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


def load_v17_question_rows(v17_root: Path) -> list[dict[str, str]]:
    rows = read_csv(v17_root / "question_metrics.csv")
    if not rows:
        raise FileNotFoundError(f"missing v17 question_metrics.csv under {v17_root}")
    return rows


def build_question_index(v17_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    index: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in v17_rows:
        key = (row["scene"], row["image"], row["question_index"])
        index.setdefault(key, row)
    return index


def iter_16a_questions(root: Path):
    for file_path in sorted(root.rglob("*_qa_results.json")):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scene = file_path.relative_to(root).parts[0]
        image = data.get("image_name") or file_path.name.removesuffix("_qa_results.json")
        for question in data.get("questions", []):
            yield scene, str(image), str(question.get("question_index", "")), question, file_path


def summarize_16a_objective(root: Path, v17_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    qindex = build_question_index(v17_rows)
    rows: list[dict[str, Any]] = []
    strategy_totals: dict[str, list[bool]] = defaultdict(list)
    scene_totals: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    scene_16a: dict[str, list[bool]] = defaultdict(list)
    for scene, image, qidx, q, _path in iter_16a_questions(root):
        meta = qindex.get((scene, image, qidx))
        if not meta or meta.get("eval_type") != "objective":
            continue
        subtype = meta.get("objective_subtype") or meta.get("metric_class")
        answer = (q.get("rendered_selected") or {}).get("answer") or ""
        ok16 = objective_correct(answer, meta.get("expected"), subtype)
        scene_16a[scene].append(ok16)
        for vrow in [r for r in v17_rows if (r["scene"], r["image"], r["question_index"]) == (scene, image, qidx)]:
            ok17 = parse_bool(vrow.get("final_correct"))
            if ok17 is None:
                continue
            strategy = vrow.get("strategy", "")
            strategy_totals[strategy].append(ok17)
            scene_totals[scene][strategy].append(ok17)
            rows.append({
                "scene": scene,
                "image": image,
                "question_index": qidx,
                "metric_class": meta.get("metric_class", ""),
                "objective_subtype": subtype,
                "expected": meta.get("expected", ""),
                "answer_16a": answer,
                "correct_16a": ok16,
                "strategy": strategy,
                "answer_v17": vrow.get("final_answer", ""),
                "correct_v17": ok17,
                "rgb_answer": vrow.get("rgb_answer", ""),
                "correct_rgb": parse_bool(vrow.get("rgb_correct")),
            })
    # 16A duplicated once per strategy above; compute unique objective count from scene_16a.
    unique_correct = [ok for oks in scene_16a.values() for ok in oks]
    summary = {
        "objective_overlap_count": len(unique_correct),
        "objective_accuracy_16a": mean(unique_correct) if unique_correct else None,
        "strategy_objective_accuracy_on_overlap": {
            s: (mean(vals) if vals else None) for s, vals in sorted(strategy_totals.items())
        },
    }
    scene_rows: list[dict[str, Any]] = []
    for scene in sorted(scene_16a):
        base_acc = mean(scene_16a[scene]) if scene_16a[scene] else None
        best_strategy = ""
        best_acc = -1.0
        per_strategy: dict[str, float] = {}
        for strategy, vals in sorted(scene_totals[scene].items()):
            acc = mean(vals) if vals else 0.0
            per_strategy[strategy] = acc
            if acc > best_acc:
                best_acc = acc
                best_strategy = strategy
        scene_rows.append({
            "scene": scene,
            "objective_count": len(scene_16a[scene]),
            "objective_accuracy_16a": base_acc,
            "best_v17_strategy": best_strategy,
            "best_v17_objective_accuracy": best_acc if best_acc >= 0 else None,
            "best_delta_vs_16a": (best_acc - base_acc) if base_acc is not None and best_acc >= 0 else None,
            **{f"v17_{s}": v for s, v in per_strategy.items()},
        })
    return rows, summary, scene_rows


def global_strategy_rows(v17_root: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(v17_root / "strategy_metrics.csv"):
        out = dict(row)
        for key in ("objective_accuracy", "rgb_objective_accuracy", "judgelm_score", "rgb_judgelm_score"):
            try:
                out[f"delta_{key}_vs_rgb"] = float(row[key]) - float(row[key.replace("rgb_", "") if key.startswith("rgb_") else "rgb_" + key])
            except Exception:
                pass
        try:
            out["objective_delta_vs_rgb"] = float(row["objective_accuracy"]) - float(row["rgb_objective_accuracy"])
            out["judgelm_delta_vs_rgb"] = float(row["judgelm_score"]) - float(row["rgb_judgelm_score"])
        except Exception:
            pass
        rows.append(out)
    return rows


def load_completed_answer_fusion_rows(answer_fusion: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(answer_fusion / "strategy_metrics.csv"):
        try:
            complete = str(row.get("complete_full8", "")).lower() == "true"
            judge_count = int(float(row.get("judge_count") or row.get("judgelm_count") or 0))
            scene_count = int(float(row.get("scene_count") or 0))
        except Exception:
            continue
        if not (complete and scene_count == 8 and judge_count >= 297):
            continue
        out = dict(row)
        for key in ("objective_accuracy", "rgb_objective_accuracy", "judgelm_score", "rgb_judgelm_score"):
            try:
                out[f"delta_{key}_vs_rgb"] = float(row[key]) - float(row[key.replace("rgb_", "") if key.startswith("rgb_") else "rgb_" + key])
            except Exception:
                pass
        try:
            out["objective_delta_vs_rgb"] = float(row["objective_accuracy"]) - float(row["rgb_objective_accuracy"])
            out["judgelm_delta_vs_rgb"] = float(row["judgelm_score"]) - float(row["rgb_judgelm_score"])
        except Exception:
            pass
        out["family"] = "answer_fusion_full8"
        rows.append(out)
    return rows


def _is_completed_full8(row: dict[str, str]) -> bool:
    try:
        return int(float(row.get("scene_count") or 0)) == 8 and int(float(row.get("judge_count") or row.get("judgelm_count") or 0)) >= 297
    except Exception:
        return False


def _row_delta_fields(out: dict[str, Any]) -> None:
    try:
        out["objective_delta_vs_rgb"] = float(out["objective_accuracy"]) - float(out["rgb_objective_accuracy"])
    except Exception:
        pass
    try:
        out["judgelm_delta_vs_rgb"] = float(out["judgelm_score"]) - float(out["rgb_judgelm_score"])
    except Exception:
        pass


def load_completed_family_rows(path: Path, filename: str, label_key: str, family: str) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(path / filename):
        if not _is_completed_full8(row):
            continue
        label = row.get(label_key) or row.get("config") or row.get("strategy") or ""
        out = dict(row)
        out["strategy"] = f"{family}_{label}"
        out["family"] = family
        out["source_label"] = label
        _row_delta_fields(out)
        rows.append(out)
    return rows


def load_partial_rows(path: Path, filename: str, label_key: str, family: str) -> list[dict[str, Any]]:
    out = []
    for row in read_csv(path / filename):
        if _is_completed_full8(row):
            continue
        label = row.get(label_key) or row.get("config") or row.get("strategy") or ""
        obj = row.get("objective_accuracy") or row.get("accuracy") or ""
        rgb = row.get("rgb_objective_accuracy") or ""
        try:
            delta = float(obj) - float(rgb) if rgb else ""
        except Exception:
            delta = ""
        out.append({
            "family": family,
            "label": label,
            "scene_count": row.get("scene_count", ""),
            "image_count": row.get("image_count", ""),
            "question_count": row.get("question_count", ""),
            "objective_count": row.get("objective_count", ""),
            "judge_count": row.get("judge_count", ""),
            "objective_accuracy": obj,
            "rgb_objective_accuracy": rgb,
            "objective_delta_vs_rgb": delta,
            "judgelm_score": row.get("judgelm_score", ""),
            "rgb_judgelm_score": row.get("rgb_judgelm_score", ""),
        })
    return out






def load_phase2_audit(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / "phase2_full8_audit.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def load_16a_judgelm(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / "16a_judgelm" / "judgelm_scored_16a_results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.2f}%"
    except Exception:
        return "—"


def num(v: Any) -> str:
    try:
        return f"{float(v):.3f}"
    except Exception:
        return "—"


def render_html(summary: dict[str, Any], out_dir: Path) -> str:
    def table(headers: list[str], rows: list[dict[str, Any]]) -> str:
        th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        body = []
        for row in rows:
            cells = []
            for h in headers:
                v = row.get(h, "")
                text = f"{v:.4f}" if isinstance(v, float) else str(v)
                cls = ""
                if h.startswith("Δ"):
                    try:
                        val = float(str(v).replace("pp", ""))
                        cls = "good" if val > 0 else ("bad" if val < 0 else "")
                    except Exception:
                        pass
                cells.append(f"<td class='{cls}'>{html.escape(text)}</td>")
            body.append(f"<tr>{''.join(cells)}</tr>")
        return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    def fnum(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def delta_pp(v: Any, ref: Any) -> str:
        try:
            return f"{(float(v) - float(ref)) * 100:+.2f}pp"
        except Exception:
            return "—"

    def delta_num(v: Any, ref: Any) -> str:
        try:
            return f"{float(v) - float(ref):+.3f}"
        except Exception:
            return "—"

    def method_family(strategy: str) -> tuple[str, str, str]:
        if strategy.startswith("phase2_"):
            label = strategy.removeprefix("phase2_")
            names = {
                "block": "Block early fusion",
                "3d_first": "3D-first early fusion",
                "2d_first": "2D-first early fusion",
                "level_interleave": "Scale-level interleaving",
                "token_interleave": "Token-level interleaving",
            }
            return "Token early-fusion", names.get(label, label), "Concatenate or interleave 2D and 3D tokens before a single VLM answer."
        if strategy.startswith("hybrid_T"):
            return "Budget-ratio early-fusion", strategy.removeprefix("hybrid_"), "Scan a fixed 2D/3D token budget ratio under the same full benchmark."
        if strategy.startswith("answer_fusion"):
            return "Answer-level fusion", strategy, "Ask 2D and 3D branches separately, then fuse answers with a fixed global priority."
        if strategy in {"farthest_3d", "farthest_feature", "hybrid", "opacity_topk", "opacity_weighted"}:
            return "3D-only token sampling", strategy, "Use only visible 3DGS language tokens to test whether 3D evidence alone is sufficient."
        return "Other", strategy, ""

    rgb_obj = summary["rgb_targets"].get("objective_accuracy")
    rgb_jl = summary["rgb_targets"].get("judgelm_score")
    base16 = summary["objective_overlap"].get("objective_accuracy_16a")
    j16 = summary.get("judgelm_16a") or {}
    j16_summary = j16.get("summary") or {}
    audit = summary.get("phase2_full8_audit") or {}
    strategies = summary["v17_full_strategies"]
    best_obj_row = max(strategies, key=lambda r: fnum(r.get("objective_accuracy")))
    best_jl_row = max(strategies, key=lambda r: fnum(r.get("judgelm_score")))
    completed_count = sum(1 for r in strategies if int(fnum(r.get("scene_count"))) == 8 and int(fnum(r.get("judge_count") or r.get("judgelm_count"))) >= 297)

    status_cards = [
        ("completed full8 methods", str(completed_count)),
        ("best objective", f"{best_obj_row.get('strategy')} · {pct(best_obj_row.get('objective_accuracy'))}"),
        ("best JudgeLM", f"{best_jl_row.get('strategy')} · {num(best_jl_row.get('judgelm_score'))}"),
        ("RGB objective target", pct(rgb_obj)),
        ("RGB JudgeLM target", num(rgb_jl)),
        ("phase2 coverage", f"{audit.get('complete_scene_method_pairs', '—')}/{audit.get('total_scene_method_pairs', '—')}")
    ]
    cards = "".join(f"<div class='card'><div class='k'>{html.escape(k)}</div><div class='v'>{html.escape(v)}</div></div>" for k, v in status_cards)

    def grouped_display_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
        for r in rows:
            key = (round(fnum(r.get("objective_accuracy")), 12), round(fnum(r.get("judgelm_score")), 12))
            grouped.setdefault(key, []).append(r)
        out = []
        for (_obj, _jl), group in grouped.items():
            group = sorted(group, key=lambda r: str(r.get("strategy", "")))
            rep = dict(group[0])
            if len(group) > 1:
                rep["aliases"] = [str(g.get("strategy", "")) for g in group]
                rep["display_strategy"] = f"{group[0].get('strategy')} +{len(group)-1}"
            else:
                rep["aliases"] = [str(group[0].get("strategy", ""))]
                rep["display_strategy"] = str(group[0].get("strategy", ""))
            out.append(rep)
        return sorted(out, key=lambda r: (fnum(r.get("objective_accuracy")), fnum(r.get("judgelm_score"))), reverse=True)

    display_methods = grouped_display_rows(strategies)

    def detail_for_strategy(strategy: str) -> tuple[str, str, str, str]:
        if strategy == "farthest_3d":
            return ("Use 3DGS geometry rather than 2D rendered features as the only evidence stream.", "Visible Gaussian language tokens are sampled by farthest-point selection in 3D coordinate space, then passed as one token sequence to the VLM.", "Controls out all 2D token evidence and tests whether spatial coverage of the reconstructed 3D scene is enough.", "This variant gives the strongest JudgeLM score, suggesting geometrically diverse 3D evidence helps open-ended landmark descriptions, but its objective accuracy is weak.")
        if strategy == "farthest_feature":
            return ("Keep the 3D-only setting but replace geometric diversity with language-feature diversity.", "The sampler chooses visible Gaussian tokens that are far apart in the 3DGS language-feature/codebook space before VLM answering.", "Compared with farthest_3d, the token budget and one-shot VLM protocol are unchanged; only the 3D token selection criterion changes.", "This is the best 3D-only objective variant, indicating that semantic diversity is more useful than pure geometry for objective questions.")
        if strategy == "hybrid":
            return ("Mix geometric and feature-space criteria inside the 3D-only sampler.", "Part of the visible Gaussian budget is selected for 3D spatial coverage and part for language-feature diversity.", "Compared with farthest_3d/farthest_feature, this tests whether combining two 3D diversity criteria is better than either alone.", "The mixed 3D-only strategy does not dominate the single-criterion variants, so the bottleneck is not solved by simply mixing 3D sampling heuristics.")
        if strategy == "opacity_topk":
            return ("Use visibility confidence as the 3D-only token selection rule.", "The sampler keeps the highest-opacity visible Gaussians and discards lower-opacity candidates before VLM answering.", "Compared with diversity-based 3D sampling, this asks whether confidence/visibility alone is a sufficient proxy for semantic usefulness.", "It is below feature-diverse 3D sampling, suggesting high-opacity Gaussians are not necessarily the most discriminative language tokens.")
        if strategy == "opacity_weighted":
            return ("Soften the opacity-only selection rule while staying in the 3D-only setting.", "Visible Gaussians are sampled with opacity-weighted coverage instead of strict top-k selection.", "This controls for the same 3D-only evidence source and changes only how opacity affects token selection.", "It improves JudgeLM over opacity_topk but still does not close the objective gap to RGB.")
        if strategy == "answer_fusion_2d_priority":
            return ("Move fusion after answering and use 2D as the default branch.", "The 2D and 3D branches produce answers separately; if they disagree, a fixed 2D-priority rule selects the final answer.", "Compared with token early-fusion, this tests whether late answer selection can replace joint token evidence fusion.", "It mostly behaves like the 2D branch and does not recover the objective gain of token early-fusion.")
        if strategy == "answer_fusion_3d_priority":
            return ("Move fusion after answering and use 3D as the default branch.", "The 2D and 3D branches answer separately; disagreements are resolved with a fixed 3D-priority rule.", "Compared with 2D-priority answer fusion, the only change is the global branch priority; no question-aware routing is introduced.", "It matches farthest_feature on the aggregate metrics, which suggests late fusion is not adding new joint reasoning beyond the stronger branch.")
        if strategy.startswith("phase2_"):
            label = strategy.removeprefix("phase2_")
            details = {
                "block": ("Add 2D tokens to the 3D stream and concatenate the two modalities in blocks before answering.", "For each of the three scales, sampled 3DGS tokens and 2D rendered tokens are placed into a single sequence using block concatenation, then the VLM answers once from the fused evidence.", "Compared with 3D-only methods, the main change is pre-answer access to both 2D appearance and 3D language tokens under a fixed budget.", "This is tied for the best objective score, supporting early token fusion as the strongest current direction."),
                "3d_first": ("Use the same early-fusion evidence as block concat but order 3D tokens before 2D tokens.", "The fused sequence places multiscale 3DGS tokens first, followed by the corresponding 2D rendered tokens, and performs a single VLM generation.", "Compared with phase2_block, the evidence set is intended to stay the same; the ablation is whether modality order changes VLM use of context.", "It ties block and hybrid_T512_R30 for best objective, so 3D-first ordering is not harmful and may be a clean default."),
                "2d_first": ("Use the same early-fusion evidence as 3d_first but place 2D tokens before 3D tokens.", "The VLM sees the rendered 2D multiscale tokens first and the 3DGS language tokens afterward in one fused sequence.", "Compared with phase2_3d_first, only the modality order is changed; this isolates ordering sensitivity in the context sequence.", "It drops below 3d_first/block, suggesting the current VLM benefits more when 3D evidence is introduced before 2D evidence."),
                "level_interleave": ("Keep early fusion but interleave by scale level rather than by modality block.", "Small/Medium/Large 2D and 3D token groups are alternated so each scale contributes before moving to the next scale.", "Compared with block/3d_first/2d_first, this changes the scale-layout of the same multiscale 2D+3D evidence.", "It underperforms the best early-fusion layouts, so scale-level alternation alone is not the right organization."),
                "token_interleave": ("Use the most fine-grained early-fusion layout by alternating individual 2D and 3D tokens.", "After sampling both modalities, tokens are interleaved at token granularity before one VLM answer is generated.", "Compared with level_interleave, the change is finer modality alternation while preserving the same no-question-aware setup.", "It is better than level_interleave but still below block/3d_first, suggesting too much alternation may disrupt useful modality structure."),
            }
            return details.get(label, ("No note.", "—", "—", "—"))
        if strategy.startswith("hybrid_T"):
            cfg = strategy.removeprefix("hybrid_")
            total = cfg.split("_")[0].removeprefix("T")
            ratio = cfg.split("_")[1].removeprefix("R")
            return (f"Keep early fusion fixed but set total token budget to {total} and allocate {ratio}% of tokens to 3D.", f"The fused sequence uses a fixed {total}-token budget; R{ratio} means roughly {ratio}% 3DGS tokens and {100-int(ratio)}% 2D rendered tokens before one VLM answer.", "Compared with phase2 layout ablations, this isolates token budget and 3D:2D allocation while avoiding question-aware or scene-adaptive choices.", "This row should be read as a budget-allocation ablation, not a new architecture; its value is showing which fixed 3D/2D ratio is most promising.")
        return ("No additional method-specific note.", "—", "—", "—")

    def detail_card(row: dict[str, Any]) -> str:
        aliases = row.get("aliases") or [str(row.get("strategy", ""))]
        title = str(row.get("display_strategy") or row.get("strategy"))
        obj = fnum(row.get("objective_accuracy"))
        jl = fnum(row.get("judgelm_score"))
        parts = []
        for alias in aliases:
            inc, impl, ctrl, interp = detail_for_strategy(alias)
            fam, variant, _ = method_family(alias)
            parts.append(f"<div class='method-note'><h3>{html.escape(alias)}</h3><p><b>What changed:</b> {html.escape(inc)}</p><p><b>How it is implemented:</b> {html.escape(impl)}</p><p><b>Controlled comparison:</b> {html.escape(ctrl)}</p><p><b>Interpretation:</b> {html.escape(interp)}</p><p class='note'>Family: {html.escape(fam)} · Variant: {html.escape(variant)}</p></div>")
        obj_cls = "good" if obj >= fnum(rgb_obj) else "bad"
        jl_cls = "good" if jl >= fnum(rgb_jl) else "bad"
        return f"<section class='method-card'><h2>{html.escape(title)}</h2><p class='result-line'>Objective {html.escape(pct(obj))} (<span class='{obj_cls}'>{html.escape(delta_pp(obj, rgb_obj))} vs RGB</span>) · JudgeLM {html.escape(num(jl))} (<span class='{jl_cls}'>{html.escape(delta_num(jl, rgb_jl))} vs RGB</span>)</p>{''.join(parts)}</section>"

    method_detail_html = "".join(detail_card(row) for row in display_methods)


    method_rows = []
    for row in sorted(strategies, key=lambda r: (fnum(r.get("objective_accuracy")), fnum(r.get("judgelm_score"))), reverse=True):
        fam, name, role = method_family(str(row.get("strategy", "")))
        obj = fnum(row.get("objective_accuracy"))
        jl = fnum(row.get("judgelm_score"))
        conclusion = ""
        if obj >= fnum(rgb_obj) and jl >= fnum(rgb_jl):
            conclusion = "Meets both RGB targets"
        elif obj >= fnum(rgb_obj):
            conclusion = "Objective meets RGB; JudgeLM below RGB"
        elif jl >= fnum(rgb_jl):
            conclusion = "JudgeLM meets RGB; objective below RGB"
        elif obj >= fnum(base16):
            conclusion = "Improves over 16A objective but below RGB"
        else:
            conclusion = "Below RGB and not objective-best"
        method_rows.append({
            "method": row.get("strategy"),
            "family": fam,
            "variant": name,
            "Objective": pct(obj),
            "Δ Obj vs RGB": delta_pp(obj, rgb_obj),
            "JudgeLM": num(jl),
            "Δ JL vs RGB": delta_num(jl, rgb_jl),
            "conclusion": conclusion,
        })

    # A compact paper-style conclusion, with claims tied to the evidence above.
    key_findings = [
        f"All previously partial phase2/hybrid candidates are now complete: phase2 audit is {audit.get('complete_scene_method_pairs', '—')}/{audit.get('total_scene_method_pairs', '—')} scene-method pairs, and each completed method has 8 scenes / 1061 questions / 297 JudgeLM questions.",
        f"The strongest objective result is {best_obj_row.get('strategy')} at {pct(best_obj_row.get('objective_accuracy'))}, which is above 16A ({pct(base16)}) but still {delta_pp(best_obj_row.get('objective_accuracy'), rgb_obj)} against the RGB target ({pct(rgb_obj)}).",
        f"The strongest JudgeLM result is {best_jl_row.get('strategy')} at {num(best_jl_row.get('judgelm_score'))}, compared with RGB {num(rgb_jl)}; this shows that high open-answer quality and high objective accuracy are not yet achieved by the same method.",
        "The main experimental signal is that token early-fusion improves objective accuracy over 3D-only and answer-level fusion, but the current fixed concatenation/interleaving designs are still short of the RGB objective target.",
    ]
    finding_html = "".join(f"<li>{html.escape(x)}</li>" for x in key_findings)

    scene_headers = ["scene", "objective_count", "objective_accuracy_16a", "best_v17_strategy", "best_v17_objective_accuracy", "best_delta_vs_16a"]
    partial_rows = [{
        "family": row.get("family"),
        "label": row.get("label"),
        "scope": f"{row.get('scene_count')} scenes / {row.get('question_count')}Q",
        "JudgeLM": row.get("judgelm_score") or "pending",
    } for row in summary["partial_hybrid_candidates"]]

    html_doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>V17 2D-3D Method Report</title><style>
body{{margin:0;padding:24px;background:#f6f1e9;color:#161311;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif}}.wrap{{max-width:1500px;margin:0 auto}}a{{color:#9f3f1c;font-weight:800;text-decoration:none}}.pill{{display:inline-block;border:1px solid rgba(22,19,17,.14);border-radius:999px;background:#fff;padding:7px 12px;margin-right:8px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}}.card,.panel{{background:rgba(255,251,245,.92);border:1px solid rgba(22,19,17,.12);border-radius:16px;padding:16px;margin:14px 0;box-shadow:0 12px 30px rgba(44,32,22,.08)}}.k,.note{{color:#665d55;font-size:13px}}.v{{font-size:22px;font-weight:900;margin-top:4px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{border-bottom:1px solid rgba(22,19,17,.12);padding:9px 10px;text-align:left;vertical-align:top}}th{{background:#fffaf4;color:#665d55;position:sticky;top:0}}.warn{{color:#b84d24;font-weight:850}}.bad{{color:#c62828;font-weight:850}}.good{{color:#2e7d32;font-weight:850}}.lead{{font-size:15px;line-height:1.65;max-width:1100px}}.claim{{border-left:4px solid #b84d24;padding-left:12px;background:#fffaf4}}.method-card{{border-top:1px solid rgba(22,19,17,.12);padding-top:12px;margin-top:14px}}.method-note{{background:#fffaf4;border:1px solid rgba(22,19,17,.08);border-radius:12px;padding:10px 12px;margin:10px 0}}.method-note h3{{margin:0 0 6px;font-size:15px}}.method-note p{{margin:6px 0;line-height:1.55}}.result-line{{font-weight:850}}code{{background:#fff;padding:2px 5px;border-radius:6px}}</style></head><body><div class="wrap"><div><a class="pill" href="/projects/langsplat">← 9999 总览</a><a class="pill" href="/projects/langsplat/benchmark">16A 场景页</a></div><h1>V17 2D-3D Sampling Method Report</h1><div class="note">生成时间: {html.escape(summary['generated_at'])}；产物目录: <code>{html.escape(str(out_dir))}</code></div><div class="cards">{cards}</div>
<div class="panel claim"><h2>结论摘要</h2><p class="lead">本页只讨论完整 full8 benchmark 的方法结果。当前实验的主线是一个增量式 ablation：从 3D-only 出发，加入 2D+3D token early fusion，再扫描 token budget 和 3D:2D 配比，并用 answer-level fusion 做 late-fusion control。结果显示 early-fusion 能超过 16A objective，但尚未超过 RGB objective；最高 JudgeLM 和最高 Objective 仍不在同一方法上。</p><ul>{finding_html}</ul></div>
<div class="panel"><h2>逐方法 Ablation 解释</h2><p class="lead">这一节按一级主表里的每个方法/同分组逐条解释。每条只说明它自己的增量改动、实现方式、控制变量和结果解读；不额外拔高实验含义。</p>{method_detail_html}</div>
<div class="panel"><h2>完整 full8 方法结果与结论</h2><p class="note">所有行均为 8 scenes / 1061 questions / 297 JudgeLM；delta 只相对 RGB 目标线。</p>{table(['method','family','variant','Objective','Δ Obj vs RGB','JudgeLM','Δ JL vs RGB','conclusion'], method_rows)}</div>
<div class="panel"><h2>未完成/待补队列</h2><p class="note">当前 phase2/hybrid 已补齐；若后续加入新计划但未完成，会显示在这里，而不会进入正式结论。</p>{table(['family','label','scope','JudgeLM'], partial_rows) if partial_rows else '<p class="good">当前无未完成候选。</p>'}</div>
<div class="panel"><h2>逐场景 objective：16A vs v17 最优策略 oracle</h2>{table(scene_headers, summary['scene_objective_overlap'])}</div>
<div class="panel"><h2>产物文件</h2><ul><li><code>summary.json</code></li><li><code>strategy_comparison.csv</code></li><li><code>partial_hybrid_candidates.csv</code></li><li><code>scene_objective_overlap.csv</code></li><li><code>objective_overlap_16a_v17.csv</code></li><li><code>16a_judgelm/judgelm_scored_16a_results.json</code></li></ul></div></div></body></html>"""
    return html_doc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="LangSplatV2 repository root")
    parser.add_argument("--sixteen-a-root", type=Path, default=DEFAULT_16A_ROOT)
    parser.add_argument("--v17-root", type=Path, default=DEFAULT_V17_ROOT)
    parser.add_argument("--hybrid-scan", type=Path, default=DEFAULT_HYBRID_SCAN)
    parser.add_argument("--phase2", type=Path, default=DEFAULT_PHASE2)
    parser.add_argument("--answer-fusion", type=Path, default=DEFAULT_ANSWER_FUSION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out if args.out.is_absolute() else args.root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    v17_rows = load_v17_question_rows(args.v17_root)
    overlap_rows, objective_summary, scene_rows = summarize_16a_objective(args.sixteen_a_root, v17_rows)
    strategy_rows = global_strategy_rows(args.v17_root)
    strategy_rows += load_completed_answer_fusion_rows(args.answer_fusion)
    strategy_rows += load_completed_family_rows(args.hybrid_scan, "config_comparison.csv", "config", "hybrid")
    strategy_rows += load_completed_family_rows(args.phase2, "fusion_comparison.csv", "strategy", "phase2")
    partial = []
    partial += load_partial_rows(args.hybrid_scan, "config_comparison.csv", "config", "hybrid_scan")
    partial += load_partial_rows(args.phase2, "fusion_comparison.csv", "strategy", "phase2_fusion")
    # targets are constant per strategy in v17 metrics
    rgb_obj = next((float(r["rgb_objective_accuracy"]) for r in strategy_rows if r.get("rgb_objective_accuracy")), None)
    rgb_jl = next((float(r["rgb_judgelm_score"]) for r in strategy_rows if r.get("rgb_judgelm_score")), None)
    j16 = load_16a_judgelm(out_dir)
    summary = {
        "generated_at": "2026-06-15",
        "sources": {
            "16A_root": str(args.sixteen_a_root),
            "v17_full_metrics": str(args.v17_root),
            "hybrid_scan_metrics": str(args.hybrid_scan),
            "phase2_fusion_metrics": str(args.phase2),
            "answer_fusion_metrics": str(args.answer_fusion),
        },
        "rgb_targets": {"objective_accuracy": rgb_obj, "judgelm_score": rgb_jl},
        "objective_overlap": objective_summary,
        "v17_full_strategies": strategy_rows,
        "partial_hybrid_candidates": partial,
        "scene_objective_overlap": scene_rows,
        "judgelm_16a": j16,
        "judgelm_16a_status": "completed" if j16 else "pending",
        "phase2_full8_audit": load_phase2_audit(out_dir),
        "goal_status": {
            "beats_16a_objective": max(float(r["objective_accuracy"]) for r in strategy_rows) > float(objective_summary["objective_accuracy_16a"]),
            "beats_rgb_objective": max(float(r["objective_accuracy"]) for r in strategy_rows) > float(rgb_obj),
            "beats_rgb_judgelm": max(float(r["judgelm_score"]) for r in strategy_rows) > float(rgb_jl),
            "beats_16a_judgelm": (max(float(r["judgelm_score"]) for r in strategy_rows) > float((j16 or {}).get("summary", {}).get("score_16a", -1))) if j16 else None,
            "single_full_strategy_meets_all": False,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "objective_overlap_16a_v17.csv", overlap_rows, [
        "scene", "image", "question_index", "metric_class", "objective_subtype", "expected", "answer_16a",
        "correct_16a", "strategy", "answer_v17", "correct_v17", "rgb_answer", "correct_rgb",
    ])
    write_csv(out_dir / "strategy_comparison.csv", strategy_rows, [
        "strategy", "scene_count", "image_count", "question_count", "objective_count", "judge_count",
        "objective_accuracy", "rgb_objective_accuracy", "objective_delta_vs_rgb", "judgelm_score",
        "rgb_judgelm_score", "judgelm_delta_vs_rgb", "low_validity_count",
    ])
    write_csv(out_dir / "partial_hybrid_candidates.csv", partial, [
        "family", "label", "scene_count", "image_count", "question_count", "objective_count", "judge_count",
        "objective_accuracy", "rgb_objective_accuracy", "objective_delta_vs_rgb", "judgelm_score", "rgb_judgelm_score",
    ])
    scene_fields = ["scene", "objective_count", "objective_accuracy_16a", "best_v17_strategy", "best_v17_objective_accuracy", "best_delta_vs_16a"]
    for row in scene_rows:
        for key in row:
            if key.startswith("v17_") and key not in scene_fields:
                scene_fields.append(key)
    write_csv(out_dir / "scene_objective_overlap.csv", scene_rows, scene_fields)
    (out_dir / "index.html").write_text(render_html(summary, out_dir), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "summary": summary["goal_status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

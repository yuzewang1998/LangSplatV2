#!/usr/bin/env python3
"""Question-level error profile for V18 methods versus RGB baseline.

This is intentionally offline and deterministic: it only reads completed
question_metrics.csv / summary.json files produced by score_v18_results.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

DEFAULT_EXPERIMENTS = [
    "v18_tokpipe_best2d_best3d_m2tokcat_m3tokcat_mmtokscale_odirect_T576_R5_C1",
    "v18_tokpipe_best2d_best3d_m2tokcat_m3tokcat_mmtokscale_odirect_T512_R5_C1",
    "v18_tokpipe_best2d_best3d_m2tokcat_m3tokcat_mmtok2d3d_odirect_T512_R30_C1",
]
DEFAULT_PRIMARY = DEFAULT_EXPERIMENTS[0]

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    (
        "identity_location",
        [
            r"\bfamous landmark\b",
            r"\bwhat landmark\b",
            r"\bwhat famous\b",
            r"\bname of (the )?(landmark|building|sculpture|group)\b",
            r"\blandmark scene\b",
            r"\bwhere\b",
            r"\bcity\b",
            r"\bcountry\b",
            r"\blocated\b",
        ],
    ),
    (
        "style_history_knowledge",
        [
            r"\barchitectural style\b",
            r"\barchitecture\b",
            r"\bneoclassical\b",
            r"\bgothic\b",
            r"\bclassical\b",
            r"\bcolumn order\b",
            r"\bdoric\b",
            r"\bionic\b",
            r"\bcorinthian\b",
            r"\binspiration\b",
            r"\bfunction\b",
            r"\boriginal\b",
            r"\bhistorical\b",
            r"\binscription\b",
            r"\blatin\b",
            r"\bmean\b",
        ],
    ),
    (
        "component_presence",
        [
            r"\bwhich (visible )?component\b",
            r"\bcomponent (is )?(not )?visibly present\b",
            r"\bnot visibly present\b",
            r"\bvisibly present\b",
            r"\bstands in front\b",
            r"\bmatches this description\b",
        ],
    ),
    (
        "spatial_view_visibility",
        [
            r"\boblique\b",
            r"\bfront(al)? view\b",
            r"\bside view\b",
            r"\bangle\b",
            r"\bvisible\b",
            r"\bcropped\b",
            r"\bcenter(ed)?\b",
            r"\bleft\b",
            r"\bright\b",
            r"\btop\b",
            r"\blower\b",
            r"\bbehind\b",
            r"\bthrough\b",
            r"\bdirection\b",
            r"\bfacing\b",
            r"\bdominates?\b",
        ],
    ),
    (
        "part_attribute_semantics",
        [
            r"\bfeature\b",
            r"\bstructure\b",
            r"\belement\b",
            r"\bcomponent\b",
            r"\bmaterial\b",
            r"\bcolor\b",
            r"\bcolour\b",
            r"\bappearance\b",
            r"\bshape\b",
            r"\bprofile\b",
            r"\brelief\b",
            r"\bentablature\b",
            r"\bpediment\b",
            r"\bfrieze\b",
            r"\bfacade\b",
            r"\bsculptur",
            r"\bdome\b",
            r"\barch\b",
            r"\bportal\b",
        ],
    ),
]

SCENE_LABELS = {
    "brandenburg_gate": "Brandenburg Gate",
    "buckingham_palace": "Buckingham Palace",
    "notre_dame_front_facade": "Notre-Dame front facade",
    "pantheon_exterior": "Pantheon exterior",
    "sacre_coeur": "Sacré-Cœur",
    "taj_mahal": "Taj Mahal",
    "temple_nara_japan": "Tōdai-ji / Nara temple",
    "trevi_fountain": "Trevi Fountain",
}


@dataclass(frozen=True)
class ExpData:
    exp_id: str
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def fnum(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        v = float(value)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.2f}%"


def pp(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:+.2f}pp"


def num(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def short_exp(exp_id: str) -> str:
    if "mmtokscale" in exp_id:
        m = re.search(r"_(T\d+)_(R\d+)_", exp_id)
        return f"mmtokscale_{m.group(1)}_{m.group(2)}" if m else "mmtokscale"
    if "mmtok2d3d" in exp_id:
        m = re.search(r"_(T\d+)_(R\d+)_", exp_id)
        return f"mmtok2d3d_{m.group(1)}_{m.group(2)}" if m else "mmtok2d3d"
    if "2donly" in exp_id:
        parts = exp_id.split("_")
        return f"2donly_{parts[2]}" if len(parts) > 2 else exp_id
    if "3donly" in exp_id:
        parts = exp_id.split("_")
        return f"3donly_{parts[3]}" if len(parts) > 3 else exp_id
    return exp_id


def question_category(row: dict[str, Any]) -> str:
    metric_class = (row.get("metric_class") or "").strip().lower()
    subtype = (row.get("objective_subtype") or "").strip().lower()
    question = (row.get("question") or "").strip().lower()
    expected = (row.get("expected") or "").strip().lower()
    # Prefer the question wording; expected answer is only appended for open JudgeLM
    # prompts where the question is intentionally short. This prevents words like
    # "columns" in an option/answer from turning style questions into counting.
    text = question if metric_class != "judgelm" else f"{question} {expected}"

    if metric_class == "number" or subtype == "number" or "answer with a number" in question or re.search(r"\bhow many\b|\bnumber of\b", question):
        return "count_numeric"

    if metric_class == "yes_no" or subtype == "yes_no" or "answer with yes or no" in question:
        if any(re.search(pat, text) for pat in [r"\boblique\b", r"\bvisible\b", r"\bcropped\b", r"\bfront(al)?\b", r"\bside\b", r"\bangle\b", r"\bcenter(ed)?\b", r"\bthrough\b", r"\bbehind\b", r"\bleft\b", r"\bright\b"]):
            return "spatial_view_visibility"
        if any(re.search(pat, text) for pat in [r"\bstyle\b", r"\border\b", r"\bdoric\b", r"\bionic\b", r"\bcorinthian\b", r"\bfunction\b", r"\bhistorical\b"]):
            return "style_history_knowledge"
        if any(re.search(pat, text) for pat in [r"\bcolumns?\b", r"\barches?\b", r"\bdomes?\b", r"\bstatues?\b", r"\bsculpture\b", r"\bwindow\b", r"\brelief\b", r"\bpediment\b", r"\bentablature\b"]):
            return "part_attribute_semantics"
        return "yes_no_general"

    for category, patterns in CATEGORY_RULES:
        if any(re.search(pattern, text) for pattern in patterns):
            return category
    if metric_class == "multiple_choice":
        return "multiple_choice_general"
    if metric_class == "judgelm":
        return "open_description_general"
    return "other"


def question_key(row: dict[str, Any]) -> str:
    return "||".join([
        str(row.get("scene", "")),
        str(row.get("image", "")),
        str(row.get("question_index", "")),
        str(row.get("question", "")),
    ])


def stable_order(row: dict[str, Any]) -> str:
    return hashlib.sha1(question_key(row).encode("utf-8", errors="ignore")).hexdigest()


def read_exp(root: Path, exp_id: str) -> ExpData:
    exp_dir = root / "results" / exp_id
    qpath = exp_dir / "question_metrics.csv"
    spath = exp_dir / "summary.json"
    if not qpath.exists():
        raise FileNotFoundError(f"missing question_metrics.csv for {exp_id}: {qpath}")
    rows = list(csv.DictReader(qpath.open("r", encoding="utf-8-sig", newline="")))
    summary = json.loads(spath.read_text(encoding="utf-8")) if spath.exists() else {}
    return ExpData(exp_id=exp_id, rows=rows, summary=summary)


def add_derived(row: dict[str, Any], exp_id: str) -> dict[str, Any]:
    out = dict(row)
    out["experiment_id"] = exp_id
    out["experiment_short"] = short_exp(exp_id)
    out["category"] = question_category(row)
    out["scene_label"] = SCENE_LABELS.get(str(row.get("scene", "")), str(row.get("scene", "")))
    eval_type = (row.get("eval_type") or "").strip().lower()
    metric_class = (row.get("metric_class") or "").strip().lower()
    out["is_objective"] = eval_type == "objective" or metric_class in {"yes_no", "multiple_choice", "number"}
    out["is_judgelm"] = eval_type == "judgelm" or metric_class == "judgelm"
    if out["is_objective"]:
        ours = truthy(row.get("final_correct"))
        rgb = truthy(row.get("rgb_correct"))
        out["ours_correct_bool"] = ours
        out["rgb_correct_bool"] = rgb
        if ours and rgb:
            outcome = "both_correct"
        elif ours and not rgb:
            outcome = "ours_only_correct"
        elif (not ours) and rgb:
            outcome = "rgb_only_correct"
        else:
            outcome = "both_wrong"
        out["objective_outcome"] = outcome
    if out["is_judgelm"]:
        s = fnum(row.get("score"))
        rs = fnum(row.get("rgb_score"))
        out["score_num"] = s
        out["rgb_score_num"] = rs
        if s is not None and rs is not None:
            delta = s - rs
            out["judgelm_delta"] = delta
            if delta > 1e-9:
                outcome = "ours_higher"
            elif delta < -1e-9:
                outcome = "rgb_higher"
            else:
                outcome = "tie"
            out["judgelm_outcome_exact"] = outcome
            if delta >= 0.25:
                out["judgelm_outcome_margin"] = "ours_higher_025"
            elif delta <= -0.25:
                out["judgelm_outcome_margin"] = "rgb_higher_025"
            else:
                out["judgelm_outcome_margin"] = "near_tie_025"
    return out


def aggregate_objective(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("is_objective"):
            groups[tuple(r.get(k, "") for k in keys)].append(r)
    out: list[dict[str, Any]] = []
    for key, items in groups.items():
        n = len(items)
        ours_ok = sum(1 for r in items if r.get("ours_correct_bool"))
        rgb_ok = sum(1 for r in items if r.get("rgb_correct_bool"))
        outcomes = Counter(str(r.get("objective_outcome")) for r in items)
        rec = {k: v for k, v in zip(keys, key)}
        rec.update(
            {
                "eval_type": "objective",
                "n": n,
                "ours_acc": ours_ok / n if n else None,
                "rgb_acc": rgb_ok / n if n else None,
                "delta_acc": (ours_ok - rgb_ok) / n if n else None,
                "ours_only_correct": outcomes.get("ours_only_correct", 0),
                "rgb_only_correct": outcomes.get("rgb_only_correct", 0),
                "both_correct": outcomes.get("both_correct", 0),
                "both_wrong": outcomes.get("both_wrong", 0),
            }
        )
        out.append(rec)
    return sorted(out, key=lambda r: (str(r.get(keys[0], "")), -abs(float(r.get("delta_acc") or 0)), -int(r.get("n") or 0)))


def aggregate_judgelm(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("is_judgelm") and r.get("score_num") is not None and r.get("rgb_score_num") is not None:
            groups[tuple(r.get(k, "") for k in keys)].append(r)
    out: list[dict[str, Any]] = []
    for key, items in groups.items():
        n = len(items)
        deltas = [float(r["judgelm_delta"]) for r in items]
        outcomes_exact = Counter(str(r.get("judgelm_outcome_exact")) for r in items)
        outcomes_margin = Counter(str(r.get("judgelm_outcome_margin")) for r in items)
        rec = {k: v for k, v in zip(keys, key)}
        rec.update(
            {
                "eval_type": "judgelm",
                "n": n,
                "ours_score": mean(float(r["score_num"]) for r in items),
                "rgb_score": mean(float(r["rgb_score_num"]) for r in items),
                "delta_score": mean(deltas),
                "ours_higher_exact": outcomes_exact.get("ours_higher", 0),
                "rgb_higher_exact": outcomes_exact.get("rgb_higher", 0),
                "tie_exact": outcomes_exact.get("tie", 0),
                "ours_higher_025": outcomes_margin.get("ours_higher_025", 0),
                "rgb_higher_025": outcomes_margin.get("rgb_higher_025", 0),
                "near_tie_025": outcomes_margin.get("near_tie_025", 0),
            }
        )
        out.append(rec)
    return sorted(out, key=lambda r: (str(r.get(keys[0], "")), -abs(float(r.get("delta_score") or 0)), -int(r.get("n") or 0)))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in names:
                    names.append(k)
        fieldnames = names
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def sample_rows(rows: list[dict[str, Any]], predicate, limit: int, sort_key=None) -> list[dict[str, Any]]:
    items = [r for r in rows if predicate(r)]
    if sort_key:
        items.sort(key=sort_key)
    else:
        items.sort(key=stable_order)
    return items[:limit]


def slim_sample(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_short": r.get("experiment_short", ""),
        "scene": r.get("scene", ""),
        "image": r.get("image", ""),
        "question_index": r.get("question_index", ""),
        "metric_class": r.get("metric_class", ""),
        "category": r.get("category", ""),
        "question": r.get("question", ""),
        "expected": r.get("expected", ""),
        "ours_answer": r.get("final_answer", ""),
        "rgb_answer": r.get("rgb_answer", ""),
        "ours_correct": r.get("final_correct", ""),
        "rgb_correct": r.get("rgb_correct", ""),
        "ours_score": r.get("score", ""),
        "rgb_score": r.get("rgb_score", ""),
        "judgelm_delta": fnum(r.get("judgelm_delta")),
        "objective_outcome": r.get("objective_outcome", ""),
        "judgelm_outcome": r.get("judgelm_outcome_exact", ""),
        "source_file": r.get("source_file", ""),
    }


def strip_instruction(question: str) -> str:
    text = (question or "").strip()
    text = re.sub(r"^Answer with (yes or no|a number|the option letter) only\.\s*", "", text, flags=re.I)
    return text.strip()


def option_map(question: str) -> dict[str, str]:
    """Extract A./B./C. option text when present."""
    text = (question or "").replace("\n", " ")
    matches = list(re.finditer(r"\b([A-D])\.\s*", text))
    out: dict[str, str] = {}
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        out[m.group(1).upper()] = text[start:end].strip()
    return out


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
}


def parse_number_answer(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    m = re.search(r"-?\d+", text)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    for word, num_value in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return num_value
    return None


def parse_yes_no(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if re.search(r"\byes\b", text):
        return "yes"
    if re.search(r"\bno\b", text):
        return "no"
    return None


def parse_option_answer(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m else None


def answer_error_pattern(row: dict[str, Any], answer_field: str) -> str:
    metric = str(row.get("metric_class") or "").lower()
    expected = row.get("expected")
    answer = row.get(answer_field)
    if metric == "number":
        exp_num = parse_number_answer(expected)
        ans_num = parse_number_answer(answer)
        if exp_num is None or ans_num is None:
            return "non_numeric"
        if ans_num == exp_num:
            return "correct"
        if ans_num < exp_num:
            return f"undercount({ans_num}<{exp_num})"
        return f"overcount({ans_num}>{exp_num})"
    if metric == "yes_no":
        exp_yn = parse_yes_no(expected)
        ans_yn = parse_yes_no(answer)
        if exp_yn is None or ans_yn is None:
            return "non_yes_no"
        if ans_yn == exp_yn:
            return "correct"
        if exp_yn == "yes" and ans_yn == "no":
            return "false_negative_yes->no"
        if exp_yn == "no" and ans_yn == "yes":
            return "false_positive_no->yes"
        return "wrong_yes_no"
    if metric == "multiple_choice":
        exp_opt = parse_option_answer(expected)
        ans_opt = parse_option_answer(answer)
        if exp_opt is None or ans_opt is None:
            return "non_option"
        if ans_opt == exp_opt:
            return "correct"
        return f"wrong_option({ans_opt}!={exp_opt})"
    return "wrong"


def side_by_side_objective(row: dict[str, Any], group: str) -> dict[str, Any]:
    opts = option_map(str(row.get("question") or ""))
    exp_opt = parse_option_answer(row.get("expected"))
    ours_opt = parse_option_answer(row.get("final_answer"))
    rgb_opt = parse_option_answer(row.get("rgb_answer"))
    return {
        "group": group,
        "scene": row.get("scene", ""),
        "image": row.get("image", ""),
        "question_index": row.get("question_index", ""),
        "metric_class": row.get("metric_class", ""),
        "category": row.get("category", ""),
        "question": strip_instruction(str(row.get("question") or "")),
        "expected": row.get("expected", ""),
        "expected_option_text": opts.get(exp_opt or "", ""),
        "v18_answer": row.get("final_answer", ""),
        "v18_option_text": opts.get(ours_opt or "", ""),
        "v18_error": answer_error_pattern(row, "final_answer"),
        "rgb_answer": row.get("rgb_answer", ""),
        "rgb_option_text": opts.get(rgb_opt or "", ""),
        "rgb_error": answer_error_pattern(row, "rgb_answer"),
        "source_file": row.get("source_file", ""),
    }


def side_by_side_judgelm(row: dict[str, Any], group: str) -> dict[str, Any]:
    delta = fnum(row.get("judgelm_delta"))
    return {
        "group": group,
        "scene": row.get("scene", ""),
        "image": row.get("image", ""),
        "question_index": row.get("question_index", ""),
        "metric_class": row.get("metric_class", ""),
        "category": row.get("category", ""),
        "question": strip_instruction(str(row.get("question") or "")),
        "expected": row.get("expected", ""),
        "v18_answer": row.get("final_answer", ""),
        "v18_score": row.get("score", ""),
        "rgb_answer": row.get("rgb_answer", ""),
        "rgb_score": row.get("rgb_score", ""),
        "delta_judgelm": delta,
        "source_file": row.get("source_file", ""),
    }


def pick_representative(rows: list[dict[str, Any]], per_group: int = 18) -> list[dict[str, Any]]:
    """Prefer diversity over raw first rows: round-robin by metric/category/scene."""
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=stable_order):
        buckets[(str(row.get("metric_class")), str(row.get("category")), str(row.get("scene")))].append(row)
    out: list[dict[str, Any]] = []
    while len(out) < per_group and buckets:
        for key in sorted(list(buckets.keys())):
            if not buckets.get(key):
                buckets.pop(key, None)
                continue
            out.append(buckets[key].pop(0))
            if len(out) >= per_group:
                break
    return out


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return "_无数据_\n"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for r in rows:
        cells = []
        for key, _ in columns:
            v = r.get(key, "")
            if isinstance(v, float):
                if "acc" in key or key.startswith("delta_acc"):
                    s = pp(v) if key.startswith("delta") else pct(v)
                else:
                    s = num(v)
            else:
                s = str(v)
            cells.append(s.replace("\n", "<br>").replace("|", "\\|"))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def html_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return "<p class='muted'>无数据</p>"
    th = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    trs = []
    for r in rows:
        tds = []
        for key, _ in columns:
            v = r.get(key, "")
            if isinstance(v, float):
                if "acc" in key or key.startswith("delta_acc"):
                    s = pp(v) if key.startswith("delta") else pct(v)
                else:
                    s = num(v)
            else:
                s = str(v)
            tds.append(f"<td>{html.escape(s)}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return "<div class='table-wrap'><table><thead><tr>" + th + "</tr></thead><tbody>" + "".join(trs) + "</tbody></table></div>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("LangSplatV2/experiment_results/benchmark/v18_pipeline_ablation"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--exp-id", action="append", default=[])
    parser.add_argument("--primary", default=DEFAULT_PRIMARY)
    args = parser.parse_args()

    root: Path = args.root
    output = args.output or root / "diagnostics" / "v18_vs_rgb_error_profile"
    exp_ids = args.exp_id or DEFAULT_EXPERIMENTS
    if args.primary not in exp_ids:
        exp_ids = [args.primary, *exp_ids]

    exps = [read_exp(root, exp_id) for exp_id in exp_ids]
    derived_by_exp: dict[str, list[dict[str, Any]]] = {
        e.exp_id: [add_derived(r, e.exp_id) for r in e.rows] for e in exps
    }
    primary_rows = derived_by_exp[args.primary]

    # Full-benchmark validation: these diagnostics should not silently use partial runs.
    validation = {}
    for e in exps:
        rows = derived_by_exp[e.exp_id]
        summary = e.summary
        validation[e.exp_id] = {
            "rows": len(rows),
            "scene_count": len({r.get("scene") for r in rows if r.get("scene")}),
            "image_count": len({(r.get("scene"), r.get("image")) for r in rows if r.get("image")}),
            "objective_count": sum(1 for r in rows if r.get("is_objective")),
            "judgelm_count": sum(1 for r in rows if r.get("is_judgelm")),
            "summary_full_benchmark_complete": bool(summary.get("full_benchmark_complete")),
            "summary_status": summary.get("status"),
        }
        if validation[e.exp_id]["scene_count"] < 8 or validation[e.exp_id]["rows"] < 1061:
            raise RuntimeError(f"Refusing partial diagnostic for {e.exp_id}: {validation[e.exp_id]}")

    objective_by_category = aggregate_objective(primary_rows, ("category", "metric_class"))
    objective_by_scene = aggregate_objective(primary_rows, ("scene", "metric_class"))
    objective_by_scene_category = aggregate_objective(primary_rows, ("scene", "category", "metric_class"))
    judgelm_by_category = aggregate_judgelm(primary_rows, ("category",))
    judgelm_by_scene = aggregate_judgelm(primary_rows, ("scene",))
    judgelm_by_scene_category = aggregate_judgelm(primary_rows, ("scene", "category"))

    all_exp_overview: list[dict[str, Any]] = []
    for e in exps:
        rows = derived_by_exp[e.exp_id]
        obj = [r for r in rows if r.get("is_objective")]
        jl = [r for r in rows if r.get("is_judgelm") and r.get("score_num") is not None and r.get("rgb_score_num") is not None]
        outcomes = Counter(str(r.get("objective_outcome")) for r in obj)
        jl_outcomes = Counter(str(r.get("judgelm_outcome_margin")) for r in jl)
        ours_acc = sum(1 for r in obj if r.get("ours_correct_bool")) / len(obj)
        rgb_acc = sum(1 for r in obj if r.get("rgb_correct_bool")) / len(obj)
        ours_jl = mean(float(r["score_num"]) for r in jl) if jl else None
        rgb_jl = mean(float(r["rgb_score_num"]) for r in jl) if jl else None
        all_exp_overview.append(
            {
                "experiment_id": e.exp_id,
                "experiment_short": short_exp(e.exp_id),
                "objective_n": len(obj),
                "ours_obj_acc": ours_acc,
                "rgb_obj_acc": rgb_acc,
                "delta_obj_acc": ours_acc - rgb_acc,
                "ours_only_correct": outcomes.get("ours_only_correct", 0),
                "rgb_only_correct": outcomes.get("rgb_only_correct", 0),
                "both_correct": outcomes.get("both_correct", 0),
                "both_wrong": outcomes.get("both_wrong", 0),
                "judgelm_n": len(jl),
                "ours_judgelm": ours_jl,
                "rgb_judgelm": rgb_jl,
                "delta_judgelm": (ours_jl - rgb_jl) if ours_jl is not None and rgb_jl is not None else None,
                "ours_higher_025": jl_outcomes.get("ours_higher_025", 0),
                "rgb_higher_025": jl_outcomes.get("rgb_higher_025", 0),
                "near_tie_025": jl_outcomes.get("near_tie_025", 0),
            }
        )

    sample_sets = {
        "objective_ours_only_correct": [slim_sample(r) for r in sample_rows(primary_rows, lambda r: r.get("objective_outcome") == "ours_only_correct", 40)],
        "objective_rgb_only_correct": [slim_sample(r) for r in sample_rows(primary_rows, lambda r: r.get("objective_outcome") == "rgb_only_correct", 40)],
        "objective_both_wrong": [slim_sample(r) for r in sample_rows(primary_rows, lambda r: r.get("objective_outcome") == "both_wrong", 40)],
        "judgelm_ours_higher": [slim_sample(r) for r in sample_rows(primary_rows, lambda r: r.get("is_judgelm") and (r.get("judgelm_delta") or 0) > 0, 40, sort_key=lambda r: -float(r.get("judgelm_delta") or 0))],
        "judgelm_rgb_higher": [slim_sample(r) for r in sample_rows(primary_rows, lambda r: r.get("is_judgelm") and (r.get("judgelm_delta") or 0) < 0, 40, sort_key=lambda r: float(r.get("judgelm_delta") or 0))],
    }

    objective_error_rows = [r for r in primary_rows if r.get("is_objective") and r.get("objective_outcome") != "both_correct"]
    objective_side_by_side_all = [
        side_by_side_objective(r, str(r.get("objective_outcome") or ""))
        for r in sorted(objective_error_rows, key=lambda r: (str(r.get("objective_outcome")), str(r.get("metric_class")), stable_order(r)))
    ]
    objective_side_by_side_selected: list[dict[str, Any]] = []
    for group in ("rgb_only_correct", "ours_only_correct", "both_wrong"):
        group_rows = [r for r in objective_error_rows if r.get("objective_outcome") == group]
        objective_side_by_side_selected.extend(side_by_side_objective(r, group) for r in pick_representative(group_rows, per_group=18))

    judgelm_side_by_side_selected = [
        side_by_side_judgelm(r, "v18_much_lower")
        for r in sample_rows(primary_rows, lambda r: r.get("is_judgelm") and (r.get("judgelm_delta") or 0) <= -0.25, 18, sort_key=lambda r: float(r.get("judgelm_delta") or 0))
    ] + [
        side_by_side_judgelm(r, "rgb_much_lower")
        for r in sample_rows(primary_rows, lambda r: r.get("is_judgelm") and (r.get("judgelm_delta") or 0) >= 0.25, 18, sort_key=lambda r: -float(r.get("judgelm_delta") or 0))
    ]

    pair_pattern_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    method_error_counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in objective_side_by_side_all:
        pair_pattern_counts[
            (
                str(row.get("group")),
                str(row.get("metric_class")),
                str(row.get("category")),
                str(row.get("v18_error")),
                str(row.get("rgb_error")),
            )
        ] += 1
        if row.get("v18_error") != "correct":
            method_error_counts[("v18", str(row.get("metric_class")), str(row.get("category")), str(row.get("v18_error")))] += 1
        if row.get("rgb_error") != "correct":
            method_error_counts[("rgb", str(row.get("metric_class")), str(row.get("category")), str(row.get("rgb_error")))] += 1

    bad_case_pair_patterns = [
        {
            "group": group,
            "metric_class": metric,
            "category": category,
            "v18_error": v18_error,
            "rgb_error": rgb_error,
            "count": count,
        }
        for (group, metric, category, v18_error, rgb_error), count in pair_pattern_counts.most_common()
    ]
    bad_case_method_error_patterns = [
        {
            "method": method,
            "metric_class": metric,
            "category": category,
            "error": error,
            "count": count,
        }
        for (method, metric, category, error), count in method_error_counts.most_common()
    ]

    category_bias_notes = []
    for rec in sorted(objective_by_category, key=lambda r: float(r.get("delta_acc") or 0), reverse=True):
        if int(rec.get("n") or 0) >= 20:
            category_bias_notes.append(rec)
    category_loss_notes = sorted([r for r in objective_by_category if int(r.get("n") or 0) >= 20], key=lambda r: float(r.get("delta_acc") or 0))
    scene_bias_notes = sorted([r for r in objective_by_scene if int(r.get("n") or 0) >= 20], key=lambda r: float(r.get("delta_acc") or 0), reverse=True)
    scene_loss_notes = sorted([r for r in objective_by_scene if int(r.get("n") or 0) >= 20], key=lambda r: float(r.get("delta_acc") or 0))
    jl_gain_notes = sorted([r for r in judgelm_by_category if int(r.get("n") or 0) >= 5], key=lambda r: float(r.get("delta_score") or 0), reverse=True)
    jl_loss_notes = sorted([r for r in judgelm_by_category if int(r.get("n") or 0) >= 5], key=lambda r: float(r.get("delta_score") or 0))

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "experiment_overview.csv", all_exp_overview)
    write_csv(output / "objective_by_category.csv", objective_by_category)
    write_csv(output / "objective_by_scene.csv", objective_by_scene)
    write_csv(output / "objective_by_scene_category.csv", objective_by_scene_category)
    write_csv(output / "judgelm_by_category.csv", judgelm_by_category)
    write_csv(output / "judgelm_by_scene.csv", judgelm_by_scene)
    write_csv(output / "judgelm_by_scene_category.csv", judgelm_by_scene_category)
    write_csv(output / "bad_case_side_by_side_all.csv", objective_side_by_side_all)
    write_csv(output / "bad_case_side_by_side_selected.csv", objective_side_by_side_selected)
    write_csv(output / "bad_case_judgelm_side_by_side_selected.csv", judgelm_side_by_side_selected)
    write_csv(output / "bad_case_pair_patterns.csv", bad_case_pair_patterns)
    write_csv(output / "bad_case_method_error_patterns.csv", bad_case_method_error_patterns)
    for name, rows in sample_sets.items():
        write_csv(output / f"{name}.csv", rows)

    primary_summary = next(r for r in all_exp_overview if r["experiment_id"] == args.primary)

    objective_category_min = min((r for r in objective_by_category if int(r.get("n") or 0) >= 10), key=lambda r: float(r.get("delta_acc") or 0), default=None)
    objective_category_max = max((r for r in objective_by_category if int(r.get("n") or 0) >= 10), key=lambda r: float(r.get("delta_acc") or 0), default=None)
    objective_scene_cat_gain = max((r for r in objective_by_scene_category if int(r.get("n") or 0) >= 10), key=lambda r: float(r.get("delta_acc") or 0), default=None)
    objective_scene_cat_loss = min((r for r in objective_by_scene_category if int(r.get("n") or 0) >= 10), key=lambda r: float(r.get("delta_acc") or 0), default=None)
    judgelm_category_gain = max((r for r in judgelm_by_category if int(r.get("n") or 0) >= 5), key=lambda r: float(r.get("delta_score") or 0), default=None)
    judgelm_category_loss = min((r for r in judgelm_by_category if int(r.get("n") or 0) >= 5), key=lambda r: float(r.get("delta_score") or 0), default=None)

    interpretation_bullets = [
        f"Objective 总体仍是 RGB 更强：{short_exp(args.primary)} 为 {pct(primary_summary['ours_obj_acc'])}，RGB 为 {pct(primary_summary['rgb_obj_acc'])}，差 {pp(primary_summary['delta_obj_acc'])}；逐题上 V18-only {primary_summary['ours_only_correct']} 题，RGB-only {primary_summary['rgb_only_correct']} 题。",
        "按大类看，V18 没有出现稳定超过 RGB 的 Objective 大类；优势主要是少数 scene×question-type 切片，而不是全局能力提升。",
    ]
    if objective_scene_cat_gain:
        interpretation_bullets.append(
            f"当前最清楚的 V18 正切片是 {objective_scene_cat_gain.get('scene')} / {objective_scene_cat_gain.get('category')} / {objective_scene_cat_gain.get('metric_class')}：Δ {pp(float(objective_scene_cat_gain.get('delta_acc') or 0))}，但要注意这是切片优势。"
        )
    if objective_category_min:
        interpretation_bullets.append(
            f"最需要修的是 {objective_category_min.get('category')} / {objective_category_min.get('metric_class')}：V18 {pct(float(objective_category_min.get('ours_acc') or 0))} vs RGB {pct(float(objective_category_min.get('rgb_acc') or 0))}，RGB-only {objective_category_min.get('rgb_only_correct')} 题。"
        )
    if objective_scene_cat_loss:
        interpretation_bullets.append(
            f"场景切片里最大负差是 {objective_scene_cat_loss.get('scene')} / {objective_scene_cat_loss.get('category')} / {objective_scene_cat_loss.get('metric_class')}：Δ {pp(float(objective_scene_cat_loss.get('delta_acc') or 0))}。"
        )
    if judgelm_category_gain and judgelm_category_loss:
        interpretation_bullets.append(
            f"JudgeLM 画像和 Objective 不完全一致：{judgelm_category_gain.get('category')} 上 V18 更高（Δ {num(float(judgelm_category_gain.get('delta_score') or 0))}），但 {judgelm_category_loss.get('category')} 明显输给 RGB（Δ {num(float(judgelm_category_loss.get('delta_score') or 0))}）。"
        )
    interpretation_bullets.append(
        "初步方向：V18 的 3D token 更像提供 landmark/全局语义先验；短板在可见部件定位、可数实例、细粒度空间/部件 grounding。这更支持回到 feature extraction / training，让 3D feature 保留小部件和实例边界，而不是继续只调 post-training sampling。"
    )

    summary = {
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "primary_experiment_id": args.primary,
        "primary_experiment_short": short_exp(args.primary),
        "compared_experiment_ids": exp_ids,
        "validation": validation,
        "primary_overview": primary_summary,
        "interpretation_bullets": interpretation_bullets,
        "strongest_objective_gain_categories": category_bias_notes[:8],
        "strongest_objective_loss_categories": category_loss_notes[:8],
        "strongest_objective_gain_scenes": scene_bias_notes[:8],
        "strongest_objective_loss_scenes": scene_loss_notes[:8],
        "strongest_judgelm_gain_categories": jl_gain_notes[:8],
        "strongest_judgelm_loss_categories": jl_loss_notes[:8],
        "outputs": {
            "experiment_overview_csv": str(output / "experiment_overview.csv"),
            "objective_by_category_csv": str(output / "objective_by_category.csv"),
            "objective_by_scene_csv": str(output / "objective_by_scene.csv"),
            "judgelm_by_category_csv": str(output / "judgelm_by_category.csv"),
            "bad_case_side_by_side_all_csv": str(output / "bad_case_side_by_side_all.csv"),
            "bad_case_side_by_side_selected_csv": str(output / "bad_case_side_by_side_selected.csv"),
            "bad_case_judgelm_side_by_side_selected_csv": str(output / "bad_case_judgelm_side_by_side_selected.csv"),
            "bad_case_pair_patterns_csv": str(output / "bad_case_pair_patterns.csv"),
            "bad_case_method_error_patterns_csv": str(output / "bad_case_method_error_patterns.csv"),
            "samples": {name: str(output / f"{name}.csv") for name in sample_sets},
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines: list[str] = []
    report_lines.append("# V18 vs RGB 逐题误差画像 / Benchmark Sampling Diagnostic\n")
    report_lines.append(f"生成时间：`{summary['generated_at']}`  ")
    report_lines.append(f"主分析方法：`{short_exp(args.primary)}` (`{args.primary}`)  ")
    report_lines.append("范围：8 scenes / 74 images / 1061 VQA；Objective 764 题、JudgeLM 297 题。\n")
    report_lines.append("## 0. 读法\n")
    report_lines.append("- **ours_only_correct**：V18 答对、RGB 答错，是我们方法相对 RGB 的正样本。\n- **rgb_only_correct**：RGB 答对、V18 答错，是当前方法需要解释/修复的负样本。\n- JudgeLM 同时给 exact 高低与 `±0.25` margin；`±0.25` 内先当近似持平，不强行过度解释。\n")
    report_lines.append("## 1. 方法总览\n")
    report_lines.append(markdown_table(all_exp_overview, [
        ("experiment_short", "method"), ("objective_n", "Obj n"), ("ours_obj_acc", "Obj ours"), ("rgb_obj_acc", "Obj RGB"), ("delta_obj_acc", "Δ Obj"),
        ("ours_only_correct", "ours-only"), ("rgb_only_correct", "rgb-only"), ("both_wrong", "both-wrong"),
        ("judgelm_n", "JL n"), ("ours_judgelm", "JL ours"), ("rgb_judgelm", "JL RGB"), ("delta_judgelm", "Δ JL"),
    ]))
    report_lines.append("## 1.5 关键解释 / Bias hint\n")
    report_lines.extend([f"- {line}\n" for line in interpretation_bullets])
    report_lines.append("## 2. Objective：按问题类别看 ours/RGB 各自优势\n")
    report_lines.append(markdown_table(sorted(objective_by_category, key=lambda r: float(r.get("delta_acc") or 0), reverse=True), [
        ("category", "category"), ("metric_class", "metric"), ("n", "n"), ("ours_acc", "ours"), ("rgb_acc", "RGB"), ("delta_acc", "Δ"),
        ("ours_only_correct", "ours-only"), ("rgb_only_correct", "rgb-only"), ("both_wrong", "both-wrong"),
    ]))
    report_lines.append("## 3. Objective：按场景看偏差\n")
    report_lines.append(markdown_table(sorted(objective_by_scene, key=lambda r: float(r.get("delta_acc") or 0), reverse=True), [
        ("scene", "scene"), ("metric_class", "metric"), ("n", "n"), ("ours_acc", "ours"), ("rgb_acc", "RGB"), ("delta_acc", "Δ"),
        ("ours_only_correct", "ours-only"), ("rgb_only_correct", "rgb-only"), ("both_wrong", "both-wrong"),
    ]))
    report_lines.append("## 4. JudgeLM：按问题类别看描述质量偏差\n")
    report_lines.append(markdown_table(sorted(judgelm_by_category, key=lambda r: float(r.get("delta_score") or 0), reverse=True), [
        ("category", "category"), ("n", "n"), ("ours_score", "ours"), ("rgb_score", "RGB"), ("delta_score", "Δ"),
        ("ours_higher_025", "ours≥+0.25"), ("rgb_higher_025", "RGB≥+0.25"), ("near_tie_025", "near tie"),
    ]))
    report_lines.append("## 5. 同题 side-by-side：坏例里两边分别答成什么样\n")
    report_lines.append("### 5.1 错误模式计数（Objective）\n")
    report_lines.append(markdown_table(bad_case_pair_patterns, [
        ("group", "group"), ("metric_class", "metric"), ("category", "category"), ("v18_error", "V18 error"), ("rgb_error", "RGB error"), ("count", "count"),
    ], limit=18))
    for title, group in [
        ("5.2 RGB 对、V18 错：V18 坏例", "rgb_only_correct"),
        ("5.3 V18 对、RGB 错：RGB 坏例", "ours_only_correct"),
        ("5.4 两边都错：共同盲点", "both_wrong"),
    ]:
        rows = [r for r in objective_side_by_side_selected if r.get("group") == group]
        report_lines.append(f"### {title}\n")
        report_lines.append(markdown_table(rows, [
            ("scene", "scene"), ("metric_class", "metric"), ("category", "category"), ("question", "question"),
            ("expected", "expected"), ("expected_option_text", "expected option"), ("v18_answer", "V18 answer"), ("v18_option_text", "V18 option"),
            ("v18_error", "V18 error"), ("rgb_answer", "RGB answer"), ("rgb_option_text", "RGB option"), ("rgb_error", "RGB error"),
        ], limit=18))
    report_lines.append("### 5.5 JudgeLM 大差异：两边回答文本对照\n")
    report_lines.append(markdown_table(judgelm_side_by_side_selected, [
        ("group", "group"), ("scene", "scene"), ("category", "category"), ("question", "question"), ("expected", "expected"),
        ("v18_answer", "V18 answer"), ("v18_score", "V18 score"), ("rgb_answer", "RGB answer"), ("rgb_score", "RGB score"), ("delta_judgelm", "ΔJL"),
    ], limit=24))
    report_lines.append("## 6. 原始样例抽样\n")
    for title, key in [
        ("6.1 V18 对、RGB 错（ours-only objective）", "objective_ours_only_correct"),
        ("6.2 RGB 对、V18 错（rgb-only objective）", "objective_rgb_only_correct"),
        ("6.3 两者都错（objective both-wrong）", "objective_both_wrong"),
        ("6.4 JudgeLM：V18 明显更高", "judgelm_ours_higher"),
        ("6.5 JudgeLM：RGB 明显更高", "judgelm_rgb_higher"),
    ]:
        report_lines.append(f"### {title}\n")
        report_lines.append(markdown_table(sample_sets[key], [
            ("scene", "scene"), ("metric_class", "metric"), ("category", "category"), ("question", "question"), ("expected", "expected"),
            ("ours_answer", "ours"), ("rgb_answer", "RGB"), ("ours_correct", "ours ok"), ("rgb_correct", "RGB ok"), ("judgelm_delta", "ΔJL"),
        ], limit=12))
    report_lines.append("## 7. 文件\n")
    report_lines.append("- `summary.json`：机器可读摘要。\n- `objective_by_category.csv`, `objective_by_scene.csv`, `objective_by_scene_category.csv`：Objective 分解。\n- `judgelm_by_category.csv`, `judgelm_by_scene.csv`, `judgelm_by_scene_category.csv`：JudgeLM 分解。\n- `bad_case_side_by_side_all.csv`, `bad_case_side_by_side_selected.csv`：同题 Expected / V18 / RGB 坏例对照。\n- `bad_case_pair_patterns.csv`, `bad_case_method_error_patterns.csv`：错误模式计数。\n- `objective_ours_only_correct.csv`, `objective_rgb_only_correct.csv`, `objective_both_wrong.csv`, `judgelm_ours_higher.csv`, `judgelm_rgb_higher.csv`：逐题抽样。\n")
    report_md = "\n".join(report_lines)
    (output / "report.md").write_text(report_md, encoding="utf-8")

    css = """
body{margin:0;padding:24px;background:linear-gradient(180deg,#f6f1e9,#ede6db);color:#161311;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif}.wrap{max-width:1600px;margin:0 auto}.nav{display:flex;gap:10px;flex-wrap:wrap}.pill{display:inline-block;padding:7px 13px;border:1px solid rgba(22,19,17,.12);border-radius:999px;background:#fff;color:#9f3f1c;text-decoration:none;font-weight:850}.panel{background:rgba(255,251,245,.94);border:1px solid rgba(22,19,17,.12);border-radius:16px;padding:16px;margin:14px 0;box-shadow:0 12px 32px rgba(44,32,22,.08)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.card{background:#fffaf4;border:1px solid rgba(22,19,17,.12);border-radius:14px;padding:14px}.k{font-size:12px;color:#665d55}.v{font-size:23px;font-weight:950}.muted{color:#665d55}.table-wrap{overflow-x:auto;border:1px solid rgba(22,19,17,.12);border-radius:14px;background:#fffaf4}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:9px 10px;border-bottom:1px solid rgba(22,19,17,.12);vertical-align:top;text-align:left}th{background:#fff7ed;color:#665d55;font-size:11px;text-transform:uppercase;letter-spacing:.04em}code{background:#fff;padding:2px 5px;border-radius:6px}h1{margin-bottom:4px}.callout{border-left:4px solid #b84d24}.good{color:#2e7d32}.bad{color:#c62828}.warn{color:#8e4d17}
"""
    html_parts = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "<title>V18 vs RGB 逐题误差画像</title><style>", css, "</style></head><body><div class='wrap'>",
        "<div class='nav'><a class='pill' href='/projects/langsplat'>← V18 实验进度</a><a class='pill' href='/projects/langsplat/v18-glossary'>命名说明</a></div>",
        f"<h1>V18 vs RGB 逐题误差画像</h1><p class='muted'>主方法：<code>{html.escape(short_exp(args.primary))}</code>；范围：8 scenes / 74 images / 1061 VQA。</p>",
        "<section class='panel callout'><b>目的：</b>不是继续报告平均值，而是看哪些题 RGB 赢、哪些题 V18 赢，用来判断后续算法调整方向和 benchmark bias。</section>",
        "<div class='cards'>",
        f"<div class='card'><div class='k'>Objective ours</div><div class='v'>{pct(primary_summary['ours_obj_acc'])}</div></div>",
        f"<div class='card'><div class='k'>Objective RGB</div><div class='v'>{pct(primary_summary['rgb_obj_acc'])}</div></div>",
        f"<div class='card'><div class='k'>Δ Objective</div><div class='v {'good' if primary_summary['delta_obj_acc']>0 else 'bad'}'>{pp(primary_summary['delta_obj_acc'])}</div></div>",
        f"<div class='card'><div class='k'>ours-only / rgb-only</div><div class='v'>{primary_summary['ours_only_correct']} / {primary_summary['rgb_only_correct']}</div></div>",
        f"<div class='card'><div class='k'>JudgeLM ours</div><div class='v'>{num(primary_summary['ours_judgelm'])}</div></div>",
        f"<div class='card'><div class='k'>Δ JudgeLM</div><div class='v {'good' if primary_summary['delta_judgelm']>0 else 'bad'}'>{num(primary_summary['delta_judgelm'])}</div></div>",
        "</div>",
        "<section class='panel callout'><h2>关键解释 / Bias hint</h2><ul>" + "".join(f"<li>{html.escape(line)}</li>" for line in interpretation_bullets) + "</ul></section>",
        "<section class='panel'><h2>1. 方法总览</h2>",
        html_table(all_exp_overview, [("experiment_short", "method"), ("ours_obj_acc", "Obj ours"), ("rgb_obj_acc", "Obj RGB"), ("delta_obj_acc", "Δ Obj"), ("ours_only_correct", "ours-only"), ("rgb_only_correct", "rgb-only"), ("both_wrong", "both-wrong"), ("ours_judgelm", "JL ours"), ("rgb_judgelm", "JL RGB"), ("delta_judgelm", "Δ JL")]),
        "</section><section class='panel'><h2>2. Objective 按问题类别</h2>",
        html_table(sorted(objective_by_category, key=lambda r: float(r.get("delta_acc") or 0), reverse=True), [("category", "category"), ("metric_class", "metric"), ("n", "n"), ("ours_acc", "ours"), ("rgb_acc", "RGB"), ("delta_acc", "Δ"), ("ours_only_correct", "ours-only"), ("rgb_only_correct", "rgb-only"), ("both_wrong", "both-wrong")]),
        "</section><section class='panel'><h2>3. Objective 按场景</h2>",
        html_table(sorted(objective_by_scene, key=lambda r: float(r.get("delta_acc") or 0), reverse=True), [("scene", "scene"), ("metric_class", "metric"), ("n", "n"), ("ours_acc", "ours"), ("rgb_acc", "RGB"), ("delta_acc", "Δ"), ("ours_only_correct", "ours-only"), ("rgb_only_correct", "rgb-only"), ("both_wrong", "both-wrong")]),
        "</section><section class='panel'><h2>4. JudgeLM 按问题类别</h2>",
        html_table(sorted(judgelm_by_category, key=lambda r: float(r.get("delta_score") or 0), reverse=True), [("category", "category"), ("n", "n"), ("ours_score", "ours"), ("rgb_score", "RGB"), ("delta_score", "Δ"), ("ours_higher_025", "ours≥+0.25"), ("rgb_higher_025", "RGB≥+0.25"), ("near_tie_025", "near tie")]),
        "</section>",
        "<section class='panel'><h2>5. 同题 side-by-side：坏例里两边分别答成什么样</h2><p class='muted'>这里是最关键的阅读区：同一道题同时列 Expected、V18 answer、RGB answer 和错误类型。</p>",
        "<h3>5.1 错误模式计数</h3>",
        html_table(bad_case_pair_patterns, [("group", "group"), ("metric_class", "metric"), ("category", "category"), ("v18_error", "V18 error"), ("rgb_error", "RGB error"), ("count", "count")], limit=18),
        "</section>",
    ]
    for title, group in [
        ("5.2 RGB 对、V18 错：V18 坏例", "rgb_only_correct"),
        ("5.3 V18 对、RGB 错：RGB 坏例", "ours_only_correct"),
        ("5.4 两边都错：共同盲点", "both_wrong"),
    ]:
        rows = [r for r in objective_side_by_side_selected if r.get("group") == group]
        html_parts.extend([
            f"<section class='panel'><h2>{html.escape(title)}</h2>",
            html_table(rows, [("scene", "scene"), ("metric_class", "metric"), ("category", "category"), ("question", "question"), ("expected", "expected"), ("expected_option_text", "expected option"), ("v18_answer", "V18 answer"), ("v18_option_text", "V18 option"), ("v18_error", "V18 error"), ("rgb_answer", "RGB answer"), ("rgb_option_text", "RGB option"), ("rgb_error", "RGB error")], limit=18),
            "</section>",
        ])
    html_parts.extend([
        "<section class='panel'><h2>5.5 JudgeLM 大差异：回答文本对照</h2>",
        html_table(judgelm_side_by_side_selected, [("group", "group"), ("scene", "scene"), ("category", "category"), ("question", "question"), ("expected", "expected"), ("v18_answer", "V18 answer"), ("v18_score", "V18 score"), ("rgb_answer", "RGB answer"), ("rgb_score", "RGB score"), ("delta_judgelm", "ΔJL")], limit=24),
        "</section>",
    ])
    sample_titles = {
        "objective_ours_only_correct": "6.1 V18 对、RGB 错（原始样例）",
        "objective_rgb_only_correct": "6.2 RGB 对、V18 错（原始样例）",
        "objective_both_wrong": "6.3 两者都错（原始样例）",
        "judgelm_ours_higher": "6.4 JudgeLM：V18 更高",
        "judgelm_rgb_higher": "6.5 JudgeLM：RGB 更高",
    }
    for key, title in sample_titles.items():
        html_parts.extend([
            f"<section class='panel'><h2>{html.escape(title)}</h2>",
            html_table(sample_sets[key], [("scene", "scene"), ("metric_class", "metric"), ("category", "category"), ("question", "question"), ("expected", "expected"), ("ours_answer", "ours"), ("rgb_answer", "RGB"), ("ours_correct", "ours ok"), ("rgb_correct", "RGB ok"), ("judgelm_delta", "ΔJL")], limit=12),
            "</section>",
        ])
    html_parts.extend([
        f"<section class='panel'><h2>输出文件</h2><p><code>{html.escape(str(output))}</code></p></section>",
        "</div></body></html>",
    ])
    (output / "index.html").write_text("".join(html_parts), encoding="utf-8")

    print(json.dumps({"status": "ok", "output": str(output), "primary": args.primary, "summary": primary_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

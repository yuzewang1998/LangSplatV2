#!/usr/bin/env python3
"""Run V18 full-benchmark sampling/token-fusion experiments.

This runner supports the first V18 executable batches:
- 2D-only sampling baselines: u2d / rep2d / norm2d
- 3D-only sampling baselines: geo3d / feat3d / opa3d / mix3d
- token-level 2D+3D fusion pipelines: tokpipe

Each experiment is resumable at image granularity. It writes raw answers under:
experiment_results/benchmark/v18_pipeline_ablation/results/<exp_id>/outputs/<scene>/

Scoring/JudgeLM is handled by score_v18_results.py so that every completed run
updates the 9999 page through summary.json / question_metrics.csv.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
LLAVA = ROOT / "LLaVA-NeXT"
sys.path.insert(0, str(LLAVA))

from llava.model.builder import load_pretrained_model  # type: ignore
from llava.mm_utils import get_model_name_from_path  # type: ignore
from verify_reconstruction_quality import answer_question  # type: ignore

DEFAULT_16A = Path("/mnt/data/wangyz/exp_results/historicalAgent/eval_results/9999/full_benchmark_16a")
DEFAULT_CKPT = Path("/mnt/data/wangyz/exp_results/historicalAgent/output/9999/full_benchmark_16a")
DEFAULT_V18 = ROOT / "experiment_results" / "benchmark" / "v18_pipeline_ablation"
SCALES = [("level_0", "Small", 0), ("level_1", "Medium", 1), ("level_2", "Large", 2)]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(root: Path) -> dict[str, Any]:
    path = root / "v18_registry.json"
    if not path.exists():
        raise FileNotFoundError(f"missing V18 registry: {path}")
    return read_json(path)


def split_budget(total: int, parts: int = 3) -> list[int]:
    base = total // parts
    out = [base] * parts
    for i in range(total - base * parts):
        out[i] += 1
    return out


def load_qas(root: Path, scene: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for p in sorted((root / scene).glob("**/analysis_no_rag/*_qa_results.json")):
        data = read_json(p)
        image = str(data.get("image_name") or p.name.removesuffix("_qa_results.json"))
        out[image] = data.get("questions", []) or []
    return out


def discover_scenes(root: Path) -> list[str]:
    scenes = []
    for p in sorted(root.iterdir() if root.exists() else []):
        if p.is_dir() and list(p.glob("**/analysis_no_rag/*_qa_results.json")):
            scenes.append(p.name)
    if not scenes:
        raise FileNotFoundError(f"no benchmark scenes discovered under {root}")
    return scenes


def feature_map_path(root: Path, scene: str, image: str, level: int) -> Path:
    p = root / scene / f"9999_16A_{scene}" / f"level{level}" / scene / image / f"feature_map_{image}.pt"
    if p.exists():
        return p
    hits = sorted((root / scene).glob(f"**/level{level}/{scene}/{image}/feature_map_{image}.pt"))
    if hits:
        return hits[0]
    raise FileNotFoundError(p)


def ckpt_path(root: Path, scene: str, level: int) -> Path:
    base = root / scene
    patterns = [
        f"{scene}_9999_16A_{scene}_{level}",
        f"{scene}_9999_16A_{scene}_{level}_{level}",
        f"{scene}_9999_16A_{scene}_{level}_*",
    ]
    for pattern in patterns:
        hits = sorted(base.glob(pattern + "/chkpnt10000.pth")) if "*" in pattern else [base / pattern / "chkpnt10000.pth"]
        for hit in hits:
            if hit.exists():
                return hit
    raise FileNotFoundError(f"no ckpt for {scene} level {level} under {base}")


def load_level_ckpt(path: Path):
    model, _it = torch.load(path, map_location="cpu", weights_only=False)
    xyz = model[1].detach().float()
    opacity = model[6].detach().float().reshape(-1)
    logits = model[7].detach().float()
    codebook = model[8].detach().float()[0]
    return xyz, opacity, logits, codebook


def fps(signature: torch.Tensor, k: int) -> torch.Tensor:
    n = signature.shape[0]
    if k >= n:
        return torch.arange(n)
    sig = F.normalize(signature.float(), dim=1, eps=1e-6)
    chosen = torch.empty(k, dtype=torch.long)
    centroid = sig.mean(0, keepdim=True)
    idx = int(torch.argmax(torch.cdist(sig, centroid).squeeze(1)))
    chosen[0] = idx
    min_d = torch.cdist(sig, sig[idx : idx + 1]).squeeze(1)
    for i in range(1, k):
        idx = int(torch.argmax(min_d))
        chosen[i] = idx
        d = torch.cdist(sig, sig[idx : idx + 1]).squeeze(1)
        min_d = torch.minimum(min_d, d)
    return chosen


def sparse_code_signature(logits: torch.Tensor, topk: int = 4) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    vals, inds = torch.topk(probs, min(topk, probs.shape[1]), dim=1)
    sig = torch.zeros_like(probs)
    sig.scatter_(1, inds, vals)
    return sig / (sig.sum(1, keepdim=True) + 1e-9)


def sample_3d(xyz: torch.Tensor, opacity: torch.Tensor, logits: torch.Tensor, codebook: torch.Tensor, k: int, sampler: str, pool_factor: int, topk: int):
    n = logits.shape[0]
    k = min(max(1, k), n)
    op = torch.sigmoid(opacity)
    pool = min(n, max(k, k * pool_factor))
    cand = torch.topk(op, pool).indices
    if sampler == "geo3d":
        idx = cand[fps(xyz[cand], k)]
    elif sampler == "feat3d":
        sig = sparse_code_signature(logits[cand], topk=topk)
        idx = cand[fps(sig, k)]
    elif sampler == "opa3d":
        idx = torch.topk(op, k).indices
    elif sampler == "mix3d":
        k_geo = k // 2
        idx_geo = cand[fps(xyz[cand], k_geo)] if k_geo else torch.empty(0, dtype=torch.long)
        sig = sparse_code_signature(logits[cand], topk=topk)
        idx_feat = cand[fps(sig, k - k_geo)]
        idx = torch.unique(torch.cat([idx_geo, idx_feat]))[:k]
        if idx.numel() < k:
            idx = torch.cat([idx, cand[: k - idx.numel()]])
    else:
        raise ValueError(f"unsupported 3D sampler: {sampler}")
    sparse = sparse_code_signature(logits[idx], topk=topk)
    feats = sparse @ codebook
    return feats.contiguous(), idx.cpu().tolist()


def sample_2d(path: Path, k: int, sampler: str):
    t = torch.load(path, map_location="cpu")
    fmap = t[0] if t.dim() == 4 else t
    h, w, c = fmap.shape
    flat = fmap.reshape(-1, c).float()
    n = flat.shape[0]
    k = min(max(1, k), n)
    if sampler == "u2d":
        idx = torch.linspace(0, n - 1, k).long()
    elif sampler == "rep2d":
        # FPS on a strided feature signature to keep CPU memory bounded.
        sig = flat[:, :: max(1, c // 64)]
        idx = fps(sig, k)
    elif sampler == "norm2d":
        idx = torch.topk(torch.linalg.vector_norm(flat, dim=1), k).indices
    else:
        raise ValueError(f"unsupported 2D sampler: {sampler}")
    feats = flat[idx].contiguous()
    meta = {"source": str(path), "h": h, "w": w, "c": c, "candidates": n, "selected": int(feats.shape[0]), "sampler": sampler, "indices": idx.cpu().tolist()[:128]}
    del t, fmap, flat
    gc.collect()
    return feats, meta


def parse_budget(fields: dict[str, str]) -> tuple[int, int, int]:
    total = int(str(fields.get("T", "T512")).removeprefix("T"))
    ratio = int(str(fields.get("R", "R0")).removeprefix("R"))
    calls = int(str(fields.get("C", "C1")).removeprefix("C"))
    return total, ratio, calls


def sampler_from_exp_id(exp_id: str, pipe: str) -> str | None:
    parts = exp_id.split("_")
    try:
        if pipe == "2donly":
            return parts[2]
        if pipe == "3donly":
            return parts[3]
    except Exception:
        return None
    return None


def resolve_best_sampler(v18_root: Path, pipe: str, fallback: str) -> str:
    """Resolve best2d/best3d from completed Phase-A summaries.

    Objective is the primary gate because V18's first decision is a sampling
    ablation under the same LLaVA answer path. JudgeLM remains reported, but
    the automatic scheduler needs one deterministic winner to keep GPUs busy.
    """
    prefix = "v18_2donly_" if pipe == "2donly" else "v18_3donly_"
    best_sampler = fallback
    best_score = float("-inf")
    results_root = v18_root / "results"
    for summary_path in sorted(results_root.glob(prefix + "*/summary.json")):
        try:
            data = read_json(summary_path)
        except Exception:
            continue
        if data.get("status") != "completed":
            continue
        score = data.get("objective_accuracy")
        if score is None:
            continue
        sampler = sampler_from_exp_id(str(data.get("experiment_id") or summary_path.parent.name), pipe)
        if sampler and float(score) > best_score:
            best_score = float(score)
            best_sampler = sampler
    return best_sampler


def resolve_fields(exp: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    fields = dict(exp.get("fields", {}) or {})
    if fields.get("s2d") == "best2d":
        fields["s2d"] = resolve_best_sampler(args.v18_root, "2donly", "rep2d")
    if fields.get("s3d") == "best3d":
        fields["s3d"] = resolve_best_sampler(args.v18_root, "3donly", "mix3d")
    return fields


def build_2d_scale_tokens(scene: str, image: str, args: argparse.Namespace, sampler: str, budgets: list[int]):
    tokens = []
    meta: dict[str, Any] = {}
    for (_lk, scale_name, level), k in zip(SCALES, budgets):
        feats, m = sample_2d(feature_map_path(args.sixteen_a_root, scene, image, level), k, sampler)
        tokens.append(feats)
        meta[scale_name] = m
    return tokens, meta


def build_3d_scale_tokens(ckpts: dict[int, Any], args: argparse.Namespace, sampler: str, budgets: list[int]):
    tokens = []
    meta: dict[str, Any] = {}
    for (level_key, _scale_name, level), k in zip(SCALES, budgets):
        xyz, opacity, logits, codebook = ckpts[level]
        feats, indices = sample_3d(xyz, opacity, logits, codebook, k, sampler, args.pool_factor, args.topk)
        tokens.append(feats)
        meta[level_key] = {"sampler": sampler, "total_gaussians": int(logits.shape[0]), "selected": int(feats.shape[0]), "indices": indices[:128]}
    return tokens, meta


def fuse_tokpipe(tokens_2d: list[torch.Tensor], tokens_3d: list[torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "mmtokscale":
        return torch.cat([torch.cat([a, b], dim=0) for a, b in zip(tokens_2d, tokens_3d)], dim=0)
    if mode == "mmtok2d3d":
        return torch.cat(tokens_2d + tokens_3d, dim=0)
    if mode == "mmtok3d2d":
        return torch.cat(tokens_3d + tokens_2d, dim=0)
    if mode == "mmtokinter":
        chunks: list[torch.Tensor] = []
        for a, b in zip(tokens_2d, tokens_3d):
            chunks.extend([a, b])
        return torch.cat(chunks, dim=0)
    if mode == "mmtok3dgate":
        # Lightweight non-learned gate: expose 3D anchors first at each scale,
        # then the 2D tokens. This tests whether 3D context should condition the
        # following 2D visual tokens without introducing question/scene bias.
        return torch.cat([torch.cat([b, a], dim=0) for a, b in zip(tokens_2d, tokens_3d)], dim=0)
    raise ValueError(f"unsupported token-level mm fusion: {mode}")


def build_tokens_for_image(exp: dict[str, Any], scene: str, image: str, args: argparse.Namespace, ckpts: dict[int, Any]):
    fields = resolve_fields(exp, args)
    pipe = fields.get("pipe")
    s2d = fields.get("s2d")
    s3d = fields.get("s3d")
    total, ratio, _calls = parse_budget(fields)
    if pipe == "2donly":
        per_level = split_budget(total, 3)
        meta = {"2d": {}, "3d": {}, "fusion": "m2tokcat/mmnone"}
        tokens, meta["2d"] = build_2d_scale_tokens(scene, image, args, s2d, per_level)
        fused = torch.cat(tokens, dim=0)
        image_size = (int(meta["2d"]["Large"]["w"]), int(meta["2d"]["Large"]["h"]))
        return fused, image_size, meta
    if pipe == "3donly":
        per_level = split_budget(total, 3)
        meta = {"2d": {}, "3d": {}, "fusion": "m3tokcat/mmnone"}
        tokens, meta["3d"] = build_3d_scale_tokens(ckpts, args, s3d, per_level)
        fused = torch.cat(tokens, dim=0)
        # Use Large 2D feature-map size only as image_size metadata for LLaVA helper.
        fpath = feature_map_path(args.sixteen_a_root, scene, image, 2)
        t = torch.load(fpath, map_location="cpu")
        fmap = t[0] if t.dim() == 4 else t
        image_size = (int(fmap.shape[1]), int(fmap.shape[0]))
        del t, fmap
        return fused, image_size, meta
    if pipe == "tokpipe":
        total_3d = int(round(total * ratio / 100.0))
        total_2d = total - total_3d
        budgets_2d = split_budget(total_2d, 3)
        budgets_3d = split_budget(total_3d, 3)
        tokens_2d, meta_2d = build_2d_scale_tokens(scene, image, args, s2d, budgets_2d)
        tokens_3d, meta_3d = build_3d_scale_tokens(ckpts, args, s3d, budgets_3d)
        mm = fields.get("mm", "mmtokscale")
        fused = fuse_tokpipe(tokens_2d, tokens_3d, mm)
        fpath = feature_map_path(args.sixteen_a_root, scene, image, 2)
        t = torch.load(fpath, map_location="cpu")
        fmap = t[0] if t.dim() == 4 else t
        image_size = (int(fmap.shape[1]), int(fmap.shape[0]))
        del t, fmap
        meta = {
            "2d": meta_2d,
            "3d": meta_3d,
            "fusion": f"m2tokcat/m3tokcat/{mm}",
            "resolved_fields": fields,
            "budgets": {"total": total, "2d": total_2d, "3d": total_3d, "per_scale_2d": budgets_2d, "per_scale_3d": budgets_3d},
        }
        return fused, image_size, meta
    raise ValueError(f"runner supports 2donly/3donly/tokpipe; got {pipe}")


def run_experiment(exp: dict[str, Any], model, tokenizer, args: argparse.Namespace) -> None:
    exp_id = exp["id"]
    out_root = args.v18_root / "results" / exp_id
    outputs_root = out_root / "outputs"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "status.json").write_text(json.dumps({"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, ensure_ascii=False, indent=2), encoding="utf-8")
    scenes = args.scenes or discover_scenes(args.sixteen_a_root)
    for scene in scenes:
        qas = load_qas(args.sixteen_a_root, scene)
        logging.info("%s scene=%s images=%d questions=%d", exp_id, scene, len(qas), sum(len(v) for v in qas.values()))
        fields = resolve_fields(exp, args)
        ckpts = {}
        if fields.get("pipe") in {"3donly", "tokpipe"}:
            ckpts = {level: load_level_ckpt(ckpt_path(args.ckpt_root, scene, level)) for level in [0, 1, 2]}
        for idx, (image, questions) in enumerate(sorted(qas.items()), 1):
            out_file = outputs_root / scene / f"{image}_qa_results.json"
            if out_file.exists() and not args.overwrite:
                logging.info("skip %s %s [%d/%d]", scene, image, idx, len(qas))
                continue
            out_file.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            fused, image_size, token_meta = build_tokens_for_image(exp, scene, image, args, ckpts)
            answers = []
            for q in questions:
                ans = answer_question(model, tokenizer, fused, q.get("question", ""), image_size=image_size, max_new_tokens=args.max_new_tokens)
                answers.append({
                    "question_index": q.get("question_index"),
                    "question": q.get("question", ""),
                    "expected": q.get("expected", ""),
                    "final_answer": ans,
                    "rgb_answer": q.get("rgb_answer", ""),
                })
            payload = {
                "experiment_id": exp_id,
                "scene": scene,
                "image_name": image,
                "fields": fields,
                "token_meta": token_meta,
                "question_count": len(answers),
                "questions": answers,
                "timing_s": round(time.time() - t0, 4),
            }
            out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logging.info("done %s %s [%d/%d] %dQ", scene, image, idx, len(qas), len(answers))
            del fused
            gc.collect()
            torch.cuda.empty_cache()
        del ckpts
        gc.collect()
    (out_root / "status.json").write_text(json.dumps({"status": "outputs_complete", "completed_outputs_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v18-root", type=Path, default=DEFAULT_V18)
    ap.add_argument("--sixteen-a-root", type=Path, default=DEFAULT_16A)
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--exp-id", action="append", default=[], help="V18 experiment id to run. Can repeat.")
    ap.add_argument("--batch", choices=["phase-a", "batch1"], default=None)
    ap.add_argument("--scenes", nargs="+", default=None, help="Default: discover all benchmark scenes.")
    ap.add_argument("--model-path", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--pool-factor", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    registry = load_registry(args.v18_root)
    runnable = [
        e for e in registry.get("experiments", [])
        if (e.get("phase") == "A" and e.get("priority") == "batch1")
        or ((e.get("fields") or {}).get("pipe") == "tokpipe")
    ]
    phase_a = [e for e in runnable if e.get("phase") == "A" and e.get("priority") == "batch1"]
    if args.list:
        for exp in runnable:
            print(exp["id"])
        return 0
    selected_ids = set(args.exp_id)
    if args.batch in {"phase-a", "batch1"}:
        selected_ids.update(e["id"] for e in phase_a)
    if not selected_ids:
        raise SystemExit("select --exp-id ... or --batch phase-a; use --list to inspect supported experiments")
    selected = [e for e in runnable if e["id"] in selected_ids]
    missing = selected_ids - {e["id"] for e in selected}
    if missing:
        raise SystemExit(f"unsupported experiments for this runner: {sorted(missing)}")
    logging.info("Loading VLM: %s", args.model_path)
    name = get_model_name_from_path(args.model_path)
    tokenizer, model, _image_processor, _max_length = load_pretrained_model(args.model_path, None, name, device_map="cuda:0")
    model.eval()
    for exp in selected:
        run_experiment(exp, model, tokenizer, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

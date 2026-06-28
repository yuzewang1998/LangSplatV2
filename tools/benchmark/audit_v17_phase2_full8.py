#!/usr/bin/env python3
"""Audit v17 phase2 fusion coverage for the full 8-scene benchmark."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SCENES = [
    "brandenburg_gate",
    "buckingham_palace",
    "notre_dame_front_facade",
    "pantheon_exterior",
    "sacre_coeur",
    "taj_mahal",
    "temple_nara_japan",
    "trevi_fountain",
]
METHODS = ["block", "3d_first", "2d_first", "level_interleave", "token_interleave"]
DEFAULT_PHASE2 = Path("/mnt/data/wangyz/exp_results/historicalAgent/eval_results/v17_phase2_fusion")
DEFAULT_16A = Path("/mnt/data/wangyz/exp_results/historicalAgent/eval_results/9999/full_benchmark_16a")
DEFAULT_OUT = Path("experiment_results/benchmark/v17_2d3d_comparison")


def expected_images(root: Path, scene: str) -> int:
    d = root / scene
    return len(list(d.rglob("*_qa_results.json")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2)
    ap.add_argument("--sixteen-a-root", type=Path, default=DEFAULT_16A)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out_dir = args.out if args.out.is_absolute() else args.root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in SCENES:
        expected = expected_images(args.sixteen_a_root, scene)
        for method in METHODS:
            d = args.phase2_root / scene / method
            count = len(list(d.glob("*_v17_p2_qa.json"))) if d.exists() else 0
            rows.append({
                "scene": scene,
                "method": method,
                "expected_images": expected,
                "actual_images": count,
                "complete": count == expected and expected > 0,
                "path": str(d),
            })
    missing = [r for r in rows if not r["complete"]]
    scripts = [
        args.root / "tools/benchmark/run_v17_phase2_fusion.py",
        args.root / "tools/benchmark/run_v17_hybrid_scan.py",
        args.root / "tools/benchmark/run_v17_3d_benchmark.py",
        args.root / "tools/benchmark/sample_3d_gaussians.py",
    ]
    payload = {
        "phase2_root": str(args.phase2_root),
        "expected_scenes": SCENES,
        "methods": METHODS,
        "complete_scene_method_pairs": sum(1 for r in rows if r["complete"]),
        "total_scene_method_pairs": len(rows),
        "missing_or_incomplete": missing,
        "missing_scenes": sorted({r["scene"] for r in missing}),
        "runner_script_status": {str(p.relative_to(args.root) if p.is_relative_to(args.root) else p): p.exists() for p in scripts},
        "status": "phase2 full8 complete" if not missing else "phase2 incomplete; missing scenes/methods require rebuild",
    }
    (out_dir / "phase2_full8_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "phase2_full8_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "method", "expected_images", "actual_images", "complete", "path"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: payload[k] for k in ("complete_scene_method_pairs", "total_scene_method_pairs", "missing_scenes", "runner_script_status")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

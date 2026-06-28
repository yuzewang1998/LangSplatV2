#!/usr/bin/env python3
"""Simple tmux-based V18 queue scheduler.

It keeps GPU0-3 busy with full V18 experiments by launching
run_v18_worker.sh <gpu> <exp_id> when the corresponding v18 GPU tmux lane is idle.

State lives in experiment_results/benchmark/v18_pipeline_ablation/v18_queue.json.
Completed experiments (summary.json status=completed) are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
V18_ROOT = ROOT / "experiment_results" / "benchmark" / "v18_pipeline_ablation"
QUEUE_PATH = V18_ROOT / "v18_queue.json"
LOG_DIR = V18_ROOT / "logs"


def now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def tmux_sessions() -> list[str]:
    proc = subprocess.run(["tmux", "ls"], text=True, capture_output=True)
    if proc.returncode != 0:
        return []
    return [line.split(":", 1)[0] for line in proc.stdout.splitlines() if line.strip()]


def gpu_busy_by_v18_session(gpu: int, sessions: list[str]) -> bool:
    prefixes = [f"v18_gpu{gpu}", f"v18q_gpu{gpu}"]
    return any(any(s.startswith(p) for p in prefixes) for s in sessions)


def safe_session_name(gpu: int, exp_id: str) -> str:
    short = re.sub(r"[^A-Za-z0-9_.-]+", "_", exp_id)
    # tmux has practical limits; include enough of the experiment id for debugging.
    return f"v18q_gpu{gpu}_{short[:120]}"


def exp_completed(exp_id: str) -> bool:
    summary = V18_ROOT / "results" / exp_id / "summary.json"
    if not summary.exists():
        return False
    try:
        data = load_json(summary)
    except Exception:
        return False
    return data.get("status") == "completed"


def exp_running_in_tmux(exp_id: str, sessions: list[str]) -> bool:
    return any(exp_id[:80] in s for s in sessions)


def launch(gpu: int, exp_id: str) -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    session = safe_session_name(gpu, exp_id)
    cmd = f"cd {ROOT} && bash tools/benchmark/run_v18_worker.sh {gpu} {exp_id}"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, cmd], check=True)
    return session


def reconcile(queue: dict[str, Any], sessions: list[str]) -> bool:
    changed = False
    for item in queue.get("experiments", []):
        exp_id = item.get("id")
        if not exp_id:
            continue
        if exp_completed(exp_id):
            if item.get("status") != "completed":
                item["status"] = "completed"
                item["completed_at"] = now()
                changed = True
        elif exp_running_in_tmux(exp_id, sessions):
            if item.get("status") != "running":
                item["status"] = "running"
                changed = True
        elif item.get("status") == "running":
            # Session disappeared without completed summary; make it retryable.
            item["status"] = "pending"
            item["retry_after_session_end_at"] = now()
            changed = True
    return changed


def schedule_once(args: argparse.Namespace) -> dict[str, Any]:
    queue = load_json(args.queue)
    sessions = tmux_sessions()
    changed = reconcile(queue, sessions)
    launched: list[dict[str, Any]] = []
    for gpu in queue.get("gpus", [0, 1, 2, 3]):
        gpu = int(gpu)
        if gpu_busy_by_v18_session(gpu, sessions):
            continue
        next_item = None
        for item in queue.get("experiments", []):
            exp_id = item.get("id")
            if not exp_id or exp_completed(exp_id) or exp_running_in_tmux(exp_id, sessions):
                continue
            if item.get("status", "pending") in {"pending", "queued"}:
                next_item = item
                break
        if next_item is None:
            continue
        exp_id = next_item["id"]
        if args.dry_run:
            session = f"DRY_RUN_GPU{gpu}"
        else:
            session = launch(gpu, exp_id)
        next_item.update({"status": "running", "gpu": gpu, "session": session, "started_at": now()})
        launched.append({"gpu": gpu, "exp_id": exp_id, "session": session})
        sessions.append(session)
        changed = True
    if changed:
        queue["updated_at"] = now()
        write_json(args.queue, queue)
    return {"time": now(), "launched": launched, "active_sessions": sessions}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, default=QUEUE_PATH)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    while True:
        try:
            result = schedule_once(args)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"time": now(), "error": repr(exc)}, ensure_ascii=False), flush=True)
        if args.once:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

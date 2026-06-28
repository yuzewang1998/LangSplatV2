#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 <gpu_id> <exp_id> [exp_id...]" >&2
  exit 2
fi
GPU="$1"; shift
cd "$(dirname "$0")/../.."
LOG_DIR="experiment_results/benchmark/v18_pipeline_ablation/logs"
mkdir -p "$LOG_DIR"
echo "[$(date -Is)] worker gpu=$GPU experiments=$*"
for exp in "$@"; do
  echo "[$(date -Is)] GPU$GPU START outputs $exp"
  CUDA_VISIBLE_DEVICES="$GPU" python -u tools/benchmark/run_v18_phase_a.py --exp-id "$exp" 2>&1 | tee -a "$LOG_DIR/gpu${GPU}.${exp}.run.log"
  echo "[$(date -Is)] GPU$GPU START scoring+JudgeLM $exp"
  CUDA_VISIBLE_DEVICES="$GPU" python -u tools/benchmark/score_v18_results.py --exp-id "$exp" --with-judgelm 2>&1 | tee -a "$LOG_DIR/gpu${GPU}.${exp}.score.log"
  echo "[$(date -Is)] GPU$GPU DONE $exp"
done
echo "[$(date -Is)] worker gpu=$GPU complete"

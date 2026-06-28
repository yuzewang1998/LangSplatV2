#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
LOG_DIR="experiment_results/benchmark/v18_pipeline_ablation/logs"
mkdir -p "$LOG_DIR"
EXPS=(
  v18_2donly_u2d_none3d_m2tokcat_m3none_mmnone_odirect_T512_R0_C1
  v18_2donly_rep2d_none3d_m2tokcat_m3none_mmnone_odirect_T512_R0_C1
  v18_2donly_norm2d_none3d_m2tokcat_m3none_mmnone_odirect_T512_R0_C1
  v18_3donly_none2d_geo3d_m2none_m3tokcat_mmnone_odirect_T512_R100_C1
  v18_3donly_none2d_feat3d_m2none_m3tokcat_mmnone_odirect_T512_R100_C1
  v18_3donly_none2d_opa3d_m2none_m3tokcat_mmnone_odirect_T512_R100_C1
  v18_3donly_none2d_mix3d_m2none_m3tokcat_mmnone_odirect_T512_R100_C1
)
for exp in "${EXPS[@]}"; do
  echo "[$(date -Is)] START outputs $exp"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u tools/benchmark/run_v18_phase_a.py --exp-id "$exp" 2>&1 | tee "$LOG_DIR/${exp}.run.log"
  echo "[$(date -Is)] START scoring $exp"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u tools/benchmark/score_v18_results.py --exp-id "$exp" --with-judgelm 2>&1 | tee "$LOG_DIR/${exp}.score.log"
  echo "[$(date -Is)] DONE $exp"
done
echo "[$(date -Is)] V18 Phase A batch complete"

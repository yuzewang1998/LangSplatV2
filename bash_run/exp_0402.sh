#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ROOT_PATH="${DATASET_ROOT_PATH:-/mnt/data/wangyz/PT}"
DATASET_NAME="${DATASET_NAME:-brandenburg_gate}"
EXP_ID="${EXP_ID:-0402}"
GPU="${GPU:-0}"
ITERATIONS="${ITERATIONS:-10000}"
TOPK="${TOPK:-4}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-128}"
OUTPUT_NAMESPACE="${OUTPUT_NAMESPACE:-exp_0402}"
JUDGE_METHOD="${JUDGE_METHOD:-judgelm}"
JUDGELM_ROOT="${JUDGELM_ROOT:-/home/wangyz/project/2past_project/JudgeLM-main}"
JUDGELM_MODEL_PATH="${JUDGELM_MODEL_PATH:-/home/wangyz/.cache/huggingface/hub/models--BAAI--JudgeLM-7B-v1.0/snapshots/dfbebe054b24c946d76bfc85c977b0d68a8be913}"
JUDGELM_MODEL_ID="${JUDGELM_MODEL_ID:-JudgeLM-7B-v1.0}"
JUDGELM_MAX_NEW_TOKENS="${JUDGELM_MAX_NEW_TOKENS:-256}"
JUDGELM_NUM_GPUS_PER_MODEL="${JUDGELM_NUM_GPUS_PER_MODEL:-1}"
JUDGELM_FAST_EVAL="${JUDGELM_FAST_EVAL:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/output/${OUTPUT_NAMESPACE}}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_DIR}/eval_result/${OUTPUT_NAMESPACE}}"

cd "${REPO_DIR}"

bash "${REPO_DIR}/run_all.sh" \
    --dataset_root_path "${DATASET_ROOT_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --exp_id "${EXP_ID}" \
    --gpu "${GPU}" \
    --iterations "${ITERATIONS}" \
    --topk "${TOPK}" \
    --codebook_size "${CODEBOOK_SIZE}" \
    --output_root "${OUTPUT_ROOT}" \
    --eval_root "${EVAL_ROOT}" \
    --judge_method "${JUDGE_METHOD}" \
    --judgelm_root "${JUDGELM_ROOT}" \
    --judgelm_model_path "${JUDGELM_MODEL_PATH}" \
    --judgelm_model_id "${JUDGELM_MODEL_ID}" \
    --judgelm_max_new_tokens "${JUDGELM_MAX_NEW_TOKENS}" \
    --judgelm_num_gpus_per_model "${JUDGELM_NUM_GPUS_PER_MODEL}" \
    --judgelm_fast_eval "${JUDGELM_FAST_EVAL}" \
    "$@"

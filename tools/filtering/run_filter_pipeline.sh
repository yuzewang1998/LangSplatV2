#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_DIR=""
INPUT_TSV=""
LANDMARK_NAME=""
OLLAMA_HOST="192.168.192.124"
OLLAMA_PORT="11434"
MODEL_NAME="llava:34b"
REQUEST_TIMEOUT="180"
FILTER_MODE="strict"
RESUME_FROM="0"
NUM_SAMPLES="20"
FULL_GRID_COLS="15"
PRIORITY_FILENAMES=""
SKIP_EXISTING_DECISIONS="0"

usage() {
    cat <<EOF
用法:
  bash tools/filtering/run_filter_pipeline.sh --dataset_dir /abs/path/to/dataset --input_tsv xxx.tsv [options]

必填参数:
  --dataset_dir PATH       数据集目录，要求包含 dense/images
  --input_tsv PATH|NAME    输入 TSV，可传绝对路径，或相对 dataset_dir 的文件名

可选参数:
  --landmark_name NAME     地标名称；默认使用 dataset_dir 名称（下划线替换为空格）
  --ollama_host HOST       Ollama 地址，默认 192.168.192.124
  --ollama_port PORT       Ollama 端口，默认 11434
  --model_name NAME        VLM 模型，默认 llava:34b
  --request_timeout SEC    单次请求超时，默认 180
  --filter_mode MODE       strict 或 benchmark_permissive，默认 strict
  --resume_from N          从第 N 行继续，默认 0
  --num_samples N          可视化随机样本数，默认 20
  --full_grid_cols N       全量总览图列数，默认 15
  --priority_filenames P   可选：优先处理的文件名列表，每行一个文件名
  --skip_existing_decisions 根据判定日志跳过已处理文件，便于断点续跑
  -h, --help               显示本帮助

输出:
  <dataset_dir>/<stem>_filtered.tsv
  <dataset_dir>/<stem>_filtered_decisions.jsonl
  <repo>/eval_result/filtering/<dataset_name>/<stem>_filtered/
EOF
}

resolve_path() {
    local value="$1"
    if [[ "${value}" = /* ]]; then
        printf '%s\n' "${value}"
    else
        printf '%s\n' "${DATASET_DIR}/${value}"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_dir) DATASET_DIR="$2"; shift 2 ;;
        --input_tsv) INPUT_TSV="$2"; shift 2 ;;
        --landmark_name) LANDMARK_NAME="$2"; shift 2 ;;
        --ollama_host) OLLAMA_HOST="$2"; shift 2 ;;
        --ollama_port) OLLAMA_PORT="$2"; shift 2 ;;
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --request_timeout) REQUEST_TIMEOUT="$2"; shift 2 ;;
        --filter_mode) FILTER_MODE="$2"; shift 2 ;;
        --resume_from) RESUME_FROM="$2"; shift 2 ;;
        --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
        --full_grid_cols) FULL_GRID_COLS="$2"; shift 2 ;;
        --priority_filenames) PRIORITY_FILENAMES="$2"; shift 2 ;;
        --skip_existing_decisions) SKIP_EXISTING_DECISIONS="1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "${DATASET_DIR}" || -z "${INPUT_TSV}" ]]; then
    usage
    exit 1
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "错误：dataset_dir 不存在: ${DATASET_DIR}"
    exit 1
fi

INPUT_TSV="$(resolve_path "${INPUT_TSV}")"

if [[ ! -f "${INPUT_TSV}" ]]; then
    echo "错误：输入 TSV 不存在: ${INPUT_TSV}"
    exit 1
fi

IMAGE_DIR="${DATASET_DIR}/dense/images"
if [[ ! -d "${IMAGE_DIR}" ]]; then
    echo "错误：图像目录不存在: ${IMAGE_DIR}"
    exit 1
fi

DATASET_NAME="$(basename "${DATASET_DIR}")"
INPUT_BASENAME="$(basename "${INPUT_TSV}")"
INPUT_STEM="${INPUT_BASENAME%.tsv}"
OUTPUT_TSV="${DATASET_DIR}/${INPUT_STEM}_filtered.tsv"
DECISION_LOG="${DATASET_DIR}/${INPUT_STEM}_filtered_decisions.jsonl"
VIS_OUTPUT_DIR="${REPO_ROOT}/eval_result/filtering/${DATASET_NAME}/${INPUT_STEM}_filtered"

if [[ -z "${LANDMARK_NAME}" ]]; then
    LANDMARK_NAME="$(printf '%s' "${DATASET_NAME}" | tr '_' ' ')"
fi

echo "========================================"
echo "通用图像过滤流程"
echo "========================================"
echo "数据集目录: ${DATASET_DIR}"
echo "输入 TSV: ${INPUT_TSV}"
echo "输出 TSV: ${OUTPUT_TSV}"
echo "判定日志: ${DECISION_LOG}"
echo "图像目录: ${IMAGE_DIR}"
echo "可视化目录: ${VIS_OUTPUT_DIR}"
echo "地标名称: ${LANDMARK_NAME}"
echo "Ollama 服务: ${OLLAMA_HOST}:${OLLAMA_PORT}"
echo "模型: ${MODEL_NAME}"
echo "过滤模式: ${FILTER_MODE}"
echo "超时: ${REQUEST_TIMEOUT}s"
if [[ -n "${PRIORITY_FILENAMES}" ]]; then
    echo "优先文件列表: ${PRIORITY_FILENAMES}"
fi
if [[ "${SKIP_EXISTING_DECISIONS}" = "1" ]]; then
    echo "断点模式: skip_existing_decisions=True"
fi
echo "========================================"
echo ""

echo "测试 Ollama 服务连接..."
if curl -s "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
    echo "✓ Ollama 服务连接成功"
else
    echo "✗ 无法连接到 Ollama 服务: http://${OLLAMA_HOST}:${OLLAMA_PORT}"
    exit 1
fi

echo "检查模型 ${MODEL_NAME} 是否可用..."
if curl -s "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags" | grep -q "${MODEL_NAME}"; then
    echo "✓ 模型 ${MODEL_NAME} 可用"
else
    echo "警告：模型 ${MODEL_NAME} 可能未安装"
    echo "继续执行，如果模型不存在将会报错"
fi

echo ""
echo "开始过滤..."
echo "========================================"

python "${REPO_ROOT}/filter_images_with_vlm.py" \
    --tsv_path "${INPUT_TSV}" \
    --output_path "${OUTPUT_TSV}" \
    --decision_log_path "${DECISION_LOG}" \
    --image_dir "${IMAGE_DIR}" \
    --ollama_host "${OLLAMA_HOST}" \
    --ollama_port "${OLLAMA_PORT}" \
    --model_name "${MODEL_NAME}" \
    --landmark_name "${LANDMARK_NAME}" \
    --filter_mode "${FILTER_MODE}" \
    --request_timeout "${REQUEST_TIMEOUT}" \
    --resume_from "${RESUME_FROM}" \
    ${PRIORITY_FILENAMES:+--priority_filenames "${PRIORITY_FILENAMES}"} \
    $(if [[ "${SKIP_EXISTING_DECISIONS}" = "1" ]]; then printf '%s' "--skip_existing_decisions"; fi)

echo ""
echo "生成可视化..."
echo "========================================"

VIS_ARGS=(
    --filtered_tsv "${OUTPUT_TSV}"
    --image_dir "${IMAGE_DIR}"
    --output_dir "${VIS_OUTPUT_DIR}"
    --num_samples "${NUM_SAMPLES}"
    --save_full_grid
    --full_grid_cols "${FULL_GRID_COLS}"
)

python "${REPO_ROOT}/tools/filtering/visualize_filtered_images.py" "${VIS_ARGS[@]}"

echo ""
echo "========================================"
echo "全部完成"
echo "========================================"
echo "过滤结果: ${OUTPUT_TSV}"
echo "判定日志: ${DECISION_LOG}"
echo "可视化目录: ${VIS_OUTPUT_DIR}"
echo "========================================"

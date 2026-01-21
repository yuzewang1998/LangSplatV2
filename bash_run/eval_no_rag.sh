#!/bin/bash
# ============================================================================
# 评估脚本 - 不使用 RAG
# ============================================================================
# 用法: bash eval_no_rag.sh [--exp_id ID] [--gpu ID]
#
# 此脚本跳过训练和渲染，仅执行评估（不使用RAG增强）
# 评估结果保存到: eval_result/{INDEX}/analysis_no_rag/
# ============================================================================

set -e

# 默认参数
DATASET_ROOT_PATH=/mnt/data/wangyz/PT
DATASET_NAME=brandenburg_gate
EXP_ID="rag_compare"
GPU=3
ITERATIONS=10000
TOPK=4
CODEBOOK_SIZE=64
SAMPLING_STRATEGY=grid
GRID_SIZE=5

# 路径
LANGSPLAT_DIR=/home/wangyz/project/0working/LangSplatV2
LLAVA_DIR=/home/wangyz/project/0working/LangSplatV2/LLaVA-NeXT

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_id) EXP_ID="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --dataset_name) DATASET_NAME="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 生成索引名
INDEX="${EXP_ID}_iter${ITERATIONS}_topk${TOPK}_cb${CODEBOOK_SIZE}"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              评估测试 - 不使用 RAG                                          ║"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║ 实验索引: ${INDEX}"
echo "║ 数据集: ${DATASET_NAME}"
echo "║ GPU: ${GPU}"
echo "║ RAG增强: ❌ 禁用"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 增加文件句柄限制
ulimit -n 65535

# ============================================================================
# 评估 (llava 环境) - 不使用 RAG
# ============================================================================
echo ""
echo "┌────────────────────────────────────────────────────────────────────────────┐"
echo "│ 评估 (conda: llava) - 无RAG增强                                            │"
echo "└────────────────────────────────────────────────────────────────────────────┘"
echo ""

cd ${LLAVA_DIR}
export CUDA_VISIBLE_DEVICES=${GPU}

# 输出目录 - 无 RAG 标记
EVAL_OUTPUT_DIR="${LANGSPLAT_DIR}/eval_result/${INDEX}/analysis_no_rag"
mkdir -p ${EVAL_OUTPUT_DIR}

RENDERED_BASE_DIR="${LANGSPLAT_DIR}/eval_result/${INDEX}"

echo ">>> 运行问答评估 (无RAG) <<<"
echo "    输出目录: ${EVAL_OUTPUT_DIR}"
echo ""

CUDA_VISIBLE_DEVICES=${GPU} conda run -n llava python verify_reconstruction_quality.py \
    --index ${INDEX} \
    --rendered_base_dir ${RENDERED_BASE_DIR} \
    --eval_all \
    --use_semantic_pooling \
    --use_qa_json \
    --sampling_strategy ${SAMPLING_STRATEGY} \
    --grid_size ${GRID_SIZE} \
    --qa_output_dir ${EVAL_OUTPUT_DIR} \
    --dataset_name ${DATASET_NAME}

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ 无RAG评估完成!                                        ║"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║ 结果目录: ${EVAL_OUTPUT_DIR}"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

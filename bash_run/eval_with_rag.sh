#!/bin/bash
# ============================================================================
# 评估脚本 - 使用 RAG 增强
# ============================================================================
# 用法: bash eval_with_rag.sh [--exp_id ID] [--gpu ID]
#
# 此脚本跳过训练和渲染，仅执行 RAG 构建/预查询 + 评估
# 评估结果保存到: eval_result/{INDEX}/analysis_with_rag/
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
RAG_CACHE_DIR=${LANGSPLAT_DIR}/rag_data

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
echo "║              评估测试 - 使用 RAG 增强                                       ║"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║ 实验索引: ${INDEX}"
echo "║ 数据集: ${DATASET_NAME}"
echo "║ GPU: ${GPU}"
echo "║ RAG增强: ✅ 启用"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 增加文件句柄限制
ulimit -n 65535

# ============================================================================
# Step 1: RAG 构建和预查询 (rag_anything 环境)
# ============================================================================
echo ""
echo "┌────────────────────────────────────────────────────────────────────────────┐"
echo "│ Step 1/2: RAG构建和预查询 (conda: rag_anything)                            │"
echo "└────────────────────────────────────────────────────────────────────────────┘"
echo ""

cd ${LLAVA_DIR}
export CUDA_VISIBLE_DEVICES=${GPU}

QA_JSON_DIR=${DATASET_ROOT_PATH}/label_llm/${DATASET_NAME}
RAG_DATASET_DIR=${RAG_CACHE_DIR}/${DATASET_NAME}
RAG_STORAGE_DIR=${RAG_DATASET_DIR}/rag_storage
RAG_CACHE_FILE=${RAG_DATASET_DIR}/query_cache/rag_contexts.json

RAG_BUILD_NEEDED=false
RAG_QUERY_NEEDED=false

if [ ! -d "$RAG_STORAGE_DIR" ]; then
    RAG_BUILD_NEEDED=true
    RAG_QUERY_NEEDED=true
    echo "📦 RAG需要构建"
else
    echo "✅ RAG存储已存在: ${RAG_STORAGE_DIR}"
fi

if [ ! -f "$RAG_CACHE_FILE" ]; then
    RAG_QUERY_NEEDED=true
    echo "📋 RAG预查询需要执行"
else
    echo "✅ RAG缓存已存在: ${RAG_CACHE_FILE}"
fi

if [ "$RAG_BUILD_NEEDED" = true ] || [ "$RAG_QUERY_NEEDED" = true ]; then
    echo ""
    echo ">>> 运行RAG管理器 <<<"
    echo ""

    RAG_ACTION="all"
    if [ "$RAG_BUILD_NEEDED" = false ]; then
        RAG_ACTION="query"
    fi

    conda run -n rag_anything python rag_manager.py ${RAG_ACTION} \
        --dataset ${DATASET_NAME} \
        --qa_json_dir ${QA_JSON_DIR} \
        --mode hybrid

    echo ""
    echo "✅ RAG准备完成"
else
    echo ""
    echo "✅ RAG已就绪"
fi

# ============================================================================
# Step 2: 评估 (llava 环境) - 使用 RAG
# ============================================================================
echo ""
echo "┌────────────────────────────────────────────────────────────────────────────┐"
echo "│ Step 2/2: 评估 (conda: llava) - 🧠 RAG增强                                 │"
echo "└────────────────────────────────────────────────────────────────────────────┘"
echo ""

cd ${LLAVA_DIR}

# 输出目录 - 带 RAG 标记
EVAL_OUTPUT_DIR="${LANGSPLAT_DIR}/eval_result/${INDEX}/analysis_with_rag"
mkdir -p ${EVAL_OUTPUT_DIR}

RENDERED_BASE_DIR="${LANGSPLAT_DIR}/eval_result/${INDEX}"

echo ">>> 运行问答评估 (RAG增强) <<<"
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
    --dataset_name ${DATASET_NAME} \
    --use_rag \
    --rag_cache_dir ${RAG_CACHE_DIR}

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ RAG增强评估完成!                                      ║"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║ 结果目录: ${EVAL_OUTPUT_DIR}"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

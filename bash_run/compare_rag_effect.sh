#!/bin/bash
# ============================================================================
# RAG 对比评估脚本
# ============================================================================
# 用法: bash compare_rag_effect.sh [--index INDEX] [--gpu ID] [--skip_eval]
#
# 此脚本依次运行有/无RAG的评估，然后生成对比报告
# ============================================================================

set -e

# 默认参数
INDEX="iter10000_topk4_cb64"  # 使用已有的渲染结果
GPU=3
DATASET_NAME=brandenburg_gate
SKIP_EVAL=false

# 路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANGSPLAT_DIR=/home/wangyz/project/0working/LangSplatV2
LLAVA_DIR=/home/wangyz/project/0working/LangSplatV2/LLaVA-NeXT
RAG_CACHE_DIR=${LANGSPLAT_DIR}/rag_data

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --index) INDEX="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --dataset_name) DATASET_NAME="$2"; shift 2 ;;
        --skip_eval) SKIP_EVAL=true; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    RAG 效果对比评估                                         ║"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║ 实验索引: ${INDEX}"
echo "║ 数据集: ${DATASET_NAME}"
echo "║ GPU: ${GPU}"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 增加文件句柄限制
ulimit -n 65535

if [ "$SKIP_EVAL" = false ]; then
    # ============================================================================
    # Step 1: 运行无RAG评估
    # ============================================================================
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "  [1/3] 运行无RAG评估..."
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""

    cd ${LLAVA_DIR}
    export CUDA_VISIBLE_DEVICES=${GPU}

    NO_RAG_OUTPUT="${LANGSPLAT_DIR}/eval_result/${INDEX}/analysis_no_rag"
    mkdir -p ${NO_RAG_OUTPUT}

    echo ">>> 运行问答评估 (无RAG) <<<"
    echo "    输出目录: ${NO_RAG_OUTPUT}"
    echo ""

    conda run -n llava python verify_reconstruction_quality.py \
        --index ${INDEX} \
        --rendered_base_dir ${LANGSPLAT_DIR}/eval_result/${INDEX} \
        --eval_all \
        --use_semantic_pooling \
        --use_qa_json \
        --sampling_strategy grid \
        --grid_size 5 \
        --qa_output_dir ${NO_RAG_OUTPUT} \
        --dataset_name ${DATASET_NAME}

    echo ""
    echo "✅ 无RAG评估完成"

    # ============================================================================
    # Step 2: 运行有RAG评估
    # ============================================================================
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "  [2/3] 运行RAG增强评估..."
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""

    # 先确保 RAG 已构建和预查询
    QA_JSON_DIR=${DATASET_ROOT_PATH:-/mnt/data/wangyz/PT}/label_llm/${DATASET_NAME}
    RAG_CACHE_FILE=${RAG_CACHE_DIR}/${DATASET_NAME}/query_cache/rag_contexts.json

    if [ ! -f "$RAG_CACHE_FILE" ]; then
        echo "📦 RAG缓存不存在，需要构建..."
        conda run -n rag_anything python rag_manager.py all \
            --dataset ${DATASET_NAME} \
            --qa_json_dir ${QA_JSON_DIR} \
            --mode hybrid
    else
        echo "✅ RAG缓存已存在: ${RAG_CACHE_FILE}"
    fi

    WITH_RAG_OUTPUT="${LANGSPLAT_DIR}/eval_result/${INDEX}/analysis_with_rag"
    mkdir -p ${WITH_RAG_OUTPUT}

    echo ""
    echo ">>> 运行问答评估 (RAG增强) <<<"
    echo "    输出目录: ${WITH_RAG_OUTPUT}"
    echo ""

    conda run -n llava python verify_reconstruction_quality.py \
        --index ${INDEX} \
        --rendered_base_dir ${LANGSPLAT_DIR}/eval_result/${INDEX} \
        --eval_all \
        --use_semantic_pooling \
        --use_qa_json \
        --sampling_strategy grid \
        --grid_size 5 \
        --qa_output_dir ${WITH_RAG_OUTPUT} \
        --dataset_name ${DATASET_NAME} \
        --use_rag \
        --rag_cache_dir ${RAG_CACHE_DIR}

    echo ""
    echo "✅ RAG增强评估完成"
fi

# ============================================================================
# Step 3: 生成对比报告
# ============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "  [3/3] 生成对比报告..."
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# 结果目录
NO_RAG_DIR="${LANGSPLAT_DIR}/eval_result/${INDEX}/analysis_no_rag"
WITH_RAG_DIR="${LANGSPLAT_DIR}/eval_result/${INDEX}/analysis_with_rag"
COMPARE_DIR="${LANGSPLAT_DIR}/eval_result/${INDEX}/comparison"
mkdir -p ${COMPARE_DIR}

# 运行对比分析 Python 脚本
cd ${LLAVA_DIR}
python compare_rag_results.py \
    --no_rag_dir ${NO_RAG_DIR} \
    --with_rag_dir ${WITH_RAG_DIR} \
    --output_dir ${COMPARE_DIR} \
    --index ${INDEX}

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ RAG对比评估完成!                                      ║"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║ 无RAG结果: ${NO_RAG_DIR}"
echo "║ 有RAG结果: ${WITH_RAG_DIR}"
echo "║ 对比报告: ${COMPARE_DIR}"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

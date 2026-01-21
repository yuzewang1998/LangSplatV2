#!/bin/bash
# ============================================================================
# 一站式训练-渲染-评估-RAG对比脚本
# ============================================================================
# 用法: bash run_all.sh [OPTIONS]
#
# 可选参数:
#   --exp_id ID            实验编号，自定义标识 (例如: exp001, test1)
#   --dataset_name NAME    数据集名称 (默认: brandenburg_gate)
#   --index NAME           实验索引，留空则自动生成
#   --gpu ID               GPU编号 (默认: 3)
#   --iterations NUM       训练迭代数 (默认: 10000)
#   --topk NUM             top-k值 (默认: 4)
#   --codebook_size NUM    码本大小 (默认: 64)
#   --skip_train           跳过训练
#   --skip_render          跳过渲染
#   --skip_rag             跳过RAG构建和预查询（同时禁用RAG评估）
#   --skip_eval            跳过评估
#   --no_rag               禁用RAG增强评估（仅运行无RAG基线）
#   --force_rebuild_rag    强制重建RAG
#   --sampling STRATEGY    采样策略: grid/hierarchical (默认: grid)
#   --grid_size NUM        网格大小 (默认: 5)
#
# 索引命名规则:
#   有exp_id: {exp_id}_iter{N}_topk{K}_cb{C}
#   无exp_id: iter{N}_topk{K}_cb{C}
#
# 流程说明:
#   Step 1: 训练 (3个level: Small/Medium/Large)
#   Step 2: 渲染 (3个level)
#   Step 3: RAG 构建 + 预查询 (如果启用)
#   Step 4: 评估
#     4.1 无RAG评估（基线，总是运行）
#     4.2 有RAG评估（如果启用）
#   Step 5: 生成RAG对比报告 (如果启用RAG)
#
# 示例:
#   bash run_all.sh --exp_id exp001                    # 完整流程，索引=exp001_iter10000_topk4_cb64
#   bash run_all.sh --exp_id test --codebook_size 128  # 索引=test_iter10000_topk4_cb128
#   bash run_all.sh --skip_train --skip_render         # 只做RAG+评估+对比
#   bash run_all.sh --no_rag                           # 不使用RAG，仅运行基线评估
#   bash run_all.sh --force_rebuild_rag                # 强制重建RAG知识库
# ============================================================================

set -e  # 遇到错误立即退出

# ============================================================================
# 默认参数
# ============================================================================
DATASET_ROOT_PATH=/mnt/data/wangyz/PT
DATASET_NAME=brandenburg_gate
EXP_ID=""  # 实验编号，自定义标识
INDEX=""   # 留空则自动生成
GPU=3
ITERATIONS=10000
TOPK=4
CHECKPOINT=${ITERATIONS}
CODEBOOK_SIZE=64

# 评估参数
SAMPLING_STRATEGY=grid
GRID_SIZE=5

# 跳过标志
SKIP_TRAIN=false
SKIP_RENDER=false
SKIP_RAG=false
SKIP_EVAL=false

# RAG 参数
USE_RAG=true
FORCE_REBUILD_RAG=false

# 是否自动生成索引名
AUTO_INDEX=true

# 路径
LANGSPLAT_DIR=/home/wangyz/project/0working/LangSplatV2
LLAVA_DIR=/home/wangyz/project/0working/LangSplatV2/LLaVA-NeXT
RAG_CACHE_DIR=${LANGSPLAT_DIR}/rag_data

# ============================================================================
# 解析命令行参数
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_id) EXP_ID="$2"; shift 2 ;;
        --dataset_name) DATASET_NAME="$2"; shift 2 ;;
        --index) INDEX="$2"; AUTO_INDEX=false; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; CHECKPOINT="$2"; shift 2 ;;
        --topk) TOPK="$2"; shift 2 ;;
        --codebook_size) CODEBOOK_SIZE="$2"; shift 2 ;;
        --sampling) SAMPLING_STRATEGY="$2"; shift 2 ;;
        --grid_size) GRID_SIZE="$2"; shift 2 ;;
        --skip_train) SKIP_TRAIN=true; shift ;;
        --skip_render) SKIP_RENDER=true; shift ;;
        --skip_rag) SKIP_RAG=true; USE_RAG=false; shift ;;
        --skip_eval) SKIP_EVAL=true; shift ;;
        --use_rag) USE_RAG=true; shift ;;
        --no_rag) USE_RAG=false; shift ;;
        --force_rebuild_rag) FORCE_REBUILD_RAG=true; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ============================================================================
# 自动生成实验索引名（包含关键参数）
# ============================================================================
if [ "$AUTO_INDEX" = true ] || [ -z "$INDEX" ]; then
    # 格式: [exp_id_]iter{迭代数}_topk{k}_cb{码本大小}
    if [ -n "$EXP_ID" ]; then
        INDEX="${EXP_ID}_iter${ITERATIONS}_topk${TOPK}_cb${CODEBOOK_SIZE}"
    else
        INDEX="iter${ITERATIONS}_topk${TOPK}_cb${CODEBOOK_SIZE}"
    fi
fi

# 更新checkpoint
CHECKPOINT=${ITERATIONS}

# ============================================================================
# 打印配置
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                     一站式训练-渲染-评估流程                                ║"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
if [ -n "$EXP_ID" ]; then
echo "║ 实验编号: ${EXP_ID}"
fi
echo "║ 实验索引: ${INDEX}"
echo "║ 数据集: ${DATASET_NAME}"
echo "║ GPU: ${GPU}"
echo "╠────────────────────────────────────────────────────────────────────────────╣"
echo "║ 训练参数: iterations=${ITERATIONS}, topk=${TOPK}, codebook=${CODEBOOK_SIZE}"
echo "║ 评估参数: sampling=${SAMPLING_STRATEGY}, grid=${GRID_SIZE}x${GRID_SIZE}, 评估所有图像"
echo "║ RAG增强: $([ "$USE_RAG" = true ] && echo "启用" || echo "禁用")"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║ 步骤: $([ "$SKIP_TRAIN" = true ] && echo "跳过训练" || echo "训练") → $([ "$SKIP_RENDER" = true ] && echo "跳过渲染" || echo "渲染") → $([ "$SKIP_RAG" = true ] && echo "跳过RAG" || echo "RAG") → $([ "$SKIP_EVAL" = true ] && echo "跳过评估" || echo "评估")"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 增加文件句柄限制
ulimit -n 65535

# ============================================================================
# Step 1: 训练 (langsplat_v2 环境)
# ============================================================================
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "┌────────────────────────────────────────────────────────────────────────────┐"
    echo "│ Step 1/3: 训练 (conda: langsplat_v2)                                       │"
    echo "└────────────────────────────────────────────────────────────────────────────┘"
    echo ""

    cd ${LANGSPLAT_DIR}
    export CUDA_VISIBLE_DEVICES=${GPU}

    TRAIN_NEEDED=false
    for level in 0 1 2
    do
        if [ $level -eq 0 ]; then
            SCALE_NAME="Small/64px"
        elif [ $level -eq 1 ]; then
            SCALE_NAME="Medium/192px"
        else
            SCALE_NAME="Large/448px"
        fi

        # 检查checkpoint是否已存在
        CKPT_PATH="output/${DATASET_NAME}_${INDEX}_${level}/chkpnt${ITERATIONS}.pth"

        if [ -f "$CKPT_PATH" ]; then
            echo "⏭️  Level ${level} (${SCALE_NAME}) 已训练完成，跳过"
            echo "   checkpoint: ${CKPT_PATH}"
        else
            TRAIN_NEEDED=true
            echo ""
            echo ">>> 训练 Level ${level} (${SCALE_NAME}) <<<"
            echo ""

            conda run -n langsplat_v2 python train.py \
                -s ${DATASET_ROOT_PATH}/${DATASET_NAME} \
                -m output/${DATASET_NAME}_${INDEX} \
                --start_checkpoint ${DATASET_ROOT_PATH}/${DATASET_NAME}/${DATASET_NAME}_vanilla3DGS/chkpnt30000.pth \
                --feature_level ${level} \
                --vq_layer_num 1 \
                --codebook_size ${CODEBOOK_SIZE} \
                --cos_loss \
                -r 2 \
                --topk ${TOPK} \
                --iterations ${ITERATIONS}

            echo ">>> Level ${level} 训练完成 <<<"
        fi
    done

    if [ "$TRAIN_NEEDED" = true ]; then
        echo ""
        echo "✅ 训练完成"
    else
        echo ""
        echo "✅ 所有Level已训练完成，无需重新训练"
    fi
    echo ""
else
    echo ""
    echo "⏭️  跳过训练 (--skip_train)"
    echo ""
fi

# ============================================================================
# Step 2: 渲染 (langsplat_v2 环境)
# ============================================================================
if [ "$SKIP_RENDER" = false ]; then
    echo ""
    echo "┌────────────────────────────────────────────────────────────────────────────┐"
    echo "│ Step 2/3: 渲染 (conda: langsplat_v2)                                       │"
    echo "└────────────────────────────────────────────────────────────────────────────┘"
    echo ""

    cd ${LANGSPLAT_DIR}
    export CUDA_VISIBLE_DEVICES=${GPU}

    GT_FEATURE_DIR=${DATASET_ROOT_PATH}/${DATASET_NAME}/llava_features_3584_multiscale
    GT_FOLDER=${DATASET_ROOT_PATH}/label_llm

    # 实验输出根目录
    EXP_OUTPUT_DIR="eval_result/${INDEX}"
    mkdir -p ${EXP_OUTPUT_DIR}

    RENDER_NEEDED=false
    for level in 0 1 2
    do
        if [ $level -eq 0 ]; then
            SCALE_NAME="Small"
        elif [ $level -eq 1 ]; then
            SCALE_NAME="Medium"
        else
            SCALE_NAME="Large"
        fi

        # 检查渲染结果是否已存在（新目录结构）
        RENDER_DIR="${EXP_OUTPUT_DIR}/level${level}/${DATASET_NAME}"

        if [ -d "$RENDER_DIR" ] && [ "$(ls -A $RENDER_DIR 2>/dev/null)" ]; then
            echo "⏭️  Level ${level} (${SCALE_NAME}) 已渲染完成，跳过"
            echo "   输出目录: ${RENDER_DIR}"
        else
            RENDER_NEEDED=true
            echo ""
            echo ">>> 渲染 Level ${level} (${SCALE_NAME}) <<<"
            echo ""

            conda run -n langsplat_v2 python render_lerf_llm.py \
                -s ${DATASET_ROOT_PATH}/${DATASET_NAME} \
                -m output/${DATASET_NAME}_${INDEX}_${level} \
                --dataset_name ${DATASET_NAME} \
                --index ${INDEX} \
                --ckpt_root_path output \
                --output_dir ${EXP_OUTPUT_DIR}/level${level} \
                --mask_thresh 0.4 \
                --json_folder ${GT_FOLDER} \
                --checkpoint ${CHECKPOINT} \
                --include_feature \
                --topk ${TOPK} \
                -r 2 \
                --visualize_comparison \
                --gt_feature_dir ${GT_FEATURE_DIR} \
                --comparison_scale ${SCALE_NAME}

            echo ">>> Level ${level} 渲染完成 <<<"
        fi
    done

    if [ "$RENDER_NEEDED" = true ]; then
        echo ""
        echo "✅ 渲染完成"
    else
        echo ""
        echo "✅ 所有Level已渲染完成，无需重新渲染"
    fi
    echo ""
else
    echo ""
    echo "⏭️  跳过渲染"
    echo ""
fi

# ============================================================================
# Step 3: RAG 构建和预查询 (rag_anything 环境)
# ============================================================================
if [ "$SKIP_RAG" = false ] && [ "$USE_RAG" = true ]; then
    echo ""
    echo "┌────────────────────────────────────────────────────────────────────────────┐"
    echo "│ Step 3/4: RAG构建和预查询 (conda: rag_anything)                            │"
    echo "└────────────────────────────────────────────────────────────────────────────┘"
    echo ""

    cd ${LLAVA_DIR}
    export CUDA_VISIBLE_DEVICES=${GPU}

    # 问答JSON目录
    QA_JSON_DIR=${DATASET_ROOT_PATH}/label_llm/${DATASET_NAME}

    # RAG缓存目录
    RAG_DATASET_DIR=${RAG_CACHE_DIR}/${DATASET_NAME}

    # 检查RAG是否已构建
    RAG_STORAGE_DIR=${RAG_DATASET_DIR}/rag_storage
    RAG_CACHE_FILE=${RAG_DATASET_DIR}/query_cache/rag_contexts.json

    RAG_BUILD_NEEDED=false
    RAG_QUERY_NEEDED=false

    if [ ! -d "$RAG_STORAGE_DIR" ] || [ "$FORCE_REBUILD_RAG" = true ]; then
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

        # 构建RAG参数
        RAG_ACTION="all"
        if [ "$RAG_BUILD_NEEDED" = false ]; then
            RAG_ACTION="query"
        fi

        FORCE_FLAG=""
        if [ "$FORCE_REBUILD_RAG" = true ]; then
            FORCE_FLAG="--force"
        fi

        conda run -n rag_anything python rag_manager.py ${RAG_ACTION} \
            --dataset ${DATASET_NAME} \
            --qa_json_dir ${QA_JSON_DIR} \
            --mode hybrid \
            ${FORCE_FLAG}

        echo ""
        echo "✅ RAG准备完成"
    else
        echo ""
        echo "✅ RAG已就绪，无需重新构建或查询"
    fi
    echo ""
else
    echo ""
    echo "⏭️  跳过RAG (--skip_rag 或 USE_RAG=false)"
    echo ""
fi

# ============================================================================
# Step 4: 评估 (llava 环境) - 同时运行有/无RAG评估并生成对比报告
# ============================================================================
if [ "$SKIP_EVAL" = false ]; then
    echo ""
    echo "┌────────────────────────────────────────────────────────────────────────────┐"
    echo "│ Step 4/5: 评估 (conda: llava)                                              │"
    echo "└────────────────────────────────────────────────────────────────────────────┘"
    echo ""

    cd ${LLAVA_DIR}
    export CUDA_VISIBLE_DEVICES=${GPU}

    # 渲染特征基础目录
    RENDERED_BASE_DIR="${LANGSPLAT_DIR}/eval_result/${INDEX}"

    # 评估结果目录
    NO_RAG_OUTPUT_DIR="${LANGSPLAT_DIR}/eval_result/${INDEX}/analysis_no_rag"
    WITH_RAG_OUTPUT_DIR="${LANGSPLAT_DIR}/eval_result/${INDEX}/analysis_with_rag"
    COMPARISON_DIR="${LANGSPLAT_DIR}/eval_result/${INDEX}/comparison"

    # ------------------------------------------------------------------------
    # Step 4.1: 无RAG评估（基线）
    # ------------------------------------------------------------------------
    echo ">>> [4.1] 运行无RAG评估（基线）<<<"
    echo ""

    mkdir -p ${NO_RAG_OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=${GPU} conda run -n llava python verify_reconstruction_quality.py \
        --index ${INDEX} \
        --rendered_base_dir ${RENDERED_BASE_DIR} \
        --eval_all \
        --use_semantic_pooling \
        --use_qa_json \
        --sampling_strategy ${SAMPLING_STRATEGY} \
        --grid_size ${GRID_SIZE} \
        --qa_output_dir ${NO_RAG_OUTPUT_DIR} \
        --dataset_name ${DATASET_NAME}

    echo ""
    echo "✅ 无RAG评估完成"
    echo ""

    # ------------------------------------------------------------------------
    # Step 4.2: 有RAG评估（如果启用）
    # ------------------------------------------------------------------------
    if [ "$USE_RAG" = true ]; then
        echo ">>> [4.2] 运行RAG增强评估 🧠 <<<"
        echo ""

        mkdir -p ${WITH_RAG_OUTPUT_DIR}

        CUDA_VISIBLE_DEVICES=${GPU} conda run -n llava python verify_reconstruction_quality.py \
            --index ${INDEX} \
            --rendered_base_dir ${RENDERED_BASE_DIR} \
            --eval_all \
            --use_semantic_pooling \
            --use_qa_json \
            --sampling_strategy ${SAMPLING_STRATEGY} \
            --grid_size ${GRID_SIZE} \
            --qa_output_dir ${WITH_RAG_OUTPUT_DIR} \
            --dataset_name ${DATASET_NAME} \
            --use_rag \
            --rag_cache_dir ${RAG_CACHE_DIR}

        echo ""
        echo "✅ RAG增强评估完成"
        echo ""
    fi
else
    echo ""
    echo "⏭️  跳过评估"
    echo ""
fi

# ============================================================================
# Step 5: 生成对比报告
# ============================================================================
if [ "$SKIP_EVAL" = false ] && [ "$USE_RAG" = true ]; then
    echo ""
    echo "┌────────────────────────────────────────────────────────────────────────────┐"
    echo "│ Step 5/5: 生成RAG对比报告                                                  │"
    echo "└────────────────────────────────────────────────────────────────────────────┘"
    echo ""

    cd ${LLAVA_DIR}
    mkdir -p ${COMPARISON_DIR}

    python compare_rag_results.py \
        --no_rag_dir ${NO_RAG_OUTPUT_DIR} \
        --with_rag_dir ${WITH_RAG_OUTPUT_DIR} \
        --output_dir ${COMPARISON_DIR} \
        --index ${INDEX}

    echo ""
    echo "✅ 对比报告已生成: ${COMPARISON_DIR}/rag_comparison_report.txt"
    echo ""
fi

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                              全部完成!                                      ║"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║ 输出目录:                                                                   ║"
echo "║   训练: ${LANGSPLAT_DIR}/output/${DATASET_NAME}_${INDEX}_*"
echo "║   实验: ${LANGSPLAT_DIR}/eval_result/${INDEX}/"
echo "║     ├── level0/, level1/, level2/  (渲染结果)"
echo "║     ├── analysis_no_rag/           (无RAG评估)"
if [ "$USE_RAG" = true ]; then
echo "║     ├── analysis_with_rag/         (有RAG评估)"
echo "║     └── comparison/                (对比报告)"
echo "║   RAG:  ${RAG_CACHE_DIR}/${DATASET_NAME}/"
echo "║     ├── rag_storage/               (知识库索引)"
echo "║     └── query_cache/               (预查询缓存)"
else
echo "║     └── analysis_no_rag/           (评估结果)"
fi
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

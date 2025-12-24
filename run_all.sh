#!/bin/bash
# ============================================================================
# 一站式训练-渲染-评估脚本
# ============================================================================
# 用法: bash run_all.sh [OPTIONS]
#
# 可选参数:
#   --dataset_name NAME    数据集名称 (默认: brandenburg_gate)
#   --index NAME           实验索引 (默认: test)
#   --gpu ID               GPU编号 (默认: 3)
#   --iterations NUM       训练迭代数 (默认: 10000)
#   --topk NUM             top-k值 (默认: 4)
#   --skip_train           跳过训练
#   --skip_render          跳过渲染
#   --skip_eval            跳过评估
#   --image_name NAME      评估的图像名称 (默认: 74972815_6880126377)
#   --sampling STRATEGY    采样策略: grid/hierarchical (默认: grid)
#
# 示例:
#   bash run_all.sh                                    # 使用默认参数
#   bash run_all.sh --gpu 0 --iterations 20000        # 指定GPU和迭代数
#   bash run_all.sh --skip_train --skip_render        # 只做评估
# ============================================================================

set -e  # 遇到错误立即退出

# ============================================================================
# 默认参数
# ============================================================================
DATASET_ROOT_PATH=/mnt/data/wangyz/PT
DATASET_NAME=brandenburg_gate
INDEX=test
GPU=3
ITERATIONS=10000
TOPK=4
CHECKPOINT=${ITERATIONS}
CODEBOOK_SIZE=64

# 评估参数
IMAGE_NAME=74972815_6880126377
SAMPLING_STRATEGY=grid
GRID_SIZE=5

# 跳过标志
SKIP_TRAIN=false
SKIP_RENDER=false
SKIP_EVAL=false

# 路径
LANGSPLAT_DIR=/home/wangyz/project/0working/LangSplatV2
LLAVA_DIR=/home/wangyz/project/0working/LangSplatV2/LLaVA-NeXT

# ============================================================================
# 解析命令行参数
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_name) DATASET_NAME="$2"; shift 2 ;;
        --index) INDEX="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; CHECKPOINT="$2"; shift 2 ;;
        --topk) TOPK="$2"; shift 2 ;;
        --image_name) IMAGE_NAME="$2"; shift 2 ;;
        --sampling) SAMPLING_STRATEGY="$2"; shift 2 ;;
        --skip_train) SKIP_TRAIN=true; shift ;;
        --skip_render) SKIP_RENDER=true; shift ;;
        --skip_eval) SKIP_EVAL=true; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ============================================================================
# 打印配置
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                     一站式训练-渲染-评估流程                                ║"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║ 数据集: ${DATASET_NAME}"
echo "║ 索引: ${INDEX}"
echo "║ GPU: ${GPU}"
echo "║ 迭代数: ${ITERATIONS}"
echo "║ Top-K: ${TOPK}"
echo "║ 评估图像: ${IMAGE_NAME}"
echo "║ 采样策略: ${SAMPLING_STRATEGY}"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║ 步骤: $([ "$SKIP_TRAIN" = true ] && echo "跳过训练" || echo "训练") → $([ "$SKIP_RENDER" = true ] && echo "跳过渲染" || echo "渲染") → $([ "$SKIP_EVAL" = true ] && echo "跳过评估" || echo "评估")"
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

    for level in 0 1 2
    do
        if [ $level -eq 0 ]; then
            SCALE_NAME="Small/64px"
        elif [ $level -eq 1 ]; then
            SCALE_NAME="Medium/192px"
        else
            SCALE_NAME="Large/448px"
        fi

        echo ""
        echo ">>> 训练 Level ${level} (${SCALE_NAME}) <<<"
        echo ""

        conda run -n langsplat_v2 python train.py \
            -s ${DATASET_ROOT_PATH}/${DATASET_NAME} \
            -m output/${DATASET_NAME}_${INDEX}_${level} \
            --start_checkpoint ${DATASET_ROOT_PATH}/${DATASET_NAME}/${DATASET_NAME}_vanilla3DGS/chkpnt30000.pth \
            --feature_level ${level} \
            --vq_layer_num 1 \
            --codebook_size ${CODEBOOK_SIZE} \
            --cos_loss \
            -r 2 \
            --topk ${TOPK} \
            --iterations ${ITERATIONS}

        echo ">>> Level ${level} 训练完成 <<<"
    done

    echo ""
    echo "✅ 训练完成"
    echo ""
else
    echo ""
    echo "⏭️  跳过训练"
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

    for level in 0 1 2
    do
        if [ $level -eq 0 ]; then
            SCALE_NAME="Small"
        elif [ $level -eq 1 ]; then
            SCALE_NAME="Medium"
        else
            SCALE_NAME="Large"
        fi

        echo ""
        echo ">>> 渲染 Level ${level} (${SCALE_NAME}) <<<"
        echo ""

        conda run -n langsplat_v2 python render_lerf_llm.py \
            -s ${DATASET_ROOT_PATH}/${DATASET_NAME} \
            -m output/${DATASET_NAME}_${INDEX}_${level} \
            --dataset_name ${DATASET_NAME} \
            --index ${INDEX} \
            --ckpt_root_path output \
            --output_dir eval_result/${DATASET_NAME}_${INDEX}_level${level} \
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
    done

    echo ""
    echo "✅ 渲染完成"
    echo ""
else
    echo ""
    echo "⏭️  跳过渲染"
    echo ""
fi

# ============================================================================
# Step 3: 评估 (llava 环境)
# ============================================================================
if [ "$SKIP_EVAL" = false ]; then
    echo ""
    echo "┌────────────────────────────────────────────────────────────────────────────┐"
    echo "│ Step 3/3: 评估 (conda: llava)                                              │"
    echo "└────────────────────────────────────────────────────────────────────────────┘"
    echo ""

    cd ${LLAVA_DIR}

    echo ">>> 运行问答评估 (${SAMPLING_STRATEGY}采样) <<<"
    echo ""

    CUDA_VISIBLE_DEVICES=${GPU} conda run -n llava python verify_reconstruction_quality.py \
        --image_name ${IMAGE_NAME} \
        --use_semantic_pooling \
        --use_qa_json \
        --sampling_strategy ${SAMPLING_STRATEGY} \
        --grid_size ${GRID_SIZE}

    echo ""
    echo "✅ 评估完成"
    echo ""
else
    echo ""
    echo "⏭️  跳过评估"
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
echo "║   渲染: ${LANGSPLAT_DIR}/eval_result/${DATASET_NAME}_${INDEX}_level*"
echo "║   评估: ${LANGSPLAT_DIR}/eval_result/feature_similarity_analysis/"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

#!/bin/bash
# 多尺度评估配置：评估Small, Medium, Large三个level
ulimit -n 65535   # 当前终端会话临时调大上限
export CUDA_VISIBLE_DEVICES=3  # 使用GPU 3（空闲）
DATASET_ROOT_PATH=/mnt/data/wangyz/PT/
DATASET_NAME=brandenburg_gate
INDEX=test
CHECKPOINT=10000
TOPK=4

# path to lerf_ovs/label
gt_folder=/mnt/data/wangyz/PT/label_llm/
# GT 特征路径（用于可视化对比）
GT_FEATURE_DIR=/mnt/data/wangyz/PT/${DATASET_NAME}/llava_features_3584_multiscale

ROOT_PATH="."

# 评估所有三个level：0=Small, 1=Medium, 2=Large
for level in 0 1 2
do
    # 根据level设置scale名称
    if [ $level -eq 0 ]; then
        SCALE_NAME="Small"
        SCALE_SIZE="64px"
    elif [ $level -eq 1 ]; then
        SCALE_NAME="Medium"
        SCALE_SIZE="192px"
    else
        SCALE_NAME="Large"
        SCALE_SIZE="448px"
    fi

    echo ""
    echo "=========================================="
    echo ">>> 评估 Level ${level} (${SCALE_NAME}/${SCALE_SIZE}) <<<"
    echo "=========================================="
    echo ""

    conda run -n langsplat_v2 python render_lerf_llm.py \
        -s ${DATASET_ROOT_PATH}/${DATASET_NAME} \
        -m "${ROOT_PATH}/output/${DATASET_NAME}_${INDEX}_${level}" \
        --dataset_name ${DATASET_NAME} \
        --index ${INDEX} \
        --ckpt_root_path ${ROOT_PATH}/output \
        --output_dir ${ROOT_PATH}/eval_result/${DATASET_NAME}_${INDEX}_level${level} \
        --mask_thresh 0.4 \
        --json_folder ${gt_folder} \
        --checkpoint ${CHECKPOINT} \
        --include_feature \
        --topk ${TOPK} \
        -r 2 \
        --visualize_comparison \
        --gt_feature_dir ${GT_FEATURE_DIR} \
        --comparison_scale ${SCALE_NAME}

    echo ""
    echo ">>> Level ${level} 评估完成 <<<"
    echo ""
done

echo ""
echo "=========================================="
echo "✅ 所有三个level评估完成！"
echo "=========================================="
echo "评估结果目录："
echo "  - Level 0 (Small):  eval_result/${DATASET_NAME}_${INDEX}_level0"
echo "  - Level 1 (Medium): eval_result/${DATASET_NAME}_${INDEX}_level1"
echo "  - Level 2 (Large):  eval_result/${DATASET_NAME}_${INDEX}_level2"
echo ""
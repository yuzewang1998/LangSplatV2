#!/bin/bash
# 多尺度训练配置：训练Small, Medium, Large三个level
DATASET_ROOT_PATH=/mnt/data/wangyz/PT/
DATASET_NAME=brandenburg_gate
INDEX=test
TOPK=4
ulimit -n 65535   # 当前终端会话临时调大上限

# 训练所有三个level：0=Small, 1=Medium, 2=Large
for level in 0 1 2
do
    echo ""
    echo "=========================================="
    echo ">>> 训练 Level ${level} ($([ $level -eq 0 ] && echo 'Small/64px' || [ $level -eq 1 ] && echo 'Medium/192px' || echo 'Large/448px')) <<<"
    echo "=========================================="
    echo ""
    python train.py \
        -s $DATASET_ROOT_PATH/$DATASET_NAME \
        -m output/${DATASET_NAME}_${INDEX} \
        --start_checkpoint $DATASET_ROOT_PATH/$DATASET_NAME/brandenburg_gate_vanilla3DGS/chkpnt30000.pth \
        --feature_level ${level} \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --cos_loss \
        -r 2 \
        --topk $TOPK \
        --iterations 10000

    echo ""
    echo ">>> Level ${level} 训练完成 <<<"
    echo ""
done

echo ""
echo "=========================================="
echo "✅ 所有三个level训练完成！"
echo "=========================================="
echo "输出目录："
echo "  - Level 0 (Small):  output/${DATASET_NAME}_${INDEX}_0"
echo "  - Level 1 (Medium): output/${DATASET_NAME}_${INDEX}_1"
echo "  - Level 2 (Large):  output/${DATASET_NAME}_${INDEX}_2"
echo "码本文件："
echo "  - llm_codebooks_64_level0_small.pt"
echo "  - llm_codebooks_64_level1_medium.pt"
echo "  - llm_codebooks_64_level2_large.pt"
echo ""
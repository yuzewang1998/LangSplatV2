#!/bin/bash
# 实验: codebook_size=128

cd /home/wangyz/project/0working/LangSplatV2

bash run_all.sh \
    --gpu 1 \
    --codebook_size 128 \
    --topk 8 \
    --exp_id 1224_01_cb128_topK8

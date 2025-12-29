#!/bin/bash
# 实验: codebook_size=128

cd /home/wangyz/project/0working/LangSplatV2

bash run_all.sh \
    --gpu 0 \
    --codebook_size 128 \
    --exp_id 1224_00_cb128

#!/usr/bin/env python3
"""
可视化最终过滤结果
展示所有通过筛选的图片
"""

import csv
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 配置
tsv_path = "/home/wangyz/data/PT/brandenburg_gate/brandenburg_filtered_v5.tsv"
image_dir = "/home/wangyz/data/PT/brandenburg_gate/dense/images"
output_path = "/home/wangyz/project/0working/LangSplatV2/visualizations_v5/final_filtered_all.png"

# 创建输出目录
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 读取所有文件名
filenames = []
with open(tsv_path, 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        filenames.append(row['filename'])

print(f"总计 {len(filenames)} 张图片")

# 计算网格大小
num_images = len(filenames)
cols = 15
rows = (num_images + cols - 1) // cols

# 创建大图
fig = plt.figure(figsize=(cols * 2, rows * 2))
fig.suptitle(f'Final Filtered Images (Total: {num_images})', fontsize=16, fontweight='bold')

for idx, filename in enumerate(sorted(filenames)):
    ax = fig.add_subplot(rows, cols, idx + 1)

    img_path = os.path.join(image_dir, filename)
    try:
        img = mpimg.imread(img_path)
        ax.imshow(img)
    except Exception as e:
        ax.text(0.5, 0.5, 'Error', ha='center', va='center')

    ax.axis('off')
    # 不显示文件名，保持简洁

plt.tight_layout()
plt.savefig(output_path, dpi=100, bbox_inches='tight')
print(f"已保存到: {output_path}")
plt.close()

#!/usr/bin/env python3
"""
创建高质量交集数据集
从 v3 和旧过滤结果中提取共同保留的图片
"""

import csv
from pathlib import Path

# 文件路径
data_dir = Path("/home/wangyz/data/PT/brandenburg_gate")
v3_tsv = data_dir / "brandenburg_filtered_v3.tsv"
old_tsv = data_dir / "brandenburg_filtered.tsv"
output_tsv = data_dir / "brandenburg_filtered_intersection.tsv"

# 读取两个 TSV 文件
def load_tsv_dict(tsv_path):
    """加载 TSV，返回 filename -> row 的字典"""
    result = {}
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            result[row['filename']] = row
    return result

print("加载过滤结果...")
v3_data = load_tsv_dict(v3_tsv)
old_data = load_tsv_dict(old_tsv)

print(f"v3: {len(v3_data)} 张")
print(f"旧过滤: {len(old_data)} 张")

# 计算交集
common_files = set(v3_data.keys()) & set(old_data.keys())
print(f"交集: {len(common_files)} 张")

# 写入交集 TSV
with open(output_tsv, 'w', newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=['filename', 'id', 'split', 'dataset'],
        delimiter='\t'
    )
    writer.writeheader()

    for filename in sorted(common_files):
        # 使用 v3 的数据（两边数据应该一致）
        writer.writerow(v3_data[filename])

print(f"\n交集已保存到: {output_tsv}")
print(f"总计: {len(common_files)} 张高质量图片")

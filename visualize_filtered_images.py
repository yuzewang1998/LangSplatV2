#!/usr/bin/env python3
"""
可视化过滤后的图像
展示通过筛选的图片样本，以及新旧过滤结果的对比
"""

import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec


def load_tsv_filenames(tsv_path: str) -> Set[str]:
    """加载 TSV 文件中的文件名列表"""
    filenames = set()
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            filenames.add(row['filename'])
    return filenames


def sample_images(
    filenames: Set[str],
    image_dir: str,
    num_samples: int = 20,
    seed: int = 42
) -> List[str]:
    """随机采样图片"""
    random.seed(seed)
    # 过滤出实际存在的图片
    existing_files = [f for f in filenames if os.path.exists(os.path.join(image_dir, f))]
    num_samples = min(num_samples, len(existing_files))
    return random.sample(existing_files, num_samples)


def visualize_grid(
    image_paths: List[str],
    title: str,
    output_path: str,
    cols: int = 5
):
    """创建图片网格可视化"""
    num_images = len(image_paths)
    rows = (num_images + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 3, rows * 3))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for idx, img_path in enumerate(image_paths):
        ax = fig.add_subplot(rows, cols, idx + 1)

        try:
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
            # 添加文件名作为子标题
            filename = os.path.basename(img_path)
            ax.set_title(filename[:20], fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error\n{str(e)}',
                   ha='center', va='center', fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"已保存可视化结果: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="可视化过滤后的图像"
    )
    parser.add_argument(
        "--new_tsv",
        type=str,
        default="/home/wangyz/data/PT/brandenburg_gate/brandenburg_filtered_new.tsv",
        help="新过滤结果 TSV 文件"
    )
    parser.add_argument(
        "--old_tsv",
        type=str,
        default="/home/wangyz/data/PT/brandenburg_gate/brandenburg_filtered.tsv",
        help="旧过滤结果 TSV 文件"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/home/wangyz/data/PT/brandenburg_gate/dense/images",
        help="图像目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/wangyz/project/0working/LangSplatV2/visualizations",
        help="可视化输出目录"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="每类采样的图片数量"
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("加载过滤结果...")
    new_filenames = load_tsv_filenames(args.new_tsv)
    old_filenames = load_tsv_filenames(args.old_tsv)

    print(f"新过滤结果: {len(new_filenames)} 张")
    print(f"旧过滤结果: {len(old_filenames)} 张")

    # 计算交集和差集
    common = new_filenames & old_filenames
    only_new = new_filenames - old_filenames
    only_old = old_filenames - new_filenames

    print(f"共同保留: {len(common)} 张")
    print(f"仅新过滤保留: {len(only_new)} 张")
    print(f"仅旧过滤保留: {len(only_old)} 张")
    print()

    # 1. 可视化新过滤结果的随机样本
    print(f"可视化新过滤结果（随机采样 {args.num_samples} 张）...")
    new_samples = sample_images(new_filenames, args.image_dir, args.num_samples)
    new_sample_paths = [os.path.join(args.image_dir, f) for f in new_samples]
    visualize_grid(
        new_sample_paths,
        f"新过滤结果样本 (Total: {len(new_filenames)})",
        os.path.join(args.output_dir, "filtered_new_samples.png")
    )

    # 2. 可视化共同保留的图片
    if len(common) > 0:
        print(f"可视化共同保留的图片（随机采样 {min(args.num_samples, len(common))} 张）...")
        common_samples = sample_images(common, args.image_dir, args.num_samples)
        common_sample_paths = [os.path.join(args.image_dir, f) for f in common_samples]
        visualize_grid(
            common_sample_paths,
            f"新旧过滤共同保留 (Total: {len(common)})",
            os.path.join(args.output_dir, "filtered_common.png")
        )

    # 3. 可视化仅新过滤保留的图片
    if len(only_new) > 0:
        print(f"可视化仅新过滤保留的图片（随机采样 {min(args.num_samples, len(only_new))} 张）...")
        only_new_samples = sample_images(only_new, args.image_dir, args.num_samples)
        only_new_paths = [os.path.join(args.image_dir, f) for f in only_new_samples]
        visualize_grid(
            only_new_paths,
            f"仅新过滤保留 (Total: {len(only_new)})",
            os.path.join(args.output_dir, "filtered_only_new.png")
        )

    # 4. 可视化仅旧过滤保留的图片
    if len(only_old) > 0:
        print(f"可视化仅旧过滤保留的图片（随机采样 {min(args.num_samples, len(only_old))} 张）...")
        only_old_samples = sample_images(only_old, args.image_dir, args.num_samples)
        only_old_paths = [os.path.join(args.image_dir, f) for f in only_old_samples]
        visualize_grid(
            only_old_paths,
            f"仅旧过滤保留 (Total: {len(only_old)})",
            os.path.join(args.output_dir, "filtered_only_old.png")
        )

    print()
    print("=" * 80)
    print("可视化完成！")
    print(f"结果保存在: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

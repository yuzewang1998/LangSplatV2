#!/usr/bin/env python3
"""可视化过滤后的图像。"""

import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Set

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval_result" / "filtering"


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


def visualize_all_images(
    filenames: Set[str],
    image_dir: str,
    output_path: str,
    cols: int = 15
):
    """创建全部过滤结果的总览图。"""
    existing_files = sorted(
        f for f in filenames if os.path.exists(os.path.join(image_dir, f))
    )
    image_paths = [os.path.join(image_dir, f) for f in existing_files]
    visualize_grid(
        image_paths,
        f"全部过滤结果 (Total: {len(existing_files)})",
        output_path,
        cols=cols
    )


def main():
    parser = argparse.ArgumentParser(
        description="可视化过滤后的图像"
    )
    parser.add_argument(
        "--filtered_tsv",
        type=str,
        default="/home/wangyz/data/PT/brandenburg_gate/brandenburg_filtered_v6.tsv",
        help="过滤结果 TSV 文件"
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
        default=str(DEFAULT_OUTPUT_DIR),
        help="可视化输出目录"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="每类采样的图片数量"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机采样种子"
    )
    parser.add_argument(
        "--save_full_grid",
        action="store_true",
        help="额外保存全部通过图片的大总览图"
    )
    parser.add_argument(
        "--full_grid_cols",
        type=int,
        default=15,
        help="总览图列数"
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("加载过滤结果...")
    filtered_filenames = load_tsv_filenames(args.filtered_tsv)
    print(f"过滤结果: {len(filtered_filenames)} 张")
    print()

    print(f"可视化过滤结果（随机采样 {args.num_samples} 张）...")
    filtered_samples = sample_images(
        filtered_filenames, args.image_dir, args.num_samples, seed=args.seed
    )
    filtered_sample_paths = [os.path.join(args.image_dir, f) for f in filtered_samples]
    visualize_grid(
        filtered_sample_paths,
        f"过滤结果样本 (Total: {len(filtered_filenames)})",
        os.path.join(args.output_dir, "filtered_samples.png")
    )

    if args.save_full_grid:
        print("生成全部过滤结果总览图...")
        visualize_all_images(
            filtered_filenames,
            args.image_dir,
            os.path.join(args.output_dir, "filtered_all.png"),
            cols=args.full_grid_cols
        )

    print()
    print("=" * 80)
    print("可视化完成！")
    print(f"结果保存在: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

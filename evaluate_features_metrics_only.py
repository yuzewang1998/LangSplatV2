#!/usr/bin/env python3
"""
仅计算量化指标的版本（不做PCA可视化）
用于快速评估渲染特征质量
"""

import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

# 导入 LLaVA-NeXT 的 crop codec
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LLaVA-NeXT'))
try:
    from crop_feature_codec import CropFeatureCodec
    CODEC_AVAILABLE = True
except ImportError:
    print("Warning: crop_feature_codec not available")
    CODEC_AVAILABLE = False


def load_gt_feature_map(gt_path: str, scale_name: str = 'Medium'):
    """加载 GT 特征并解码为完整的 feature map"""
    if not CODEC_AVAILABLE:
        raise RuntimeError("CropFeatureCodec not available")

    data = torch.load(gt_path, map_location='cpu')
    scale_data = data['feature_maps'][scale_name]

    # 使用 codec 解码
    feature_map, mask = CropFeatureCodec.decode_to_full_map(
        scale_data, overlap_mode='average'
    )

    # 转换为 numpy
    feature_map = feature_map.numpy() if isinstance(feature_map, torch.Tensor) else feature_map
    mask = (mask > 0).numpy() if isinstance(mask, torch.Tensor) else (mask > 0)

    return feature_map, mask


def load_rendered_feature_map(rendered_path: str):
    """加载渲染特征"""
    data = torch.load(rendered_path, map_location='cpu')

    if isinstance(data, torch.Tensor):
        # [1, H, W, C] -> [H, W, C]
        if data.dim() == 4 and data.shape[0] == 1:
            data = data.squeeze(0)
        feature_map = data.numpy() if isinstance(data, torch.Tensor) else data
    else:
        raise ValueError("Invalid rendered feature file format")

    return feature_map


def compute_metrics(rendered_feat: np.ndarray, gt_feat: np.ndarray, mask: np.ndarray = None):
    """计算渲染特征与 GT 特征的量化指标"""
    H, W, C = rendered_feat.shape

    # 展平
    rendered_flat = rendered_feat.reshape(-1, C)
    gt_flat = gt_feat.reshape(-1, C)

    # 如果有 mask，只计算有效区域
    if mask is not None:
        mask_flat = mask.reshape(-1)
        rendered_flat = rendered_flat[mask_flat]
        gt_flat = gt_flat[mask_flat]

    # 过滤无效值
    finite_mask = np.isfinite(rendered_flat).all(axis=1) & np.isfinite(gt_flat).all(axis=1)
    rendered_flat = rendered_flat[finite_mask]
    gt_flat = gt_flat[finite_mask]

    # 过滤全零特征
    valid_mask = (np.linalg.norm(rendered_flat, axis=1) > 1e-6) & (np.linalg.norm(gt_flat, axis=1) > 1e-6)
    rendered_flat = rendered_flat[valid_mask]
    gt_flat = gt_flat[valid_mask]

    if len(rendered_flat) == 0:
        return {
            'cosine_similarity': 0.0,
            'l1_distance': float('inf'),
            'l2_distance': float('inf'),
            'valid_pixels': 0,
            'total_pixels': H * W,
            'coverage': 0.0
        }

    # 余弦相似度
    rendered_norm = rendered_flat / (np.linalg.norm(rendered_flat, axis=1, keepdims=True) + 1e-8)
    gt_norm = gt_flat / (np.linalg.norm(gt_flat, axis=1, keepdims=True) + 1e-8)
    cosine_sim = (rendered_norm * gt_norm).sum(axis=1).mean()

    # L1 距离
    l1_dist = np.abs(rendered_flat - gt_flat).mean()

    # L2 距离
    l2_dist = np.sqrt(((rendered_flat - gt_flat) ** 2).sum(axis=1)).mean()

    return {
        'cosine_similarity': float(cosine_sim),
        'l1_distance': float(l1_dist),
        'l2_distance': float(l2_dist),
        'valid_pixels': len(rendered_flat),
        'total_pixels': H * W,
        'coverage': len(rendered_flat) / (H * W)
    }


def evaluate_features(rendered_path: str, gt_path: str, scale_name: str = 'Medium'):
    """评估渲染特征质量"""
    print(f"\n{'='*60}")
    print(f"Feature Evaluation")
    print(f"{'='*60}")
    print(f"Rendered: {rendered_path}")
    print(f"GT: {gt_path}")
    print(f"Scale: {scale_name}")
    print()

    # 加载特征
    print("Loading rendered features...")
    rendered_feat = load_rendered_feature_map(rendered_path)
    print(f"  Shape: {rendered_feat.shape}")

    print("\nLoading GT features...")
    gt_feat, gt_mask = load_gt_feature_map(gt_path, scale_name)
    print(f"  Shape: {gt_feat.shape}")
    print(f"  Mask coverage: {gt_mask.mean():.2%}")

    # 确保尺寸一致
    if rendered_feat.shape[:2] != gt_feat.shape[:2]:
        print("\nResizing GT feature to match rendered feature...")
        gt_feat_tensor = torch.from_numpy(gt_feat).permute(2, 0, 1).unsqueeze(0)
        gt_mask_tensor = torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        target_size = rendered_feat.shape[:2]
        gt_feat_tensor = F.interpolate(gt_feat_tensor, size=target_size, mode='bilinear', align_corners=False)
        gt_mask_tensor = F.interpolate(gt_mask_tensor, size=target_size, mode='nearest')

        gt_feat = gt_feat_tensor.squeeze(0).permute(1, 2, 0).numpy()
        gt_mask = (gt_mask_tensor.squeeze(0).squeeze(0) > 0.5).numpy()

    # 计算指标
    print("\nComputing metrics...")
    metrics = compute_metrics(rendered_feat, gt_feat, gt_mask)

    # 打印结果
    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    print(f"Cosine Similarity:  {metrics['cosine_similarity']:.4f}")
    print(f"L1 Distance:        {metrics['l1_distance']:.4f}")
    print(f"L2 Distance:        {metrics['l2_distance']:.4f}")
    print(f"Valid Pixels:       {metrics['valid_pixels']:,} / {metrics['total_pixels']:,}")
    print(f"Coverage:           {metrics['coverage']:.2%}")
    print(f"{'='*60}\n")

    return metrics


def evaluate_all_frames(output_dir: str, gt_dir: str, scale_name: str = 'Medium'):
    """评估所有帧"""
    output_path = Path(output_dir)
    frame_dirs = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('frame_')])

    if not frame_dirs:
        print(f"No frame directories found in {output_dir}")
        return

    all_metrics = []

    for frame_dir in frame_dirs:
        frame_name = frame_dir.name
        frame_idx = int(frame_name.split('_')[1])

        # 构建路径
        rendered_path = frame_dir / f'feature_map_{frame_name}.pt'
        gt_path = Path(gt_dir) / f'{frame_name}.pth'

        if not rendered_path.exists():
            print(f"Skipping {frame_name}: rendered feature not found")
            continue

        if not gt_path.exists():
            print(f"Skipping {frame_name}: GT feature not found")
            continue

        # 评估
        metrics = evaluate_features(str(rendered_path), str(gt_path), scale_name)
        metrics['frame'] = frame_name
        all_metrics.append(metrics)

    # 打印汇总
    if all_metrics:
        print(f"\n{'='*60}")
        print("Summary Statistics:")
        print(f"{'='*60}")
        print(f"Total frames evaluated: {len(all_metrics)}")

        avg_cosine = np.mean([m['cosine_similarity'] for m in all_metrics])
        avg_l1 = np.mean([m['l1_distance'] for m in all_metrics])
        avg_l2 = np.mean([m['l2_distance'] for m in all_metrics])
        avg_coverage = np.mean([m['coverage'] for m in all_metrics])

        print(f"Average Cosine Similarity: {avg_cosine:.4f}")
        print(f"Average L1 Distance:       {avg_l1:.4f}")
        print(f"Average L2 Distance:       {avg_l2:.4f}")
        print(f"Average Coverage:          {avg_coverage:.2%}")
        print(f"{'='*60}\n")

    return all_metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate rendered features (metrics only, no visualization)')
    parser.add_argument('--rendered_path', type=str, help='Path to single rendered feature file')
    parser.add_argument('--gt_path', type=str, help='Path to single GT feature file')
    parser.add_argument('--output_dir', type=str, help='Directory containing all rendered features')
    parser.add_argument('--gt_dir', type=str, help='Directory containing all GT features')
    parser.add_argument('--scale', type=str, default='Medium', choices=['Small', 'Medium', 'Large'])

    args = parser.parse_args()

    if args.rendered_path and args.gt_path:
        # 单帧评估
        evaluate_features(args.rendered_path, args.gt_path, args.scale)
    elif args.output_dir and args.gt_dir:
        # 批量评估
        evaluate_all_frames(args.output_dir, args.gt_dir, args.scale)
    else:
        parser.print_help()

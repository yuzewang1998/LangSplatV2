#!/usr/bin/env python3
"""
最简单的特征对比可视化 - 直接用前3维作为RGB
不需要PCA，速度极快！
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F

def load_gt_feature_map(gt_path: str, scale_name: str = 'Medium'):
    """加载 GT 特征（使用 no_interp 方法，避免 averaging 和 bilinear interpolation）"""
    data = torch.load(gt_path, map_location='cpu')

    if 'feature_maps' not in data:
        raise ValueError(f"文件格式错误：缺少 'feature_maps' 键")

    feature_maps_data = data['feature_maps']

    # 尝试获取指定 scale
    if scale_name not in feature_maps_data:
        available_scales = list(feature_maps_data.keys())
        print(f"  Warning: {scale_name} not available. Available scales: {available_scales}")
        # 回退到第一个可用的
        scale_name = available_scales[0]
        print(f"  Using {scale_name} instead")

    scale_data = feature_maps_data[scale_name]
    crop_features = scale_data['crop_features']
    image_size = scale_data['image_size']  # (W, H)

    H, W = image_size[1], image_size[0]

    # 如果只有一个 crop，直接返回
    if len(crop_features) == 1:
        feature = crop_features[0]['feature']  # [H_crop, W_crop, C]
        feature_np = feature.numpy() if isinstance(feature, torch.Tensor) else np.asarray(feature)
        H_crop, W_crop, C = feature_np.shape

        # 生成全覆盖 mask
        mask = np.ones((H_crop, W_crop), dtype=bool)

        return feature_np, mask

    # 多个 crops: 使用 last-write-wins（不做 averaging）
    C = crop_features[0]['feature'].shape[-1]
    feature_map = np.zeros((H, W, C), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)

    for crop_data in crop_features:
        feature = crop_data['feature']
        bbox = crop_data['bbox']  # (x, y, w, h)

        feature_np = feature.numpy() if isinstance(feature, torch.Tensor) else np.asarray(feature)
        x, y, w, h = bbox
        x2, y2 = x + w, y + h

        crop_h, crop_w = int(y2 - y), int(x2 - x)
        if feature_np.shape[0] != crop_h or feature_np.shape[1] != crop_w:
            # 使用 nearest neighbor resize（避免 bilinear 平滑）
            feature_torch = torch.from_numpy(feature_np).permute(2, 0, 1).unsqueeze(0)
            feature_torch = F.interpolate(
                feature_torch, size=(crop_h, crop_w), mode='nearest'
            )
            feature_np = feature_torch.squeeze(0).permute(1, 2, 0).numpy()

        # 直接覆盖（last-write-wins，不做 averaging）
        feature_map[y:y2, x:x2] = feature_np
        mask[y:y2, x:x2] = True

    return feature_map, mask


def load_rendered_feature_map(rendered_path: str):
    """加载渲染特征"""
    data = torch.load(rendered_path, map_location='cpu')
    if isinstance(data, torch.Tensor):
        if data.dim() == 4 and data.shape[0] == 1:
            data = data.squeeze(0)
        feature_map = data.numpy() if isinstance(data, torch.Tensor) else data
    else:
        raise ValueError("Invalid rendered feature file format")
    return feature_map


def compute_metrics(rendered_feat: np.ndarray, gt_feat: np.ndarray, mask: np.ndarray = None):
    """计算量化指标"""
    H, W, C = rendered_feat.shape

    rendered_flat = rendered_feat.reshape(-1, C)
    gt_flat = gt_feat.reshape(-1, C)

    if mask is not None:
        mask_flat = mask.reshape(-1)
        rendered_flat = rendered_flat[mask_flat]
        gt_flat = gt_flat[mask_flat]

    # 过滤无效值
    finite_mask = np.isfinite(rendered_flat).all(axis=1) & np.isfinite(gt_flat).all(axis=1)
    rendered_flat = rendered_flat[finite_mask]
    gt_flat = gt_flat[finite_mask]

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

    # L1 和 L2 距离
    l1_dist = np.abs(rendered_flat - gt_flat).mean()
    l2_dist = np.sqrt(((rendered_flat - gt_flat) ** 2).sum(axis=1)).mean()

    return {
        'cosine_similarity': float(cosine_sim),
        'l1_distance': float(l1_dist),
        'l2_distance': float(l2_dist),
        'valid_pixels': len(rendered_flat),
        'total_pixels': H * W,
        'coverage': len(rendered_flat) / (H * W)
    }


def features_to_rgb_simple(features: np.ndarray, mask: np.ndarray = None):
    """
    最简单的特征可视化：直接用前3个维度作为RGB
    """
    H, W, C = features.shape
    print(f"  Feature shape: {features.shape}")

    # 取前3个维度
    rgb_features = features[:, :, :3]  # [H, W, 3]

    # 归一化到[0, 1]
    if mask is not None:
        # 只考虑有效区域的统计
        valid_features = rgb_features[mask]
        for i in range(3):
            channel = rgb_features[:, :, i]
            valid_channel = channel[mask]

            low = np.percentile(valid_channel, 2)
            high = np.percentile(valid_channel, 98)

            if high - low > 1e-6:
                channel_norm = np.clip((channel - low) / (high - low), 0, 1)
            else:
                channel_norm = np.ones_like(channel) * 0.5

            rgb_features[:, :, i] = channel_norm

        # 无效区域设为灰色
        rgb_features[~mask] = 0.5
    else:
        # 全局归一化
        for i in range(3):
            channel = rgb_features[:, :, i]
            low = np.percentile(channel, 2)
            high = np.percentile(channel, 98)

            if high - low > 1e-6:
                rgb_features[:, :, i] = np.clip((channel - low) / (high - low), 0, 1)
            else:
                rgb_features[:, :, i] = 0.5

    # 转uint8
    rgb_image = (rgb_features * 255).astype(np.uint8)

    return rgb_image


def visualize_comparison(rendered_path: str, gt_path: str, image_path: str,
                        output_path: str, scale_name: str = 'Medium'):
    """最简单的对比可视化（不用PCA）"""
    print(f"\n{'='*60}")
    print("Feature Comparison Visualization (Simple RGB)")
    print(f"{'='*60}")
    print(f"Rendered: {rendered_path}")
    print(f"GT: {gt_path}")
    print(f"Image: {image_path}")
    print()

    # 加载特征
    print("1. Loading rendered features...")
    rendered_feat = load_rendered_feature_map(rendered_path)
    print(f"   Shape: {rendered_feat.shape}")

    print("\n2. Loading GT features...")
    gt_feat, gt_mask = load_gt_feature_map(gt_path, scale_name)
    print(f"   Shape: {gt_feat.shape}")
    print(f"   Mask coverage: {gt_mask.mean():.2%}")

    # 确保尺寸一致
    if rendered_feat.shape[:2] != gt_feat.shape[:2]:
        print("\n3. Resizing GT feature...")
        gt_feat_tensor = torch.from_numpy(gt_feat).permute(2, 0, 1).unsqueeze(0)
        gt_mask_tensor = torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        target_size = rendered_feat.shape[:2]
        gt_feat_tensor = F.interpolate(gt_feat_tensor, size=target_size, mode='bilinear', align_corners=False)
        gt_mask_tensor = F.interpolate(gt_mask_tensor, size=target_size, mode='nearest')

        gt_feat = gt_feat_tensor.squeeze(0).permute(1, 2, 0).numpy()
        gt_mask = (gt_mask_tensor.squeeze(0).squeeze(0) > 0.5).numpy()

    # 加载原图
    print("\n4. Loading original image...")
    original_image = Image.open(image_path).convert('RGB')
    img_resized = original_image.resize((rendered_feat.shape[1], rendered_feat.shape[0]), Image.LANCZOS)
    img_array = np.array(img_resized)

    # 计算指标
    print("\n5. Computing metrics...")
    metrics = compute_metrics(rendered_feat, gt_feat, gt_mask)
    for k, v in metrics.items():
        if isinstance(v, float) and v != float('inf'):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")

    # 简单RGB可视化 - 直接用前3维
    print("\n6. Generating RGB visualizations (first 3 dimensions)...")
    print("   GT features:")
    gt_rgb = features_to_rgb_simple(gt_feat, gt_mask)

    print("   Rendered features:")
    rendered_rgb = features_to_rgb_simple(rendered_feat, None)

    # 差异图
    print("\n7. Computing difference map...")
    diff_map = np.sqrt(((rendered_feat - gt_feat) ** 2).sum(axis=2))
    valid_diff = diff_map[gt_mask]
    if len(valid_diff) > 0:
        vmin, vmax = np.percentile(valid_diff, [2, 98])
        diff_map_norm = np.clip((diff_map - vmin) / (vmax - vmin + 1e-8), 0, 1)
    else:
        diff_map_norm = diff_map

    # 创建可视化
    print("\n8. Creating visualization...")
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)

    # 第一行
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_array)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gt_rgb)
    ax2.set_title(f'GT Features (First 3 Dims)\nScale: {scale_name}, Coverage: {gt_mask.mean():.1%}',
                  fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(rendered_rgb)
    ax3.set_title('Rendered Features (First 3 Dims)', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # 第二行
    ax4 = fig.add_subplot(gs[1, 0])
    overlay_gt = img_array.astype(float) / 255.0
    feature_overlay_gt = gt_rgb.astype(float) / 255.0
    alpha_gt = gt_mask.astype(float) * 0.6
    blended_gt = overlay_gt.copy()
    for c in range(3):
        blended_gt[:, :, c] = overlay_gt[:, :, c] * (1 - alpha_gt) + feature_overlay_gt[:, :, c] * alpha_gt
    ax4.imshow(blended_gt)
    ax4.set_title('GT Features Overlay', fontsize=14, fontweight='bold')
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    blended_rendered = img_array.astype(float) / 255.0 * 0.4 + rendered_rgb.astype(float) / 255.0 * 0.6
    ax5.imshow(blended_rendered)
    ax5.set_title('Rendered Features Overlay', fontsize=14, fontweight='bold')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    im = ax6.imshow(diff_map_norm, cmap='hot', vmin=0, vmax=1)
    ax6.set_title('Feature Difference (L2 Distance)', fontsize=14, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

    # 添加指标文本
    metrics_text = f"""Metrics:
━━━━━━━━━━━━━━━━━━━━━━
Cosine Similarity: {metrics['cosine_similarity']:.4f}
L1 Distance: {metrics['l1_distance']:.4f}
L2 Distance: {metrics['l2_distance']:.4f}
Valid Pixels: {metrics['valid_pixels']:,} / {metrics['total_pixels']:,}
Coverage: {metrics['coverage']:.2%}
━━━━━━━━━━━━━━━━━━━━━━
Note: Using first 3 dimensions as RGB (no PCA)
"""

    fig.text(0.5, 0.02, metrics_text, ha='center', va='bottom',
             fontsize=12, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    frame_name = Path(rendered_path).parent.name
    fig.suptitle(f'Feature Comparison: {frame_name}', fontsize=16, fontweight='bold', y=0.98)

    # 保存
    print(f"\n9. Saving to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✅ Done!")
    print(f"{'='*60}\n")

    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--rendered_path', type=str, required=True)
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--scale', type=str, default='Medium', choices=['Small', 'Medium', 'Large'])

    args = parser.parse_args()

    if args.output_path is None:
        frame_name = Path(args.rendered_path).parent.name
        output_dir = Path(args.rendered_path).parent
        args.output_path = str(output_dir / f'{frame_name}_comparison.png')

    visualize_comparison(
        args.rendered_path,
        args.gt_path,
        args.image_path,
        args.output_path,
        args.scale
    )

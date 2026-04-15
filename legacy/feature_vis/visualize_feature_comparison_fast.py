#!/usr/bin/env python3
"""
快速版本的特征对比可视化工具
使用独立PCA（不使用全局PCA）以加快速度
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import torch.nn.functional as F

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

    if 'feature_maps' not in data:
        raise ValueError("Invalid GT feature file format")

    if scale_name not in data['feature_maps']:
        raise ValueError(f"Scale {scale_name} not found in GT features")

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


def compute_metrics_fast(rendered_feat: np.ndarray, gt_feat: np.ndarray, mask: np.ndarray = None):
    """快速计算量化指标"""
    H, W, C = rendered_feat.shape

    # 展平
    rendered_flat = rendered_feat.reshape(-1, C)
    gt_flat = gt_feat.reshape(-1, C)

    # 如果有 mask，只计算有效区域
    if mask is not None:
        mask_flat = mask.reshape(-1)
        rendered_flat = rendered_flat[mask_flat]
        gt_flat = gt_flat[mask_flat]

    # 过滤无效值和全零特征
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
            'valid_pixels': 0
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


def pca_to_rgb_fast(features: np.ndarray, mask: np.ndarray = None, max_samples: int = 10000):
    """
    快速版PCA-RGB转换：使用独立PCA，更少的采样
    """
    H, W, C = features.shape

    # 展平特征
    features_flat = features.reshape(-1, C)

    # 过滤无效值
    if mask is not None:
        mask_flat = mask.reshape(-1)
        valid_features = features_flat[mask_flat]
    else:
        valid_features = features_flat
        mask_flat = np.ones(H * W, dtype=bool)

    # 过滤NaN/Inf和全零
    valid_mask = np.isfinite(valid_features).all(axis=1) & (np.linalg.norm(valid_features, axis=1) > 1e-6)
    valid_features = valid_features[valid_mask]

    if len(valid_features) < 10:
        print("Warning: Not enough valid features for PCA")
        return np.zeros((H, W, 3), dtype=np.uint8)

    # 标准化
    mean = valid_features.mean(axis=0)
    std = valid_features.std(axis=0) + 1e-8
    valid_features_norm = (valid_features - mean) / std

    # 标准化所有特征
    features_flat_norm = (features_flat - mean) / std

    # PCA（使用少量采样）
    pca = PCA(n_components=3, svd_solver='randomized', random_state=42)

    n_samples = min(len(valid_features_norm), max_samples)
    if len(valid_features_norm) > n_samples:
        sample_idx = np.random.choice(len(valid_features_norm), n_samples, replace=False)
        pca.fit(valid_features_norm[sample_idx])
    else:
        pca.fit(valid_features_norm)

    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # 转换所有像素
    pca_features = pca.transform(features_flat_norm)

    # 归一化到[0,1]
    rgb_normalized = np.zeros_like(pca_features)
    for i in range(3):
        channel = pca_features[:, i]
        valid_channel = channel[mask_flat]
        valid_channel = valid_channel[np.isfinite(valid_channel)]

        if len(valid_channel) > 0:
            low = np.percentile(valid_channel, 2)
            high = np.percentile(valid_channel, 98)

            if high - low > 1e-6:
                normalized = (channel - low) / (high - low)
            else:
                normalized = np.zeros_like(channel)

            rgb_normalized[:, i] = np.clip(normalized, 0, 1)

    # Reshape回图像
    rgb_image = rgb_normalized.reshape(H, W, 3)

    # 无效区域设为灰色
    if mask is not None:
        rgb_image[~mask] = 0.5

    # 转uint8
    rgb_image = (rgb_image * 255).astype(np.uint8)

    return rgb_image


def visualize_feature_comparison_fast(rendered_path: str, gt_path: str, image_path: str,
                                       output_path: str, scale_name: str = 'Medium'):
    """快速版特征对比可视化（使用独立PCA）"""
    print(f"Loading features...")
    print(f"  Rendered: {rendered_path}")
    print(f"  GT: {gt_path}")
    print(f"  Image: {image_path}")

    # 加载特征
    print("\n1. Loading rendered features...")
    rendered_feat = load_rendered_feature_map(rendered_path)
    print(f"   Shape: {rendered_feat.shape}")

    print("\n2. Loading GT features...")
    gt_feat, gt_mask = load_gt_feature_map(gt_path, scale_name)
    print(f"   Shape: {gt_feat.shape}")
    print(f"   Mask coverage: {gt_mask.mean():.2%}")

    # 确保尺寸一致
    if rendered_feat.shape[:2] != gt_feat.shape[:2]:
        print(f"\n3. Resizing GT feature to match rendered feature...")
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

    # 计算量化指标
    print("\n5. Computing metrics...")
    metrics = compute_metrics_fast(rendered_feat, gt_feat, gt_mask)
    for k, v in metrics.items():
        if isinstance(v, float) and v != float('inf'):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")

    # PCA可视化
    print("\n6. Generating PCA-RGB visualizations...")
    print("   GT features PCA...")
    gt_rgb = pca_to_rgb_fast(gt_feat, gt_mask, max_samples=5000)

    print("   Rendered features PCA...")
    rendered_rgb = pca_to_rgb_fast(rendered_feat, None, max_samples=5000)

    # 计算差异图
    print("\n7. Computing difference map...")
    diff_map = np.sqrt(((rendered_feat - gt_feat) ** 2).sum(axis=2))

    diff_map_vis = diff_map.copy()
    if gt_mask is not None:
        valid_diff = diff_map[gt_mask]
        if len(valid_diff) > 0:
            vmin, vmax = np.percentile(valid_diff, [2, 98])
            diff_map_vis = np.clip((diff_map - vmin) / (vmax - vmin + 1e-8), 0, 1)

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
    ax2.set_title(f'GT Features (PCA-RGB)\nScale: {scale_name}, Coverage: {gt_mask.mean():.1%}',
                  fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(rendered_rgb)
    ax3.set_title(f'Rendered Features (PCA-RGB)\nCoverage: 100%',
                  fontsize=14, fontweight='bold')
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
    overlay_rendered = img_array.astype(float) / 255.0
    feature_overlay_rendered = rendered_rgb.astype(float) / 255.0
    blended_rendered = overlay_rendered * 0.4 + feature_overlay_rendered * 0.6
    ax5.imshow(blended_rendered)
    ax5.set_title('Rendered Features Overlay', fontsize=14, fontweight='bold')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    im = ax6.imshow(diff_map_vis, cmap='hot', vmin=0, vmax=1)
    ax6.set_title('Feature Difference (L2 Distance)', fontsize=14, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

    # 添加量化指标文本
    metrics_text = f"""Quantitative Metrics:
━━━━━━━━━━━━━━━━━━━━━━
Cosine Similarity: {metrics['cosine_similarity']:.4f}
L1 Distance: {metrics['l1_distance']:.4f}
L2 Distance: {metrics['l2_distance']:.4f}
Valid Pixels: {metrics['valid_pixels']:,} / {metrics['total_pixels']:,}
Coverage: {metrics['coverage']:.2%}
━━━━━━━━━━━━━━━━━━━━━━
Note: Using independent PCA per feature (fast mode)
"""

    fig.text(0.5, 0.02, metrics_text, ha='center', va='bottom',
             fontsize=12, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 添加总标题
    frame_name = Path(rendered_path).parent.name
    fig.suptitle(f'Feature Comparison: {frame_name}', fontsize=16, fontweight='bold', y=0.98)

    # 保存
    print(f"\n9. Saving to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✅ Visualization saved successfully!")
    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fast Feature Comparison Visualization')
    parser.add_argument('--rendered_path', type=str, required=True)
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--scale', type=str, default='Medium',
                       choices=['Small', 'Medium', 'Large'])

    args = parser.parse_args()

    if args.output_path is None:
        frame_name = Path(args.rendered_path).parent.name
        output_dir = Path(args.rendered_path).parent
        args.output_path = str(output_dir / f'{frame_name}_comparison.png')

    visualize_feature_comparison_fast(
        args.rendered_path,
        args.gt_path,
        args.image_path,
        args.output_path,
        args.scale
    )

#!/usr/bin/env python3
"""
特征对比可视化工具
对比渲染特征与 GT 特征，包括：
1. PCA-RGB 可视化
2. 并排对比
3. 差异可视化
4. 量化指标计算
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
    """
    加载 GT 特征并解码为完整的 feature map

    Args:
        gt_path: GT .pth 文件路径
        scale_name: 指定 scale 名称（Small/Medium/Large）

    Returns:
        feature_map: [H, W, C] numpy array
        mask: [H, W] boolean numpy array
    """
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
    """
    加载渲染特征

    Args:
        rendered_path: 渲染特征 .pt 文件路径

    Returns:
        feature_map: [H, W, C] numpy array
    """
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
    """
    计算渲染特征与 GT 特征的量化指标

    Args:
        rendered_feat: [H, W, C] 渲染特征
        gt_feat: [H, W, C] GT 特征
        mask: [H, W] 有效区域掩码

    Returns:
        dict: 包含各种指标的字典
    """
    H, W, C = rendered_feat.shape

    # 展平
    rendered_flat = rendered_feat.reshape(-1, C)
    gt_flat = gt_feat.reshape(-1, C)

    # 如果有 mask，只计算有效区域
    if mask is not None:
        mask_flat = mask.reshape(-1)
        rendered_flat = rendered_flat[mask_flat]
        gt_flat = gt_flat[mask_flat]

    # 过滤 NaN 和 Inf
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


def pca_to_rgb(features: np.ndarray, mask: np.ndarray = None,
               pca_model: PCA = None, feature_mean: np.ndarray = None, feature_std: np.ndarray = None,
               percentile_clip: float = 2.0):
    """
    将高维特征通过 PCA 压缩到 3 维 RGB

    Args:
        features: [H, W, C] 特征图
        mask: [H, W] 有效区域掩码
        pca_model: 预训练的 PCA 模型，None 则新建
        feature_mean: 特征均值（用于标准化）
        feature_std: 特征标准差（用于标准化）
        percentile_clip: 用于归一化的百分位数裁剪

    Returns:
        rgb_image: [H, W, 3] RGB 图像 (0-255 uint8)
        pca_model: 训练好的 PCA 模型
        explained_variance: PCA 解释方差
    """
    H, W, C = features.shape

    # 展平特征
    features_flat = features.reshape(-1, C)  # [H*W, C]

    # 如果提供了均值和标准差，进行标准化
    if feature_mean is not None and feature_std is not None:
        features_flat = (features_flat - feature_mean) / (feature_std + 1e-8)

    # 如果有 mask，只对有效区域进行 PCA
    if mask is not None:
        mask_flat = mask.reshape(-1)
        valid_features = features_flat[mask_flat]
    else:
        valid_features = features_flat
        mask_flat = np.ones(H * W, dtype=bool)

    # 过滤掉全零的特征和无效值
    non_zero_mask = np.isfinite(valid_features).all(axis=1) & (np.linalg.norm(valid_features, axis=1) > 1e-6)
    valid_features = valid_features[non_zero_mask]

    if len(valid_features) < 10:
        print("Warning: Not enough valid features for PCA")
        return np.zeros((H, W, 3), dtype=np.uint8), None, 0.0

    # PCA 降维到 3 维
    if pca_model is None:
        pca_model = PCA(n_components=3, svd_solver='randomized', random_state=42)
        # 采样部分数据进行拟合（如果数据太大）
        max_samples = min(len(valid_features), 20000)
        if len(valid_features) > max_samples:
            sample_idx = np.random.choice(len(valid_features), max_samples, replace=False)
            pca_model.fit(valid_features[sample_idx])
        else:
            pca_model.fit(valid_features)

    explained_variance = pca_model.explained_variance_ratio_.sum()

    # 对所有像素进行变换
    pca_features = pca_model.transform(features_flat)  # [H*W, 3]

    # 归一化到 [0, 1]
    rgb_normalized = np.zeros_like(pca_features)
    for i in range(3):
        channel = pca_features[:, i]
        # 只考虑有效区域的统计量
        valid_channel = channel[mask_flat]
        valid_channel = valid_channel[np.isfinite(valid_channel)]

        if len(valid_channel) > 0:
            low = np.percentile(valid_channel, percentile_clip)
            high = np.percentile(valid_channel, 100 - percentile_clip)

            # 归一化
            if high - low > 1e-6:
                normalized = (channel - low) / (high - low)
            else:
                normalized = np.zeros_like(channel)

            rgb_normalized[:, i] = np.clip(normalized, 0, 1)

    # Reshape 回图像格式
    rgb_image = rgb_normalized.reshape(H, W, 3)

    # 将无效区域设为灰色
    if mask is not None:
        rgb_image[~mask] = 0.5

    # 转换为 uint8
    rgb_image = (rgb_image * 255).astype(np.uint8)

    return rgb_image, pca_model, explained_variance


def visualize_feature_comparison(rendered_path: str, gt_path: str, image_path: str,
                                  output_path: str, scale_name: str = 'Medium'):
    """
    对比可视化渲染特征与 GT 特征

    生成一个综合对比图，包括：
    - 原图
    - GT PCA-RGB
    - 渲染 PCA-RGB
    - 差异图
    - 量化指标

    Args:
        rendered_path: 渲染特征路径
        gt_path: GT 特征路径
        image_path: 原始图像路径
        output_path: 输出路径
        scale_name: GT 特征的 scale（Small/Medium/Large）
    """
    print(f"Loading features...")
    print(f"  Rendered: {rendered_path}")
    print(f"  GT: {gt_path}")
    print(f"  Image: {image_path}")

    # 加载特征
    rendered_feat = load_rendered_feature_map(rendered_path)
    gt_feat, gt_mask = load_gt_feature_map(gt_path, scale_name)

    print(f"Rendered feature shape: {rendered_feat.shape}")
    print(f"GT feature shape: {gt_feat.shape}")
    print(f"GT mask coverage: {gt_mask.mean():.2%}")

    # 确保尺寸一致
    if rendered_feat.shape[:2] != gt_feat.shape[:2]:
        print(f"Warning: Size mismatch. Resizing GT feature to match rendered feature.")
        # Resize GT feature
        gt_feat_tensor = torch.from_numpy(gt_feat).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        gt_mask_tensor = torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        target_size = rendered_feat.shape[:2]
        gt_feat_tensor = F.interpolate(gt_feat_tensor, size=target_size, mode='bilinear', align_corners=False)
        gt_mask_tensor = F.interpolate(gt_mask_tensor, size=target_size, mode='nearest')

        gt_feat = gt_feat_tensor.squeeze(0).permute(1, 2, 0).numpy()
        gt_mask = (gt_mask_tensor.squeeze(0).squeeze(0) > 0.5).numpy()

    # 加载原图
    original_image = Image.open(image_path).convert('RGB')
    # Resize 原图到与特征图一致
    img_resized = original_image.resize((rendered_feat.shape[1], rendered_feat.shape[0]), Image.LANCZOS)
    img_array = np.array(img_resized)

    # 计算量化指标
    print("\nComputing metrics...")
    metrics = compute_metrics(rendered_feat, gt_feat, gt_mask)

    # 使用全局 PCA（在 GT 和渲染特征上一起训练）
    print("\nFitting global PCA...")
    all_features = []

    # 收集 GT 有效特征
    gt_flat = gt_feat.reshape(-1, gt_feat.shape[-1])
    gt_mask_flat = gt_mask.reshape(-1)
    gt_valid = gt_flat[gt_mask_flat]
    # 过滤 NaN 和 Inf
    gt_valid = gt_valid[np.isfinite(gt_valid).all(axis=1)]
    gt_valid = gt_valid[np.linalg.norm(gt_valid, axis=1) > 1e-6]

    print(f"GT valid features: {len(gt_valid)}, has NaN: {np.isnan(gt_valid).any()}, has Inf: {np.isinf(gt_valid).any()}")

    # 收集渲染特征
    rendered_flat = rendered_feat.reshape(-1, rendered_feat.shape[-1])
    # 过滤 NaN 和 Inf
    rendered_flat_filtered = rendered_flat[np.isfinite(rendered_flat).all(axis=1)]
    rendered_valid = rendered_flat_filtered[np.linalg.norm(rendered_flat_filtered, axis=1) > 1e-6]

    print(f"Rendered valid features: {len(rendered_valid)}, has NaN: {np.isnan(rendered_valid).any()}, has Inf: {np.isinf(rendered_valid).any()}")

    # 采样以控制内存（更激进的采样）
    max_per_source = 10000  # 每个来源最多1万个样本
    if len(gt_valid) > max_per_source:
        sample_idx = np.random.choice(len(gt_valid), max_per_source, replace=False)
        gt_valid = gt_valid[sample_idx]
        print(f"GT sampled: {len(gt_valid)}, has NaN: {np.isnan(gt_valid).any()}, has Inf: {np.isinf(gt_valid).any()}")

    if len(rendered_valid) > max_per_source:
        sample_idx = np.random.choice(len(rendered_valid), max_per_source, replace=False)
        rendered_valid = rendered_valid[sample_idx]
        print(f"Rendered sampled: {len(rendered_valid)}, has NaN: {np.isnan(rendered_valid).any()}, has Inf: {np.isinf(rendered_valid).any()}")

    all_features = np.concatenate([gt_valid, rendered_valid], axis=0)
    print(f"After concat: {len(all_features)}, has NaN: {np.isnan(all_features).any()}, has Inf: {np.isinf(all_features).any()}")

    # 再次检查并过滤 NaN 和 Inf
    all_features = all_features[np.isfinite(all_features).all(axis=1)]
    print(f"After filter: {len(all_features)}, has NaN: {np.isnan(all_features).any()}, has Inf: {np.isinf(all_features).any()}")

    print(f"Total features for global PCA: {len(all_features)}")

    if len(all_features) < 10:
        print("Error: Not enough valid features for PCA")
        return None

    # 标准化特征以提高数值稳定性
    print("Standardizing features...")
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0) + 1e-8  # 避免除零
    all_features_normalized = (all_features - mean) / std

    print(f"After normalization: has NaN: {np.isnan(all_features_normalized).any()}, has Inf: {np.isinf(all_features_normalized).any()}")

    # 使用更快的 randomized PCA，并且只用少量样本
    print("Fitting PCA (using randomized solver for speed)...")
    global_pca = PCA(n_components=3, svd_solver='randomized', random_state=42)

    # 大幅减少样本数量以加快速度
    max_samples = min(len(all_features_normalized), 20000)  # 最多2万个样本
    if len(all_features_normalized) > max_samples:
        sample_idx = np.random.choice(len(all_features_normalized), max_samples, replace=False)
        global_pca.fit(all_features_normalized[sample_idx])
        print(f"PCA fitted on {max_samples} samples (sampled from {len(all_features_normalized)})")
    else:
        global_pca.fit(all_features_normalized)
        print(f"PCA fitted on all {len(all_features_normalized)} samples")

    print(f"Global PCA explained variance: {global_pca.explained_variance_ratio_}")
    print(f"Total explained: {global_pca.explained_variance_ratio_.sum():.3f}")

    # 对 GT 和渲染特征进行 PCA 降维
    print("\nGenerating PCA-RGB visualizations...")
    gt_rgb, _, gt_variance = pca_to_rgb(gt_feat, gt_mask, pca_model=global_pca, feature_mean=mean, feature_std=std)
    rendered_rgb, _, rendered_variance = pca_to_rgb(rendered_feat, None, pca_model=global_pca, feature_mean=mean, feature_std=std)

    # 计算差异图
    print("\nComputing difference map...")
    # 对于差异图，我们计算特征空间的 L2 距离
    diff_map = np.sqrt(((rendered_feat - gt_feat) ** 2).sum(axis=2))  # [H, W]

    # 归一化差异图
    diff_map_vis = diff_map.copy()
    if gt_mask is not None:
        valid_diff = diff_map[gt_mask]
        if len(valid_diff) > 0:
            vmin, vmax = np.percentile(valid_diff, [2, 98])
            diff_map_vis = np.clip((diff_map - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # 创建可视化
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)

    # 第一行
    # 1. 原图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_array)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 2. GT PCA-RGB
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gt_rgb)
    ax2.set_title(f'GT Features (PCA-RGB)\nScale: {scale_name}, Coverage: {gt_mask.mean():.1%}',
                  fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 3. 渲染 PCA-RGB
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(rendered_rgb)
    ax3.set_title(f'Rendered Features (PCA-RGB)\nCoverage: 100%',
                  fontsize=14, fontweight='bold')
    ax3.axis('off')

    # 第二行
    # 4. GT + 原图叠加
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

    # 5. 渲染 + 原图叠加
    ax5 = fig.add_subplot(gs[1, 1])
    overlay_rendered = img_array.astype(float) / 255.0
    feature_overlay_rendered = rendered_rgb.astype(float) / 255.0
    blended_rendered = overlay_rendered * 0.4 + feature_overlay_rendered * 0.6
    ax5.imshow(blended_rendered)
    ax5.set_title('Rendered Features Overlay', fontsize=14, fontweight='bold')
    ax5.axis('off')

    # 6. 差异图 + 量化指标
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
PCA Explained Variance: {global_pca.explained_variance_ratio_.sum():.3f}
"""

    fig.text(0.5, 0.02, metrics_text, ha='center', va='bottom',
             fontsize=12, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 添加总标题
    frame_name = Path(rendered_path).parent.name
    fig.suptitle(f'Feature Comparison: {frame_name}', fontsize=16, fontweight='bold', y=0.98)

    # 保存
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✅ Visualization saved to: {output_path}")
    print(f"\nMetrics summary:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Feature Comparison Visualization')
    parser.add_argument('--rendered_path', type=str, required=True,
                       help='Path to rendered feature .pt file')
    parser.add_argument('--gt_path', type=str, required=True,
                       help='Path to GT feature .pth file')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to original image')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path for visualization')
    parser.add_argument('--scale', type=str, default='Medium',
                       choices=['Small', 'Medium', 'Large'],
                       help='GT feature scale to use')

    args = parser.parse_args()

    # 默认输出路径
    if args.output_path is None:
        frame_name = Path(args.rendered_path).parent.name
        output_dir = Path(args.rendered_path).parent
        args.output_path = str(output_dir / f'{frame_name}_comparison.png')

    visualize_feature_comparison(
        args.rendered_path,
        args.gt_path,
        args.image_path,
        args.output_path,
        args.scale
    )

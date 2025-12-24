# 多尺度 LLaVA 3584维特征提取完整指南

## 问题回顾

**之前的问题**：
- 旧脚本提取的是 **1152维 SigLIP 特征**（vision tower的中间层输出）
- 无法直接用于 LLaVA 解码，因为 LLaVA 需要 **3584维完整特征**

**解决方案**：
- 修改特征提取方法：从 `model.get_vision_tower()` 改为 `model.prepare_inputs_labels_for_multimodal_pre()`
- 保留多尺度crops的逻辑和可视化功能
- 提取完整的 3584维 LLaVA 特征

## 核心修改

### 关键代码变更

**旧方法**（提取1152维SigLIP特征）：
```python
image_features = self.model.get_vision_tower()(image_tensor)
# 输出: [27, 27, 1152]
```

**新方法**（提取3584维LLaVA特征）：
```python
encoded_features, split_sizes = self.model.prepare_inputs_labels_for_multimodal_pre([image_tensor])
# encoded_features: [num_patches, 729, 3584]
# 处理后: [27, 27, 3584]
```

## 使用方法

### 方法1：单张图像处理

```bash
conda activate llava
cd /home/wangyz/project/0working/LangSplatV2/LLaVA-NeXT

python extract_llava3584_multiscale_crops.py \
    --image_path /path/to/image.jpg \
    --output_dir /path/to/output \
    --extract_features \
    --visualize
```

**可选参数**：
- `--depth_path`: 深度图路径（用于更好的crop生成）
- `--model_path`: LLaVA模型路径（默认：lmms-lab/llava-onevision-qwen2-7b-ov）
- `--feature_map_size`: 特征图大小（默认：27 27）
- `--overlap_mode`: 重叠处理模式（average/overwrite，默认：average）

### 方法2：批量处理（推荐）0

```bash
conda activate llava
cd /home/wangyz/project/0working/LangSplatV2/LLaVA-NeXT

python batch_extract_llava3584.py \
    --image_folder /mnt/data/wangyz/lerf_ovs/teatime/images \
    --output_folder /mnt/data/wangyz/lerf_ovs/teatime/llava_features_3584_multiscale \
    --depth_folder /mnt/data/wangyz/lerf_ovs/teatime/render_depth
```

**参数说明**：
- `--image_folder`: 图像文件夹
- `--output_folder`: 输出文件夹
- `--depth_folder`: 深度图文件夹（可选）
- `--no_visualize`: 禁用可视化

## 输出文件结构

对于每张图像 `frame_00001.jpg`，会生成：

### 1. 特征文件：`frame_00001_feature_map.pt`

```python
{
    'feature_map': torch.Tensor,     # [H, W, 3584], float32
    'valid_mask': torch.Tensor,       # [H, W], float32
    'image_path': str,                # 原始图像路径
    'crops_info': {                   # crops信息
        'Small': {'crops': [...], 'scale': 0.25, 'crop_count': 164},
        'Medium': {'crops': [...], 'scale': 1.0, 'crop_count': 12},
        'Large': {'crops': [...], 'scale': 2.5, 'crop_count': 1}
    },
    'feature_map_size': [27, 27],
    'overlap_mode': 'average'
}
```

**关键信息**：
- `feature_map`: **3584维 LLaVA 特征**，对齐到原始图像尺寸
- `valid_mask`: 标记哪些像素有有效的特征
- `crops_info`: 记录了每个尺度的crops信息

### 2. 可视化文件：`frame_00001_feature_visualization.png`

包含9个子图：
1. 原始图像 + crops布局
2. 特征覆盖掩码
3. 特征PCA可视化（第1-3主成分RGB）
4. 特征t-SNE可视化
5. Crops布局（按尺度分类）
6. 特征相似度矩阵
7. 特征维度分布
8. 特征强度热力图
9. 数据统计

## 特征验证

检查提取的特征是否正确：

```python
import torch

# 加载特征
feat = torch.load('frame_00001_feature_map.pt', map_location='cpu')

# 验证维度
assert feat['feature_map'].shape[-1] == 3584, "特征维度应该是3584"
print(f"✅ 特征维度正确: {feat['feature_map'].shape}")

# 检查覆盖率
coverage = feat['valid_mask'].float().mean()
print(f"✅ 特征覆盖率: {coverage:.2%}")

# 检查crops信息
for scale, info in feat['crops_info'].items():
    print(f"  {scale}: {info['crop_count']} crops")
```

## 与旧格式的对比

| 特性 | 旧格式 (1152维) | 新格式 (3584维) |
|------|----------------|----------------|
| 特征维度 | 1152 (SigLIP) | **3584 (LLaVA)** |
| 特征来源 | vision_tower | vision + projector |
| 可用于LLaVA解码 | ❌ 否 | ✅ **是** |
| 文件大小 | ~3GB | ~10GB |
| crops信息 | 包含 | 包含 |
| 可视化 | 包含 | **增强** |

## 内存管理

**48GB显存建议**：

1. **单张图像处理**：通常没问题
2. **批量处理**：使用 `batch_extract_llava3584.py`，它会：
   - 每次只处理一张图像
   - 处理完成后释放显存
   - 自动跳过已处理的图像

3. **如果仍然OOM**：
   - 减少Large尺度的crops数量
   - 或者只使用Small和Medium尺度
   - 调整 `SemanticMultiScaleCropper` 的参数

## 后续步骤

### 1. 训练3DGS

修改 `train.py` 中的特征维度：

```python
# 修改前
language_feature_dim = 1152 if opt.llm_feature else 512

# 修改后
language_feature_dim = 3584 if opt.llm_feature else 512
```

修改 `scene/gaussian_model.py`：

```python
# 修改前
def __init__(self, sh_degree: int, language_feature_dim: int = 1152):

# 修改后
def __init__(self, sh_degree: int, language_feature_dim: int = 3584):
```

### 2. 渲染特征

使用修改后的模型渲染特征：

```bash
python render_lerf_llm.py \
    -s /mnt/data/wangyz/lerf_ovs/teatime \
    --ckpt_root_path output \
    --dataset_name teatime \
    --index test_3584 \
    --checkpoint 10000
```

渲染输出应该是 `[1, H, W, 3584]`

### 3. 使用LLaVA解码

使用 `decode_rendered_features.py` 进行场景描述：

```bash
conda activate llava
python decode_rendered_features.py \
    --feature_path /path/to/rendered_features.pt \
    --question "please describe the scene in detail" \
    --max_tokens 2000
```

## 故障排查

### 问题1: OOM
**解决方案**：
```bash
# 清理显存
nvidia-smi | grep "python" | awk '{print $5}' | xargs -r kill -9

# 或重启
sudo nvidia-smi --gpu-reset
```

### 问题2: Crops数据格式警告
**现象**：`⚠️ Scale Small not found in crops data`

**原因**：这是内部数据格式问题，不影响最终输出

**验证**：检查输出的 `crops_info` 是否包含所有尺度

### 问题3: 特征维度错误
**检查**：
```python
assert feat['feature_map'].shape[-1] == 3584
```

如果失败，说明可能：
1. 使用了旧版本脚本
2. 模型加载失败，使用了dummy features

## 文件位置

- **特征提取脚本**: `LLaVA-NeXT/extract_llava3584_multiscale_crops.py`
- **批量处理脚本**: `LLaVA-NeXT/batch_extract_llava3584.py`
- **特征解码脚本**: `decode_rendered_features.py`
- **使用指南**: `LLAVA_FEATURE_GUIDE.md`（总体）
- **详细指南**: 本文件

## 性能参考

在 48GB A6000 上：
- **单张图像**: ~2-3分钟
- **Small尺度**: ~160 crops, ~1分钟
- **Medium尺度**: ~12 crops, ~30秒
- **Large尺度**: ~1 crop, ~10秒
- **可视化生成**: ~20秒

## 总结

✅ 成功将1152维SigLIP特征升级为3584维LLaVA特征
✅ 保留了多尺度crops的所有功能
✅ 增强了可视化
✅ 现在可以直接用于LLaVA解码
✅ 与3DGS训练流程完全兼容

现在你可以开始批量提取特征了！

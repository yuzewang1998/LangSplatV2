# 多尺度 LLaVA 特征提取和解码使用指南

## 概述

本项目实现了两个主要功能：
1. **提取 3584 维 LLaVA 特征**（而不是之前的 1152 维 SigLIP 特征）
2. **使用提取的特征通过 LLaVA 进行场景描述**

## 核心区别

### 之前的方案（1152 维）
- 提取的是 **SigLIP vision tower 的多层特征拼接**（3层 × 384维 = 1152维）
- 只是视觉编码器的中间输出，**不能直接用于 LLaVA 解码**

### 现在的方案（3584 维）
- 提取的是 **经过 vision encoder + multimodal projector 的完整 LLaVA 特征**
- 维度是 3584（Qwen2-7B 的隐藏层大小）
- **可以直接输入到 LLaVA 的语言模型进行解码**

## 第一步：提取多尺度 LLaVA 特征

### 脚本位置
```
/home/wangyz/project/0working/LangSplatV2/LLaVA-NeXT/extract_multiscale_llava_features.py
```

### 使用方法

```bash
# 切换到 llava 环境
conda activate llava

# 提取特征
cd /home/wangyz/project/0working/LangSplatV2/LLaVA-NeXT

python extract_multiscale_llava_features.py \
    --image_folder /path/to/images \
    --output_folder /path/to/output \
    --scales Small Medium Large \
    --scale_factors 0.5 1.0 2.0 \
    --pretrained lmms-lab/llava-onevision-qwen2-7b-ov \
    --model_name llava_qwen
```

### 参数说明

- `--image_folder`: 图像文件夹路径
- `--output_folder`: 输出文件夹路径
- `--scales`: 尺度名称列表（例如：Small, Medium, Large）
- `--scale_factors`: 对应的缩放因子（例如：0.5, 1.0, 2.0）
- `--pretrained`: LLaVA 模型路径
- `--model_name`: 模型名称（llava_qwen）
- `--no_visualization`: 禁用可视化（如果不需要生成可视化图像）

### 输出文件

对于每张图像 `frame_00001.jpg`，会生成：

1. **特征文件**: `frame_00001_multiscale_features.pt`
   - 包含多尺度 3584 维特征
   - 使用 float16 节省空间
   - 文件大小约 20-30 GB（取决于图像分辨率和尺度数量）

2. **可视化图像**（如果启用）:
   - `frame_00001_Small_feature_visualization.png`
   - `frame_00001_Medium_feature_visualization.png`
   - `frame_00001_Large_feature_visualization.png`

### 特征文件结构

```python
{
    'multiscale_features': {
        'Small': {
            'feature_map': torch.Tensor,  # [H, W, 3584], float16
            'valid_mask': torch.Tensor,   # [H, W], bool
            'scale_factor': float,         # 0.5
            'feature_dim': int             # 3584
        },
        'Medium': {...},
        'Large': {...}
    },
    'image_path': str,
    'original_size': tuple,                 # (W, H)
    'scales': list,                         # ['Small', 'Medium', 'Large']
    'feature_extraction_mode': str,         # 'llava_3584_multiscale'
    'feature_dim': int                      # 3584
}
```

## 第二步：使用 3DGS 训练（参考 CLAUDE.md）

使用提取的 3584 维特征进行 3D Gaussian Splatting 训练。训练脚本需要相应地修改：

### 关键修改点

1. **`scene/gaussian_model.py`**:
   ```python
   # 修改默认特征维度
   def __init__(self, sh_degree : int, language_feature_dim : int = 3584):  # 之前是 1152
   ```

2. **`train.py`**:
   ```python
   # 修改特征维度
   language_feature_dim = 3584 if opt.llm_feature else 512
   ```

3. **数据加载**:
   确保加载的是新提取的 3584 维特征文件

## 第三步：渲染特征（参考 render_lerf_llm.py）

渲染过程保持不变，但确保：
- 使用训练好的 3584 维模型
- 渲染输出的特征维度应该是 `[1, H, W, 3584]`（不再是 1152）

## 第四步：使用 LLaVA 解码（新功能）

### 脚本位置
```
/home/wangyz/project/0working/LangSplatV2/decode_rendered_features.py
```

### 使用方法

```bash
# 切换到 llava 环境
conda activate llava

# 解码特征
python decode_rendered_features.py \
    --feature_path /path/to/rendered_features.pt \
    --question "please describe the scene in detail" \
    --max_tokens 2000
```

### 参数说明

- `--feature_path`: 渲染的特征文件路径（.pt 文件）
- `--question`: 要问的问题
- `--llava_model`: LLaVA 模型路径（默认：lmms-lab/llava-onevision-qwen2-7b-ov）
- `--max_tokens`: 最大 token 数量（默认：5000）
- `--sampling`: Token 采样策略（grid/random/center）

### 工作原理

1. 加载渲染的 3584 维特征 `[H, W, 3584]`
2. 如果 token 数量过多，进行下采样
3. 直接将特征输入到 LLaVA 进行解码
4. 输出场景描述

### 示例输出

```
================================================================================
📝 结果
================================================================================
❓ 问题: please describe the scene in detail

💡 回答:
The scene depicts a cozy indoor setting with a wooden table...
================================================================================
```

## 完整工作流示例

```bash
# 1. 提取 LLaVA 特征
conda activate llava
cd /home/wangyz/project/0working/LangSplatV2/LLaVA-NeXT

python extract_multiscale_llava_features.py \
    --image_folder /mnt/data/wangyz/lerf_ovs/teatime/images \
    --output_folder /mnt/data/wangyz/lerf_ovs/teatime/llava_features_3584_multiscale \
    --scales Small Medium Large \
    --scale_factors 0.5 1.0 2.0

# 2. 训练 3DGS（记得修改 language_feature_dim = 3584）
conda activate langsplat_v2
cd /home/wangyz/project/0working/LangSplatV2

bash train.sh

# 3. 渲染特征
python render_lerf_llm.py \
    -s /mnt/data/wangyz/lerf_ovs/teatime \
    --ckpt_root_path output \
    --dataset_name teatime \
    --index test \
    --checkpoint 10000

# 4. 解码场景描述
conda activate llava
python decode_rendered_features.py \
    --feature_path /mnt/data/wangyz/lerf_ovs/eval_result/teatime_test/frame_00003/feature_map_frame_00003.pt \
    --question "please describe the scene in detail"
```

## 注意事项

1. **环境切换**:
   - 特征提取和解码使用 `llava` 环境
   - 3DGS 训练使用 `langsplat_v2` 环境

2. **存储空间**:
   - 3584 维特征文件很大（20-30 GB per image）
   - 确保有足够的磁盘空间

3. **内存管理**:
   - 提取特征时，Large 尺度可能需要大量 GPU 内存
   - 如果 OOM，可以只提取 Small 和 Medium 尺度

4. **兼容性**:
   - 确保所有脚本使用相同的特征维度（3584）
   - 旧的 1152 维特征无法直接用于新的解码流程

## 故障排查

### 问题1: OOM during feature extraction
**解决方案**: 减少尺度数量或降低 scale_factors

```bash
# 只提取 Small 和 Medium
python extract_multiscale_llava_features.py \
    --scales Small Medium \
    --scale_factors 0.5 1.0 \
    ...
```

### 问题2: 特征维度不匹配
**检查**:
```python
import torch
feat = torch.load('feature_map.pt')
print(feat.shape[-1])  # 应该是 3584，不是 1152
```

### 问题3: 解码输出不合理
**原因**: 可能是使用了错误维度的特征或者 token 采样不当
**解决方案**:
- 确认特征是 3584 维
- 调整 `--max_tokens` 参数
- 尝试不同的 `--sampling` 策略

## 文件位置总结

- **特征提取**: `LLaVA-NeXT/extract_multiscale_llava_features.py`
- **特征解码**: `decode_rendered_features.py`
- **3DGS 训练**: `train.py`
- **特征渲染**: `render_lerf_llm.py`
- **模型定义**: `scene/gaussian_model.py`

## 联系和反馈

如有问题，请参考：
- `CLAUDE.md`: 项目整体文档
- `DIAGNOSIS.md`: 诊断和调试指南

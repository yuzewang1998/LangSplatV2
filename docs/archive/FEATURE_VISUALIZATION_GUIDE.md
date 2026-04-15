# 特征可视化对比功能使用指南

## 功能概述

本功能为 `render_lerf_llm.py` 添加了完整的**特征对比可视化**能力，可以将渲染的语言特征与Ground Truth (GT)特征进行全面对比分析。

## 核心功能

### 1. PCA-RGB 可视化
- 将3584维的高维语言特征通过PCA降维到3维RGB空间进行可视化
- 使用全局PCA确保渲染特征和GT特征在同一颜色空间中，便于直接对比
- 支持特征标准化以提高数值稳定性

### 2. 量化指标计算
自动计算以下指标来评估渲染质量：
- **余弦相似度 (Cosine Similarity)**: 衡量特征方向的一致性
- **L1 距离**: 平均绝对差异
- **L2 距离**: 均方根误差
- **有效像素覆盖率**: GT mask覆盖的像素比例

### 3. 综合对比可视化
生成一张包含6个子图的对比图：
1. **原始RGB图像**
2. **GT特征的PCA-RGB可视化**
3. **渲染特征的PCA-RGB可视化**
4. **GT特征叠加到原图**
5. **渲染特征叠加到原图**
6. **特征差异热力图** (L2距离, 颜色越热表示差异越大)

底部显示详细的量化指标。

## 使用方法

### 方法1：使用脚本（推荐）

直接运行 `eval_lerf_llm.sh`，脚本已经配置好所有参数：

```bash
bash eval_lerf_llm.sh
```

脚本会自动：
- 渲染测试集的语言特征
- 加载对应的GT特征
- 生成对比可视化图

### 方法2：手动指定参数

```bash
python render_lerf_llm.py \
    -s /path/to/dataset \
    -m /path/to/model \
    --dataset_name teatime \
    --index test \
    --ckpt_root_path ./output \
    --output_dir ./eval_result \
    --json_folder /path/to/labels \
    --checkpoint 10000 \
    --include_feature \
    --topk 4 \
    -r 2 \
    --visualize_comparison \
    --gt_feature_dir /path/to/gt_features \
    --comparison_scale Medium
```

### 方法3：单独可视化某一帧

如果你已经有渲染好的特征，可以单独运行可视化脚本：

```bash
python visualize_feature_comparison.py \
    --rendered_path eval_result/teatime_test/frame_00003/feature_map_frame_00003.pt \
    --gt_path /path/to/gt/frame_00003.pth \
    --image_path /path/to/images/frame_00003.jpg \
    --output_path output_comparison.png \
    --scale Medium
```

## 命令行参数说明

### render_lerf_llm.py 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--visualize_comparison` | flag | False | 启用特征对比可视化 |
| `--gt_feature_dir` | str | None | GT特征文件目录（包含`.pth`文件）|
| `--comparison_scale` | str | "Medium" | GT特征的scale，可选：Small/Medium/Large |

### visualize_feature_comparison.py 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--rendered_path` | str | 是 | 渲染特征文件路径（.pt） |
| `--gt_path` | str | 是 | GT特征文件路径（.pth） |
| `--image_path` | str | 是 | 原始RGB图像路径 |
| `--output_path` | str | 否 | 输出PNG路径（默认自动生成） |
| `--scale` | str | 否 | GT特征scale（默认Medium） |

## 输出文件

对于每个测试帧，会生成以下文件：

```
eval_result/
└── teatime_test/
    └── frame_00003/
        ├── feature_map_frame_00003.pt          # 渲染的语言特征
        └── frame_00003_comparison.png          # 对比可视化图
```

## 性能说明

### 计算时间
- **渲染特征**: 取决于场景复杂度，通常几秒到几十秒
- **PCA可视化**: 对于3584维特征，PCA计算可能需要**1-5分钟**每帧
  - 这是因为需要对大量高维数据进行SVD分解
  - 首次计算最慢，后续帧会复用部分计算

### 优化建议
如果PCA计算太慢，可以在 `visualize_feature_comparison.py` 中调整：

1. **减少采样数量**（第350行附近）：
   ```python
   if len(all_features_normalized) > 50000:  # 原来是100000
       sample_idx = np.random.choice(len(all_features_normalized), 50000, replace=False)
   ```

2. **使用randomized PCA**（第349行）：
   ```python
   global_pca = PCA(n_components=3, svd_solver='randomized')  # 原来是'full'
   ```

## 技术细节

### GT特征格式
GT特征使用LLaVA-NeXT提取，以crop-level格式存储：
- 每个scale（Small/Medium/Large）包含多个crop
- 每个crop是27×27×3584的特征张量
- 使用 `CropFeatureCodec.decode_to_full_map()` 解码为完整特征图

### 渲染特征格式
渲染特征是直接从3D Gaussian Splatting渲染得到：
- 形状：`[1, H, W, 3584]`
- 每个像素有3584维的语言特征向量

### PCA流程
1. **特征收集**: 从GT和渲染特征中收集有效像素
2. **标准化**: 对特征进行z-score标准化以提高数值稳定性
3. **全局PCA**: 在混合的特征集上训练PCA模型
4. **投影**: 将GT和渲染特征分别投影到PCA空间
5. **归一化**: 将3维PCA特征归一化到[0,255]作为RGB值

### 数值稳定性
代码包含多重保护措施：
- 过滤NaN和Inf值
- 特征标准化
- 使用full SVD solver确保精度
- Percentile clipping避免极端值影响

## 故障排除

### 问题1: ModuleNotFoundError: No module named 'crop_feature_codec'
**解决**: 确保LLaVA-NeXT子模块已正确安装：
```bash
cd LLaVA-NeXT
# 检查 crop_feature_codec.py 是否存在
```

### 问题2: GT特征文件找不到
**解决**: 检查GT特征目录路径和文件命名：
```bash
ls /path/to/gt_features/
# 应该看到 frame_00001.pth, frame_00002.pth 等文件
```

### 问题3: PCA计算太慢
**解决**: 按照上面"优化建议"部分调整采样率或使用randomized PCA

### 问题4: 内存不足 (OOM)
**解决**:
- 减少PCA采样数量
- 使用更小的特征维度
- 分批处理测试帧

## 示例输出解读

### 量化指标典型值
- **Cosine Similarity**: 0.7-0.9 表示较好，>0.9 表示优秀
- **L1 Distance**: 越小越好，通常在0.5-2.0范围
- **L2 Distance**: 越小越好，通常在1.0-5.0范围

### 可视化解读
- **GT vs 渲染特征**: 颜色相似表示特征相似
- **差异热力图**: 蓝色（冷色）表示差异小，红色（热色）表示差异大
- **叠加图**: 有助于理解特征在场景中的空间分布

## 扩展开发

如果需要添加新的可视化或指标，可以修改 `visualize_feature_comparison.py`：

- **添加新指标**: 在 `compute_metrics()` 函数中添加
- **自定义可视化**: 修改 `visualize_feature_comparison()` 中的matplotlib代码
- **改变布局**: 调整 `fig.add_gridspec()` 的参数

## 参考文档

- LLAVA特征提取: `LLAVA_FEATURE_GUIDE.md`
- 项目架构: `CLAUDE.md`
- 主训练文档: `README.md`

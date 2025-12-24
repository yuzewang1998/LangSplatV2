# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

LangSplatV2 是一个用于高维 3D 语言高斯喷溅（3D Language Gaussian Splatting）的研究项目，实现了 450+ FPS 的性能。该项目基于 3D Gaussian Splatting，通过向量量化（Vector Quantization）技术将高维语言特征嵌入到 3D 场景中。

## 核心架构

### 主要组件

1. **场景表示 (scene/)**
   - `GaussianModel`: 核心 3D 高斯模型，包含几何属性（位置、旋转、缩放、不透明度）和语言特征
   - `Scene`: 场景管理，负责加载数据集和相机参数
   - 支持两种模式：原始 RGB 高斯喷溅（12 参数）和语言特征高斯喷溅（14 参数）

2. **语言特征量化 (utils/vq_utils.py)**
   - `ResidualVectorQuantizationWithClustering`: 残差向量量化类，使用 MiniBatchKMeans 进行聚类
   - `LMMFeatureStream`: 流式特征加载器，支持大规模特征数据的内存高效处理
   - 支持多尺度特征（Small/Medium/Large）和混合模式
   - 为避免 OOM，量化过程采用分批处理策略，每次只加载部分数据到 GPU

3. **渲染器 (gaussian_renderer/)**
   - 自定义 CUDA 光栅化器（基于 efficient-langsplat-rasterization 子模块）
   - 渲染 RGB 图像和语言特征权重图（language_feature_weight_map）

4. **评估模块 (eval/)**
   - `OpenCLIPNetwork`: CLIP 编码器用于评估语言特征质量
   - LERF 数据集评估工具（分割 IoU 和定位准确率）

### 数据流

**训练流程：**
1. 预处理：`preprocess.py` 使用 SAM + OpenCLIP 提取 2D 图像的语言特征
2. 训练分为三个阶段（对应三个特征层级 1/2/3）：
   - Stage 1: 初始化全局语义码本（Global Semantic Codebook）
   - Stage 2-3: 逐层训练稀疏系数场（Sparse Coefficient Field）
3. 每个高斯点学习：
   - `_language_feature_logits`: logits 用于计算稀疏系数（通过 top-k softmax）
   - `_language_feature_codebooks`: 共享的码本 [vq_layer_num, codebook_size, feature_dim]

**推理流程：**
1. 从 logits 计算稀疏系数（top-k soft codes）
2. 通过码本重建高维特征：`feature = codebook.T @ soft_code`
3. 渲染语言特征图用于下游任务（开放词汇 3D 分割、定位）

## 常用命令

### 环境设置
```bash
conda env create --file environment.yml
conda activate langsplat_v2
```

### 数据预处理
```bash
python preprocess.py --dataset_path <数据集路径> \
    [--resolution <分辨率>] \
    [--sam_ckpt_path <SAM 检查点路径>]
```

### 训练
```bash
# 使用脚本训练（会自动训练三个层级）
bash train.sh

# 或手动训练单个层级
python train.py \
    -s <数据集路径> \
    -m <输出路径> \
    --start_checkpoint <RGB 高斯喷溅检查点> \
    --feature_level <层级：1/2/3> \
    --vq_layer_num <VQ 层数，默认 1> \
    --codebook_size <码本大小，默认 64> \
    --topk <top-k 稀疏系数数量> \
    [--cos_loss] \
    [--l1_loss] \
    [--normalize] \
    [--llm_feature]  # 使用 LLM 特征而非 CLIP 特征
```

**关键训练参数：**
- `--include_feature`: 训练语言特征（默认 True）
- `--llm_feature`: 使用多模态 LLM 特征（llava_features_multiscale）
- `--topk`: 稀疏系数的 top-k 值（影响内存和性能）
- `--cos_loss` / `--l1_loss`: 损失函数类型
- `--accum_iter`: 梯度累积步数（用于大批量训练）

### 评估
```bash
# LERF 数据集评估
bash eval_lerf.sh <场景名> <模型索引> <检查点>

# 或直接调用
python eval_lerf.py \
    -s <数据集路径> \
    --dataset_name <场景名> \
    --index <模型索引> \
    --ckpt_root_path <检查点根目录> \
    --output_dir <输出目录> \
    --json_folder <标注文件夹> \
    --checkpoint <检查点迭代数> \
    --mask_thresh <掩码阈值> \
    --topk <top-k 值> \
    [--quick_render]  # 快速渲染模式
    [--include_feature]
```

## 数据集格式

项目需要以下数据集结构：
```
<dataset_name>/
├── images/                    # 原始图像
├── sparse/0/                  # COLMAP 稀疏重建结果
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
├── output/<dataset_name>/     # 预训练的 RGB 高斯喷溅模型
│   ├── point_cloud/iteration_30000/point_cloud.ply
│   ├── cameras.json
│   ├── cfg_args
│   └── chkpnt30000.pth
└── language_features/         # (预处理后生成) 语言特征
    或 llava_features_multiscale/
```

## 重要实现细节

### 内存优化
1. **流式特征加载**: `LMMFeatureStream` 逐块加载特征，避免一次性加载所有数据到内存
2. **分批量化**: `_quantize_with_centers()` 使用动态批量大小，根据 GPU 内存调整
3. **稀疏表示**: Top-k 稀疏系数减少存储和计算开销

### 码本初始化
- 单层 VQ (`num_levels == 1`): 使用 `MiniBatchKMeans` 流式拟合，适合大规模数据
- 多层 VQ: 逐层量化残差（当前代码主要使用单层）

### 模型检查点格式
- **不含语言特征** (12 参数): `(sh_degree, xyz, features_dc, features_rest, scaling, rotation, opacity, max_radii2D, xyz_gradient_accum, denom, optimizer_state, spatial_lr_scale)`
- **含语言特征** (14 参数): 额外包含 `language_feature_logits` 和 `language_feature_codebooks`

### 损失函数
- 语言特征训练: 余弦损失 (`cos_loss`) 或 L1 损失，比较渲染的特征图与 GT 特征
- RGB 训练: L1 + DSSIM 损失

## 子模块

项目依赖三个子模块（需要 `--recursive` 克隆）：
1. `submodules/segment-anything-langsplat`: 用于 SAM 分割
2. `submodules/efficient-langsplat-rasterization`: 自定义 CUDA 光栅化器
3. `submodules/simple-knn`: KNN 工具

## 调试技巧

1. **检查码本是否正确加载**: 在训练初期打印 `gaussians._language_feature_codebooks.shape`
2. **验证特征流**: 运行 `python utils/vq_utils.py` 测试流式特征加载
3. **可视化特征图**: 使用 `eval/colormaps.py` 中的工具
4. **监控 GPU 内存**: 量化过程会打印内存使用信息，注意 OOM 警告

## 已知问题

1. 训练在 10000 次迭代后会自动返回（[train.py:208](train.py#L208)），这是为了快速测试
2. 预处理需要 SAM 检查点（`ckpts/sam_vit_h_4b8939.pth`），确保下载
3. Hugging Face 可能需要配置镜像源以解决网络问题

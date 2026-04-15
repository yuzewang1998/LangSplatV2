# PTH 文件结构详解

训练后保存的 `.pth` 文件包含模型的完整状态，用于恢复训练或进行推理。

## 文件保存位置

```python
# train.py 第 219 行
torch.save((gaussians.capture(opt.include_feature), iteration),
           scene.model_path + "/chkpnt" + str(iteration) + ".pth")
```

例如：`output/teatime/chkpnt30000.pth`

## 文件结构

PTH 文件是一个 **tuple**，包含两个元素：
```python
(model_params, iteration)
```

### 顶层结构

| 索引 | 内容 | 类型 | 说明 |
|------|------|------|------|
| [0] | `model_params` | tuple | 模型参数（12 或 14 个元素） |
| [1] | `iteration` | int | 训练迭代次数 |

---

## `model_params` 详解

根据是否包含语言特征，`model_params` 有两种格式：

### 格式 1: 不含语言特征（12 个参数）

**标准 3D Gaussian Splatting 模型**

| 索引 | 参数名 | 形状 | 数据类型 | 含义 |
|------|--------|------|----------|------|
| [0] | `active_sh_degree` | 标量 | int | 当前激活的球谐函数阶数（0-3） |
| [1] | `_xyz` | [N, 3] | float32 | **高斯中心位置**（3D 坐标） |
| [2] | `_features_dc` | [N, 1, 3] | float32 | **颜色特征 DC 分量**（0 阶球谐系数，RGB） |
| [3] | `_features_rest` | [N, 15, 3] | float32 | **颜色特征其余分量**（1-3 阶球谐系数） |
| [4] | `_scaling` | [N, 3] | float32 | **高斯尺度**（3 个轴的缩放因子） |
| [5] | `_rotation` | [N, 4] | float32 | **高斯旋转**（四元数表示） |
| [6] | `_opacity` | [N, 1] | float32 | **不透明度**（经过 sigmoid 激活前的 logit） |
| [7] | `max_radii2D` | [N] | float32 | **2D 投影最大半径**（用于剔除） |
| [8] | `xyz_gradient_accum` | [N, 1] | float32 | **位置梯度累积**（用于密集化） |
| [9] | `denom` | [N, 1] | float32 | **梯度计数器**（用于平均梯度） |
| [10] | `optimizer.state_dict()` | dict | - | **优化器状态**（Adam 动量等） |
| [11] | `spatial_lr_scale` | 标量 | float64 | **空间学习率缩放因子** |

**示例：**
```python
checkpoint = torch.load("chkpnt30000.pth")
model_params, iteration = checkpoint

print(f"迭代次数: {iteration}")  # 30000
print(f"高斯点数量: {model_params[1].shape[0]}")  # 例如 2,145,699
print(f"位置: {model_params[1].shape}")  # [2145699, 3]
print(f"颜色 DC: {model_params[2].shape}")  # [2145699, 1, 3]
```

---

### 格式 2: 包含语言特征（14 个参数）

**扩展的 Language Gaussian Splatting 模型**

| 索引 | 参数名 | 形状 | 数据类型 | 含义 |
|------|--------|------|----------|------|
| [0] | `active_sh_degree` | 标量 | int | 当前激活的球谐函数阶数 |
| [1] | `_xyz` | [N, 3] | float32 | 高斯中心位置 |
| [2] | `_features_dc` | [N, 1, 3] | float32 | 颜色特征 DC 分量 |
| [3] | `_features_rest` | [N, 15, 3] | float32 | 颜色特征其余分量 |
| [4] | `_scaling` | [N, 3] | float32 | 高斯尺度 |
| [5] | `_rotation` | [N, 4] | float32 | 高斯旋转 |
| [6] | `_opacity` | [N, 1] | float32 | 不透明度 |
| [7] | `_language_feature_logits` | [N, L×K] | float32 | **语言特征 logits**（稀疏系数的 logits） |
| [8] | `_language_feature_codebooks` | [L, K, D] | float32 | **语言特征码本**（共享的语义码本） |
| [9] | `max_radii2D` | [N] | float32 | 2D 投影最大半径 |
| [10] | `xyz_gradient_accum` | [N, 1] | float32 | 位置梯度累积 |
| [11] | `denom` | [N, 1] | float32 | 梯度计数器 |
| [12] | `optimizer.state_dict()` | dict | - | 优化器状态 |
| [13] | `spatial_lr_scale` | 标量 | float64 | 空间学习率缩放因子 |

**语言特征参数说明：**

#### [7] `_language_feature_logits`
- **形状**: `[N, L×K]`
  - `N`: 高斯点数量（如 2,145,699）
  - `L`: VQ 层数（如 1）
  - `K`: 码本大小（如 1024）
  - 例如: `[2145699, 1024]`

- **含义**: 每个高斯点对应的语言特征 logits，用于计算稀疏系数
  - 通过 `softmax + top-k` 得到稀疏权重
  - 稀疏权重 × 码本 = 高维语言特征

- **用途**:
  ```python
  # 计算稀疏系数
  soft_code = softmax_to_topk_soft_code(logits[:, 0:1024], k=4)
  # 重建语言特征
  language_feature = codebook.T @ soft_code  # [1152, N]
  ```

#### [8] `_language_feature_codebooks`
- **形状**: `[L, K, D]`
  - `L`: VQ 层数（如 1）
  - `K`: 码本大小（如 1024）
  - `D`: 特征维度（如 1152，对应 LLaVA 特征）
  - 例如: `[1, 1024, 1152]`

- **含义**: 全局共享的语义码本
  - 每一行是一个语义"原子"（semantic atom）
  - 所有高斯点共享这个码本
  - 通过不同的稀疏系数组合，表达不同的语义

- **用途**:
  ```python
  # 每个高斯点的语言特征是码本的线性组合
  # feature[i] = sum(soft_code[i, j] * codebook[j])
  ```

**示例：**
```python
checkpoint = torch.load("chkpnt_with_language_30000.pth")
model_params, iteration = checkpoint

print(f"参数数量: {len(model_params)}")  # 14

# 基础几何参数
print(f"高斯点数量: {model_params[1].shape[0]}")  # 2,145,699
print(f"位置: {model_params[1].shape}")  # [2145699, 3]

# 语言特征参数
logits = model_params[7]
codebooks = model_params[8]

print(f"Logits shape: {logits.shape}")  # [2145699, 1024]
print(f"Codebooks shape: {codebooks.shape}")  # [1, 1024, 1152]

# 计算一个高斯点的语言特征
point_idx = 0
point_logits = logits[point_idx]  # [1024]
soft_code = softmax_to_topk_soft_code(point_logits.unsqueeze(0), k=4)  # [1, 1024]
language_feature = codebooks[0].T @ soft_code.T  # [1152, 1]
print(f"语言特征维度: {language_feature.shape}")  # [1152, 1]
```

---

## 优化器状态字典（optimizer.state_dict()）

包含 Adam 优化器的完整状态：

```python
optimizer_state = model_params[10]  # (不含语言特征)
# 或
optimizer_state = model_params[12]  # (含语言特征)

# 结构：
{
    'state': {
        0: {'step': tensor(...), 'exp_avg': tensor(...), 'exp_avg_sq': tensor(...)},
        1: {'step': tensor(...), 'exp_avg': tensor(...), 'exp_avg_sq': tensor(...)},
        ...
    },
    'param_groups': [
        {
            'lr': 0.00016,
            'betas': (0.9, 0.999),
            'eps': 1e-15,
            'weight_decay': 0,
            'amsgrad': False,
            'maximize': False,
            'foreach': None,
            'capturable': False,
            'differentiable': False,
            'fused': None,
            'params': [0, 1, 2, ...]
        },
        ...
    ]
}
```

**字段说明：**
- `state`: 每个参数的优化器状态
  - `step`: 优化步数
  - `exp_avg`: 一阶矩估计（动量）
  - `exp_avg_sq`: 二阶矩估计（RMSprop）
- `param_groups`: 参数组配置（学习率、权重衰减等）

---

## 如何加载和使用

### 1. 加载检查点

```python
import torch

# 加载
checkpoint = torch.load("chkpnt30000.pth", weights_only=False)
model_params, iteration = checkpoint

print(f"从迭代 {iteration} 恢复训练")
print(f"参数数量: {len(model_params)}")

# 判断是否包含语言特征
has_language_feature = len(model_params) == 14
print(f"包含语言特征: {has_language_feature}")
```

### 2. 恢复到模型

```python
from scene import GaussianModel

# 创建模型
gaussians = GaussianModel(sh_degree=3, language_feature_dim=1152)

# 恢复参数
class MockArgs:
    include_feature = (len(model_params) == 14)

gaussians.restore(model_params, MockArgs(), mode='eval')
```

### 3. 提取特定参数

```python
# 提取位置
xyz = model_params[1]  # [N, 3]

# 提取颜色（需要组合 DC 和 rest）
features_dc = model_params[2]  # [N, 1, 3]
features_rest = model_params[3]  # [N, 15, 3]
all_features = torch.cat([features_dc, features_rest], dim=1)  # [N, 16, 3]

# 提取语言特征（如果有）
if len(model_params) == 14:
    logits = model_params[7]
    codebooks = model_params[8]

    # 计算第 i 个点的语言特征
    i = 0
    soft_code = softmax_to_topk_soft_code(logits[i].unsqueeze(0), k=4)
    lang_feat = codebooks[0].T @ soft_code.T  # [1152, 1]
```

---

## 参数数量和内存占用

### 示例：N = 2,145,699 个高斯点

#### 不含语言特征（12 参数）

| 参数 | 形状 | 元素数 | 内存 (GB) |
|------|------|--------|-----------|
| xyz | [N, 3] | 6,437,097 | 0.024 |
| features_dc | [N, 1, 3] | 6,437,097 | 0.024 |
| features_rest | [N, 15, 3] | 96,556,455 | 0.366 |
| scaling | [N, 3] | 6,437,097 | 0.024 |
| rotation | [N, 4] | 8,582,796 | 0.033 |
| opacity | [N, 1] | 2,145,699 | 0.008 |
| **总计** | - | **126,596,241** | **~0.48 GB** |

#### 含语言特征（14 参数）

在上述基础上增加：

| 参数 | 形状 | 元素数 | 内存 (GB) |
|------|------|--------|-----------|
| logits | [N, 1024] | 2,197,195,776 | 8.34 |
| codebooks | [1, 1024, 1152] | 1,179,648 | 0.004 |
| **语言特征总计** | - | **2,198,375,424** | **~8.35 GB** |
| **全部总计** | - | **2,324,971,665** | **~8.83 GB** |

---

## 版本兼容性

### 加载不同版本的检查点

```python
def load_checkpoint_safe(checkpoint_path):
    """安全加载检查点，兼容不同版本"""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_params, iteration = checkpoint

    param_count = len(model_params)

    if param_count == 12:
        print("标准 Gaussian Splatting 模型（无语言特征）")
        has_language = False
    elif param_count == 14:
        print("Language Gaussian Splatting 模型（含语言特征）")
        has_language = True
    else:
        raise ValueError(f"未知的参数数量: {param_count}")

    return model_params, iteration, has_language
```

### 转换检查点格式

```python
def add_language_features(checkpoint_12_path, output_path, codebook_path):
    """将 12 参数检查点转换为 14 参数"""
    # 加载原始检查点
    checkpoint = torch.load(checkpoint_12_path, weights_only=False)
    model_params_12, iteration = checkpoint

    # 加载码本
    codebooks = torch.load(codebook_path)  # [1, 1024, 1152]

    N = model_params_12[1].shape[0]
    L, K, D = codebooks.shape

    # 初始化 logits（全零）
    logits = torch.zeros(N, L * K)

    # 构建 14 参数
    model_params_14 = (
        model_params_12[0],   # active_sh_degree
        model_params_12[1],   # xyz
        model_params_12[2],   # features_dc
        model_params_12[3],   # features_rest
        model_params_12[4],   # scaling
        model_params_12[5],   # rotation
        model_params_12[6],   # opacity
        logits,               # language_feature_logits (新增)
        codebooks,            # language_feature_codebooks (新增)
        model_params_12[7],   # max_radii2D
        model_params_12[8],   # xyz_gradient_accum
        model_params_12[9],   # denom
        model_params_12[10],  # optimizer.state_dict()
        model_params_12[11],  # spatial_lr_scale
    )

    # 保存
    torch.save((model_params_14, iteration), output_path)
    print(f"转换完成: {checkpoint_12_path} -> {output_path}")
```

---

## 常见问题

### Q1: 如何查看检查点的基本信息？

```python
import torch

checkpoint = torch.load("chkpnt30000.pth", weights_only=False)
model_params, iteration = checkpoint

print(f"迭代次数: {iteration}")
print(f"参数数量: {len(model_params)}")
print(f"高斯点数量: {model_params[1].shape[0]:,}")

if len(model_params) == 14:
    print(f"Logits shape: {model_params[7].shape}")
    print(f"Codebooks shape: {model_params[8].shape}")
```

### Q2: 检查点文件太大怎么办？

可以只保存必要的参数：

```python
# 只保存推理需要的参数（不含优化器状态）
essential_params = model_params[:10]  # 去掉 optimizer 和其他训练状态
torch.save((essential_params, iteration), "lightweight.pth")
```

### Q3: 如何可视化高斯点分布？

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

xyz = model_params[1].numpy()  # [N, 3]

# 采样显示（太多点会很慢）
sample_size = 10000
indices = np.random.choice(xyz.shape[0], sample_size, replace=False)
xyz_sample = xyz[indices]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz_sample[:, 0], xyz_sample[:, 1], xyz_sample[:, 2],
           c=xyz_sample[:, 2], cmap='viridis', s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
```

---

## 总结

- **12 参数格式**: 标准 3D Gaussian Splatting（仅 RGB）
- **14 参数格式**: Language Gaussian Splatting（RGB + 语言特征）
- **关键新增参数**:
  - `logits [N, L×K]`: 稀疏系数的 logits
  - `codebooks [L, K, D]`: 全局共享的语义码本
- **内存占用**: 语言特征约占 8.35 GB（对于 2M 点，1024 码本）

这个结构设计允许每个高斯点通过稀疏的码本组合来表达复杂的语义信息！

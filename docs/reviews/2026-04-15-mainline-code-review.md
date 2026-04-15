# 当前实验主链路代码审查（2026-04-15）

## 范围

本次审查覆盖当前主链路：

- `run_all.sh`
- `train.py`
- `preprocess.py`
- `render_lerf_llm.py`
- `scene/cameras.py`
- `utils/vq_utils.py`
- `LLaVA-NeXT/verify_reconstruction_quality.py`
- `LLaVA-NeXT/compare_rag_results.py`
- `LLaVA-NeXT/rag_manager.py`

说明：当前 worker worktree 中 `LLaVA-NeXT/` 目录为空，本报告对这 3 个脚本的审查基于 leader 工作区中的同路径文件内容完成；未审查 `legacy/`、`ready_to_remove/` 和第三方内部模块。

## 本次顺手修复（低风险、确定性 bug）

1. `scene/cameras.py:29-44, 248-250`
   - **问题**：`get_llm_feature_tile()` 调用了未定义的 `CropFeatureCodec._resize_feature_map`，一旦未来启用 tile 解码或 quick render 分支，会直接 `NameError`。
   - **修复**：提取本地 `_resize_feature_map_nearest()`，让完整图解码与 tile 解码共用同一套 nearest-neighbor resize 逻辑，不再依赖缺失符号。

2. `preprocess.py:268-277`
   - **问题**：`mask_nms()` 在 fallback 分支里把一维布尔张量写成了二维索引（`keep_conf[index, 0] = True` 等）。当所有 mask 都被阈值筛掉时，这里会触发 `IndexError`，导致预处理直接崩溃。
   - **修复**：改成一维索引，并对 `topk` 做 `min(3, scores.numel())` 保护。

---

## 主要问题（按严重程度）

## HIGH

### 1. `render_lerf_llm.py` 的 quick render 路径目前不可用

- 位置：`render_lerf_llm.py:321-328`
- 证据：
  - `eval_gt_lerfdata()` 实际返回 2 个值（`gt_qa, img_paths`），但 `render_all_quick()` 按 3 个值接收：`gt_ann, image_shape, image_paths = ...`
  - `for i, idx in enumerate[int](tqdm(eval_index_list)):` 语法/调用本身错误
  - `eval_index_list = [int(idx) for idx in list(gt_ann.keys())]` 假设 key 可直接转 int，但主流程里 key 实际是图像名/帧名字符串
  - 下游继续使用旧版 `frame_{idx:05d}` 命名假设，与当前主流程 `idx` 直接等于图像 stem 的实现不一致
- 影响：`--quick_render` 一旦被打开，几乎必然在进入主循环前后直接崩溃；这不是边缘 case，而是整条备用渲染路径失效。
- 建议：要么明确删掉/禁用 `--quick_render`，要么把这条分支完整迁移到当前命名与数据结构。

### 2. 训练循环会因为缺失 scale 特征而“吞掉迭代”，导致有效训练步数不可控

- 位置：
  - `scene/cameras.py:166-179`：缺失某个 scale 时，`get_llm_feature()` 直接返回全零特征 + 空 mask
  - `train.py:186-211`：训练端遇到空 mask 后直接 `continue`
- 机制：
  1. 相机已经被随机采样并从 `viewpoint_stack` 中弹出；
  2. 渲染已经做完；
  3. 发现该尺度没有有效 GT 像素后直接 `continue`；
  4. 本次 iteration 不反向、不优化、不补采样。
- 影响：名义上训练了 `N` 次，但实际有效更新次数会小于 `N`，而且不同尺度/不同数据集缺失率不同，会把训练预算随机打折，导致结果不稳定且难复现。这个问题尤其会污染 Small / Medium / Large 横向比较。
- 建议：不要直接吞掉 iteration；至少应在空 mask 时重新采样 view，或单独统计“有效优化步数”并据此控制训练结束条件。

## MEDIUM

### 3. `run_all.sh` 的渲染完成判定过于宽松，部分失败会被误判为“已完成”

- 位置：`run_all.sh:291-299`
- 证据：只要 `RENDER_DIR` 存在且“非空”，就跳过该 level 的重新渲染。
- 影响：如果上次运行只留下日志、半截可视化图、单张 `feature_map` 或中途中断的残留文件，后续 rerun 会被错误短路，直接把不完整结果送入评估/RAG，对实验结论有污染风险。
- 建议：至少检查该 level 是否存在完整的 per-image 输出，或者检查预期数量/关键产物（例如每张图的 `feature_map_*.pt`）。

### 4. 结果目录约定不一致，容易把不同实验树混在一起

- 位置：
  - `run_all.sh:145-146, 272, 428-433`
  - `README.md:168-178`
  - `docs/PROJECT_GUIDE.md:18`
- 现象：
  - `run_all.sh` 默认 `EVAL_ROOT=eval_result`，因此主流程默认写到 `eval_result/{index}`；
  - 文档与 `bash_run/exp_0402.sh` 约定的是 `eval_result/exp_0402/{index}`；
  - leader 工作区里这两棵结果树目前同时存在。
- 影响：对比脚本、人工复盘、后续实验继承时，容易拿错一棵结果树，尤其是在比较 RAG 报告或 summary 时。
- 建议：统一只保留一个结果根目录约定，并在 wrapper / README / PROJECT_GUIDE 中同步。

### 5. 当前 RAG 注入策略对“可直接看图回答”的问题有系统性副作用

- 位置：`LLaVA-NeXT/verify_reconstruction_quality.py:389-412`
- 证据：所有启用 RAG 的问题都会先拼接一段背景资料，再附加“如果是视觉可见内容请忽略背景资料”的自然语言提示。
- 结合 comparison 输出的现象：
  - 当前 JudgeLM 报告（`eval_result/exp_0402/0402_iter10000_topk4_cb128/comparison/rag_comparison_report.txt`）显示：
    - Render Small: `+1.6667`
    - Render Medium: `+0.5667`
    - Render Large: `-0.4333`
  - 明显改进问题多是 landmark identity / city-country / historical name 这类知识题；
  - 明显退步问题则集中在 traffic sign、柱子数量、椭圆徽章下方细节等必须依赖图像局部观察的问题。
- 结论：当前实现更像“统一加外挂知识”，而不是“按题型选择性补知识”。对知识题有帮助，但对视觉细节题会把模型往语言先验上推，特别是 Large 本来视觉信息就最强，RAG 反而拉低了分数。
- 建议：
  - 先做 question-type gating（知识题再启用 RAG）；或
  - 先生成 image-only answer，再在知识缺口存在时做 second-pass RAG refinement。

## LOW

### 6. `run_all.sh` 的训练/评估链路已经强依赖仓库外部路径，复现实验时可移植性较差

- 位置：`run_all.sh:73-78, 152-154`
- 现象：JudgeLM、LLaVA、RAG-Anything、数据根目录都写死为本机路径。
- 影响：对当前作者本机不是 bug，但会显著增加别人复现实验或迁移机器时的配置成本。
- 建议：逐步转成环境变量优先、脚本参数覆盖、README 给最小配置模板。

---

## 方法层面的结论（基于当前 comparison 输出）

当前结果并不支持“RAG 对主链路稳定增益”的强结论，更准确的说法是：

1. **RAG 对知识恢复是有效的**：当 3D 重建后的特征不足以支撑 landmark identity / location / named-entity 问题时，RAG 明显补上了这部分信息。
2. **RAG 对纯视觉细节并不稳**：traffic sign、局部结构数量、构件位置关系这类题，RAG 会让模型更倾向于复述背景知识而不是读图。
3. **Large scale 已经接近 GT / RGB 时，额外背景注入更容易伤害而不是帮助**：这与报告里 Large 的平均收益为负一致。
4. **因此真正需要优化的，不只是检索质量，而是“什么时候该用 RAG、什么时候不要用”。**

这与 `LLaVA-NeXT/compare_rag_results.py:1419-1442` 里已经写出来的四段式解释是一致的：当前瓶颈已经不只是 feature quality，而是 feature / RAG / selector 三段之间的策略耦合。

---

## 建议优先级

### P0（应先处理）
- 修掉或下线 `render_lerf_llm.py --quick_render`
- 修正“空 mask 直接吞 iteration”的训练逻辑

### P1（建议尽快处理）
- 强化 `run_all.sh` 的完成判定，避免 partial output 被当成成功
- 统一结果目录根路径约定

### P2（下一轮方法实验）
- 对问题做知识题 / 视觉题拆分，再决定是否启用 RAG
- 把 image-only 与 RAG-refined 两阶段回答分开评估，而不是一次性拼 prompt


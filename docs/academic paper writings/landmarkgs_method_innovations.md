# LandmarkGS Method Innovation Draft

> 目标：把当前 LandmarkGS / LangSplatV2 代码中已经实现或有实验支撑的内容，整理成可放入学术论文 `Method` 部分的中文方法草稿。本文档遵循 `research-paper-writing` 的写法：先给出方法动机与模块划分，再说明每个模块“怎么做、为什么需要、为什么有效”，最后给出 claim-evidence map，避免把尚未由代码支持的内容写成既成事实。

## 0. Method mini-outline

**任务设定。** 给定一个 in-the-wild landmark 图像集合，LandmarkGS 先重建可渲染的 3D Gaussian 场，再将多尺度 LLaVA 语义特征蒸馏到 3D Gaussian 上，最后面向任意视角的 landmark 讲解 / VQA 任务，从 2D 渲染证据与 3D Gaussian 语义 token 中采样、融合，并交给 VLM 生成答案。系统目标不是只做开放词汇分割，而是服务于 XR 场景中的地标讲解：用户站在任意视角看到 landmark 时，系统需要回答“这是什么、这个部件在哪里/是什么、它有什么历史或文化含义”。

**核心贡献可以组织为三组：**

1. **Clean Multi-scale VLM Supervision for LandmarkGS.** 我们不是用单尺度 CLIP / segmentation feature 去监督 3D 语言场，而是从 LLaVA 的中间视觉 token 中提取 Small / Medium / Large 三个尺度的 crop-level 语义特征，并在重建前引入 “less is more” 的严格图像过滤，使 3DGS 与语言监督都建立在更干净、更一致的 landmark 观测上。
2. **Sparse Multi-scale Language Gaussian Field with 2D--3D Token-level Fusion.** 我们把高维 LLaVA 语义压缩为稀疏 codebook 系数并附着到 Gaussian 上；推理时不是简单把 3D 渲染成图再问 VLM，也不是先分别回答再拼接文本，而是在 LLaVA 前端 token 层面把 view-conditioned 2D rendered tokens 与 object/geometry-conditioned 3D Gaussian tokens 进行固定、无问题偏置的 scale-aligned fusion。
3. **Landmark-XR Benchmark and Evaluation Interface.** 我们构建了面向 landmark 讲解的 full benchmark（当前 V17/V18 使用的版本为 8 scenes / 74 images / 1061 VQA，其中 764 objective + 297 JudgeLM），并配套 9999 可视化页面与人工审核/导出机制。该部分可以写成 XR-facing benchmark and annotation/evaluation system：代码中已经有 web evaluation layer，VR/XR runtime 本身应表述为应用目标或系统接口，而不要过度声称已有完整 VR runtime。

---

## 1. Overview: LandmarkGS for XR landmark explanation

LandmarkGS 将地标讲解建模为一个 **view-conditioned 3D language field reasoning** 问题。输入是一组围绕同一 landmark 的 in-the-wild 照片。传统 3DGS 只优化 RGB 可渲染性，难以回答诸如“顶部雕塑是什么”“门洞与柱廊的位置关系是什么”“这个部件对应什么历史语义”这类 XR 导览问题；传统 2D VLM 又只能对当前图片推理，无法显式利用同一 landmark 的多视角、三维一致性。LandmarkGS 的方法路径是：

1. 先用严格的 VLM filter 过滤重建照片，得到 clean reconstruction set；
2. 对保留图像提取 Small / Medium / Large 三个尺度的 LLaVA crop features；
3. 在冻结 RGB 3DGS 几何与外观的基础上，只学习附着在 Gaussian 上的稀疏语言系数与 codebook；
4. 对任意测试视角渲染多尺度 3D language feature map，并与 2D rendered feature tokens / 3D Gaussian tokens 做 token-level fusion；
5. 用 landmark-specific benchmark 同时评估 objective correctness 与 open-ended JudgeLM quality。

这个 pipeline 的关键是把 “landmark 讲解” 从单张图像问答，提升为以 3DGS 为载体的多视角语言场问答。它强调三类信息的互补：clean views 保证重建与监督的稳定性，multi-scale LLaVA features 提供细粒度部件到全局语义的覆盖，2D--3D token fusion 则让 VLM 同时看到当前视角证据和跨视角 3D 语义锚点。

---

## 2. Clean multi-scale LLaVA feature supervision

### 2.1 Less-is-more reconstruction set

**Motivation.** In-the-wild landmark photos 往往包含游客、车辆、广告牌、强滤镜、夜景、极端天气、低清晰度或严重裁剪。对一般 VQA 数据集而言，这些图片有时仍可用；但对 3DGS 重建与 3D 语言场蒸馏而言，它们会引入不稳定几何、漂浮物、错误遮挡与跨视角不一致的语义监督。LandmarkGS 的第一处方法创新是把数据清洗前置为重建方法的一部分：不是“照片越多越好”，而是 **less is more**——少量但自然、清晰、无遮挡、重建友好的照片，比大量噪声图像更适合构建可解释的 landmark 3D language field。

**Design.** 代码中的 `filter_images_with_vlm.py` 与 `tools/filtering/run_filter_pipeline.sh` 实现了 VLM-based filtering。严格模式 `benchmark_reconstruction_clean` 要求保留图像必须满足：没有真实游客/身体部位、没有车辆、没有临时遮挡物或文字广告、没有 heavy HDR / oversaturation / sepia / black-and-white / night / fog / storm / severe exposure 等非自然外观，并且图像清晰、主体裁剪完整。过滤逻辑还针对 landmark 固定部件做了显式保护，例如 Brandenburg Gate 的 Quadriga、Trevi Fountain 的雕塑/水体、Taj Mahal 的水池/宣礼塔、Notre-Dame 的立面雕像、Pantheon 的柱廊与方尖碑等，不把这些永久建筑元素误判为人或 clutter。

**Technical advantage.** 该策略与常见 in-the-wild reconstruction pipeline 的差异在于：它不是只依赖 SfM/重建阶段的几何鲁棒性，而是在语义训练之前就用 landmark-aware VLM 判断“哪些图像适合作为 3D 语义场的监督源”。这使后续 LLaVA feature extraction、codebook fitting 与 Gaussian language training 都在更一致的视觉分布上进行，从源头降低了游客、车辆、滤镜、夜景等临时因素被写入 3D 语言场的风险。

### 2.2 Multi-scale LLaVA crop features

**Motivation.** Landmark 讲解问题同时需要小尺度细节、中尺度结构和大尺度场景语义。例如，Quadriga、rose window、golden shibi、Oceanus sculpture 属于局部部件；columns、arches、eaves、facade symmetry 属于中尺度结构；city/country、landmark identity、architectural style 等又依赖全局上下文。单一尺度的 2D feature 很容易在“细节”和“全局”之间折中失败。因此 LandmarkGS 将 LLaVA 的视觉 token 提取改为显式多尺度。

**Design.** `LLaVA-NeXT/extract_llava_features.py` 中的 `SemanticMultiScaleCropperFixed` 为每张图生成三个尺度的 crop：Small / Medium / Large，对应默认 crop size 约 64 / 192 / 448。crop 生成由 semantic mask 约束，只在建筑/地标区域采样；同时引入 depth-adaptive crop size，使近处区域与远处区域拥有不同视野；overlap ratio 被控制以避免大量重复 crop。对每个 crop，`LLaVAFeatureExtractor.extract_crop_feature` 通过 LLaVA 的 multimodal preparation 得到 27×27×3584 的视觉 token map，并以 crop-level 形式保存到 `llava_features_3584_multiscale/*.pth`。`scene/cameras.py` 中的 `_decode_crop_features_to_full_map` 再用 center-weighted overlap blending 将 crop token 解码回 dense feature map；`get_llm_feature` 通过 feature level 0/1/2 分别读取 Small/Medium/Large。

**Technical advantage.** 这里的关键不是简单把图片 resize 成三种分辨率，而是把 VLM 内部视觉 token 作为可监督的语言特征，并保留 crop 的 bbox、scale 和 native 27×27 token layout。这样 Small scale 更偏向 landmark 部件，Medium scale 覆盖局部结构关系，Large scale 捕获整体外观和上下文。与只用 CLIP embedding 或 segmentation label 的方法相比，这种监督更贴近 VLM 后续回答问题时实际使用的 feature space，因此更适合训练一个能被 VLM 重新消费的 3D language field。

---

## 3. Sparse multi-scale language Gaussian field

**Motivation.** LLaVA feature 维度为 3584，如果直接给每个 Gaussian 存完整高维向量，会带来极高的显存、存储和优化成本，也容易在有限视角监督下过拟合。LandmarkGS 因此把语言场表示为“codebook + sparse coefficients”：Gaussian 不直接存 dense semantic vector，而是学习对全局 codebook 的稀疏组合。

**Design.** 在 `train.py` 中，当 `--llm_feature` 启用时，系统将 language feature dimension 设置为 3584，并按 feature level 加载或训练 `llm_codebooks_L{layer}_K{size}_level{level}_{small|medium|large}.pt`。`utils/vq_utils.py` 中的 `ResidualVectorQuantizationWithClustering` / `LMMFeatureStream` 支持从磁盘流式读取多尺度 crop features，并用 MiniBatchKMeans 拟合 codebook，避免把所有 3584 维 feature 一次性载入显存。`scene/gaussian_model.py` 为每个 Gaussian 维护 `_language_feature_logits` 与 `_language_feature_codebooks`；`get_render_weights` 对 logits 做 top-k softmax，得到每个 Gaussian 的稀疏语言系数；`compute_final_feature_map` / `compute_layer_feature_map` 再把 rendered language weight map 与 codebook 相乘，恢复出渲染视角下的 LLaVA feature map。

训练时，`GaussianModel.training_setup` 冻结 RGB 3DGS 的 xyz、SH、scale、rotation、opacity，只优化语言 logits 与 codebook。这一选择把几何/外观重建与语言蒸馏解耦：RGB 3DGS 保持可渲染性，语言训练只负责给已有 3D structure 写入语义。`train.py` 还提供 `llm_crop_native_supervision_loss`，它不强制先把所有 crop feature 展开成巨大的 dense map，而是将渲染特征按 crop bbox 裁剪并 resize 到原生 27×27 token grid 后计算损失。这与前面的 crop-level LLaVA supervision 保持一致，减少了高维 dense supervision 的内存压力。

**Technical advantage.** 稀疏 codebook 表示让 3D language field 兼具可渲染性、压缩性和 token 可采样性：同一个 Gaussian 的语言向量可以被渲染成 2D feature map，也可以作为 3D token 直接进入 VLM。更重要的是，该表示把“训练后采样”变成可能：我们可以从 visible Gaussians 中按几何覆盖、语义多样性或 opacity 选择 token，而无需重新训练 3DGS。

---

## 4. Scale-aligned 2D--3D token-level fusion

### 4.1 Why token-level fusion

LandmarkGS 的第二组主要创新是融合位置：fusion 不发生在最终文本答案之后，而发生在 LLaVA 输入 token 之前。原因是，text-level fusion 会让 2D branch 与 3D branch 先各自做出离散答案，后续 LLM 只能在答案之间选择或总结，难以恢复被某个 branch 早期丢失的视觉证据。相反，token-level fusion 让 VLM 在一次 forward/generation 中同时访问 2D rendered evidence 与 3D Gaussian evidence，从而保留联合推理的机会。

这里的 “2D” 与 “3D” 不是谁优先谁次要的关系。2D rendered tokens 保留当前视角中的纹理、可见边界和局部外观；3D Gaussian tokens 来自重建场，包含跨视角稳定的 object/geometry anchors。方法上我们避免 question-aware routing 或 scene-adaptive rule，因为它们会把 benchmark 问题类型或场景身份先验注入系统，造成额外偏置；当前设计采用固定 token budget 与固定 3D ratio，使实验更接近可复现的 architecture ablation。

### 4.2 Token sampling and scale-wise modality fusion

**Design.** `tools/benchmark/run_v18_phase_a.py` 实现了训练后采样的 V18 ablation。对 2D rendered feature map，系统支持 uniform sampling (`u2d`)、representative sampling (`rep2d`) 与 feature-norm sampling (`norm2d`)；对 3D Gaussians，系统支持 geometry FPS (`geo3d`)、feature/code-signature FPS (`feat3d`)、opacity top-k (`opa3d`) 与 geometry+feature mixed sampling (`mix3d`)。采样由总 token budget `T` 与 3D ratio `R` 控制：`T` 表示送入 VLM 的总 feature token 数，`R` 表示其中分配给 3D Gaussian tokens 的比例。

最有代表性的 fusion layout 是 `mmtokscale`，可写成：

\[
Z = [Z^{2D}_{small}, Z^{3D}_{small},
     Z^{2D}_{medium}, Z^{3D}_{medium},
     Z^{2D}_{large}, Z^{3D}_{large}],
\]

其中每个 \(Z\) 都是同一 scale 下采样得到的 LLaVA feature token 序列。也就是说，fusion 先按 scale 对齐，再在每个 scale 内合并 2D 与 3D evidence，最后把三个 scale 的结果作为一条连续 feature sequence 传给 LLaVA。`verify_reconstruction_quality.py` 中的 `answer_question` 接收 `encoded_image_features=features`，将融合后的 feature sequence 直接作为 VLM 的视觉 token 输入，而不是重新渲染成 RGB 或先生成多个文本答案。

**Technical advantage.** `mmtokscale` 的意义在于同时保留两种结构：multi-scale structure 与 modality structure。Small scale 的 2D/3D token 共同描述细粒度部件，Medium scale 的 2D/3D token 共同描述局部建筑结构，Large scale 的 2D/3D token 共同描述 landmark identity 与全局布局。相比简单的 all-2D-then-all-3D 或 token-by-token interleaving，scale-wise fusion 更符合 landmark 语义的层级性。

**Current empirical signal.** V18 是“训练后采样”的 ablation study，不应过度写成最终性能结论。当前 full benchmark 上，代表性方法 `v18_tokpipe_best2d_best3d_m2tokcat_m3tokcat_mmtokscale_odirect_T576_R5_C1` 在 8 scenes / 74 images / 1061 questions 上达到 Objective 77.23%，低于 RGB objective 79.71%；在诊断性 `full_minus_rgb_only_correct_v1` 子集上达到 Objective 82.52%，高于该子集 RGB 78.32%。这说明 token-level 2D--3D fusion 是当前采样 ablation 中较强的方向，但完整 benchmark 尚未证明其超过 RGB，后续改进应更多回到 feature extraction / training quality 与更原则性的 fusion architecture。

---

## 5. Landmark-XR benchmark and evaluation interface

**Motivation.** 面向 XR landmark explanation 的方法不能只用通用 VQA 指标评价。系统需要知道自己是否能识别地标、定位/区分关键部件、回答视觉+知识混合问题，并在不同 view 下保持稳定。因此 LandmarkGS 配套构建了一个 landmark-specific benchmark，而不是只汇报重建质量或单张图问答结果。

**Design.** 当前 V17/V18 使用的 full benchmark 覆盖 8 个 landmark scenes：Brandenburg Gate、Buckingham Palace、Notre-Dame front facade、Pantheon exterior、Sacré-Cœur、Taj Mahal、Temple in Nara、Trevi Fountain。评测集包含 74 张测试图、1061 个问题，其中 764 个 objective questions 用 exact/objective accuracy 评价，297 个开放式问题用 JudgeLM score 评价。`/home/wangyz/project/3vibe_tools/harness/server.py` 中的 `LangSplatBenchmarkAdapter` 定义了 scene titles、视觉/混合/知识三类问题标签、每个场景的重点部件说明，并支持 9999 页面上的图像卡片、QA 审核、人工状态记录与 curated export。

**Technical advantage.** 这个 benchmark 的价值在于它专门针对 landmark XR 导览需要的能力拆分：

- **visual questions** 检查当前视角中具体部件是否可见、数量/位置/类别是否正确；
- **mixed questions** 检查视觉证据与 landmark 常识是否能结合，例如建筑材料、风格、部件名称；
- **knowledge questions** 检查地标身份、城市国家、历史语义等讲解内容。

9999 可视化网页不是论文中的核心算法，但它是方法闭环的重要系统组件：它让我们能逐图查看问题、答案、错误类型、不同方法/RGB baseline 的对比，并持续导出 curated benchmark。若论文中要加入 VR system，可以把它表述为 **XR-facing Landmark Explanation System**：当前代码已实现 benchmark visualization / annotation / evaluation layer 与从 3DGS 到 VLM answer 的 pipeline；完整 VR runtime 若没有对应代码，不应写成已经完成的独立系统，而应写成该 evaluation layer 可以服务的 XR deployment target。

---

## 6. Suggested paper-style contribution wording

可以在 introduction 或 method overview 中压缩成如下贡献句：

1. **Multi-scale VLM supervision for 3D language Gaussians.** We introduce a clean multi-scale LLaVA feature extraction pipeline for landmark language fields, where landmark-aware VLM filtering removes reconstruction-hostile in-the-wild images and Small/Medium/Large crop tokens provide part-, structure-, and scene-level supervision.
2. **Sparse language Gaussian representation with scale-aligned 2D--3D token fusion.** We represent high-dimensional LLaVA features through sparse codebook coefficients attached to Gaussians, and reason over landmarks by fusing rendered 2D tokens and sampled 3D Gaussian tokens at the input-token level of LLaVA, rather than via post-hoc answer concatenation.
3. **A landmark-centered XR explanation benchmark.** We build a multi-scene landmark VQA benchmark and web-based evaluation interface covering objective and open-ended questions, enabling systematic diagnosis of whether 3D language fields improve landmark explanation beyond RGB-only VLM baselines.

如果需要突出 “fusion” 作为单独创新，而不把 sparse field 写成贡献，可以改成四点：

1. less-is-more clean reconstruction + multi-scale LLaVA feature extraction；
2. sparse codebook-based 3D language Gaussian field；
3. scale-aligned 2D--3D token-level fusion；
4. landmark-XR benchmark and visualization/evaluation system。

---

## 7. Claim-evidence map from code reading

| Claim | Code / artifact evidence | Safe wording boundary |
|---|---|---|
| Multi-scale LLaVA feature extraction exists | `LLaVA-NeXT/extract_llava_features.py`: `SemanticMultiScaleCropperFixed`, default scales 64/192/448, saved `feature_maps` with crop `bbox` and 27×27×3584 feature | 可写为已实现；不要写成已证明所有尺度都单独优于 baseline，除非补 ablation 表 |
| Crop-level native supervision exists | `train.py`: `llm_crop_native_supervision_loss`; `scene/cameras.py`: `_decode_crop_features_to_full_map` | 可写为 memory-aware / scale-preserving supervision |
| Less-is-more VLM filtering exists | `filter_images_with_vlm.py`, `tools/filtering/run_filter_pipeline.sh`; strict rejection of people/vehicles/clutter/filter/night/weather/blur | 可写为 data curation / reconstruction-clean filtering；若宣称提升重建质量，需要配套 before/after numbers |
| Sparse codebook language Gaussian field exists | `scene/gaussian_model.py`: language logits/codebooks/top-k soft coefficients; `utils/vq_utils.py`: RVQ/KMeans/LMMFeatureStream | 可写为压缩高维 LLaVA feature 的实际表示 |
| Geometry frozen during language training | `GaussianModel.training_setup`: xyz/SH/scale/rotation/opacity frozen; only language logits/codebooks optimized | 可写为 decoupled semantic distillation on pretrained RGB 3DGS |
| 2D/3D token-level fusion exists | `tools/benchmark/run_v18_phase_a.py`: `sample_2d`, `sample_3d`, `fuse_tokpipe`, `mmtokscale`; `verify_reconstruction_quality.py`: `answer_question(encoded_image_features=...)` | 可写为 current ablation-supported direction；完整 benchmark 尚未超过 RGB objective |
| Benchmark exists | `v17_2d3d_comparison/summary.json`, V18 summaries: 8 scenes / 74 images / 1061 Q / 764 objective / 297 JudgeLM; `server.py`: `LangSplatBenchmarkAdapter` | 可写为 benchmark/evaluation interface；raw label pool 数量可能更大，论文应以最终 frozen split 为准 |
| VR/XR system | 代码中主要证据是 web dashboard / evaluation adapter / VQA pipeline；未发现完整 VR runtime | 写成 XR-facing benchmark/system interface 或 application target；不要声称完整 VR runtime 已实现 |

---

## 8. Self-review for paper readiness

- **清晰性：** 三个核心创新分别对应数据/特征、3D 表示与 fusion、benchmark/system，避免把所有实验命名堆在 method 里。
- **新颖性表述：** 强调 multi-scale LLaVA feature supervision 与 prior semantic feature extraction 的差别；强调 less-is-more 是 landmark reconstruction 的数据方法；强调 fusion 是 token-level joint reasoning，而不是 text answer 拼接。
- **证据边界：** V18 当前是训练后采样 ablation。可说 token-level fusion 是“当前最强方向/代表性策略”，但不能说 full benchmark 已经超过 RGB objective。
- **XR 边界：** 目标是 landmark explanation in XR；当前代码支持 web-based benchmark/evaluation layer 与 VLM answering pipeline。除非后续补 VR runtime 代码，否则论文中把 VR 写成 application scenario / deployment target 更稳妥。
- **下一步可补实验：** 若要把 method claim 写得更强，建议补 clean vs unfiltered reconstruction 对比、Small/Medium/Large 单尺度 vs 多尺度对比、crop-native supervision vs dense supervision、mmtokscale vs text-fusion 的 frozen split ablation。

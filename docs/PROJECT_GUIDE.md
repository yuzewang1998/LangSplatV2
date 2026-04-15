# LangSplatV2 项目整理说明

## 当前主流程

目前真正的主入口是：

- `run_all.sh`: 训练 -> 渲染 -> RAG -> 评估 -> 对比报告的一站式流程
- `bash_run/exp_0402.sh`: 当前唯一保留的实验入口，负责把本轮训练和评估统一输出到 `exp_0402/`
- `train.py` / `render_lerf_llm.py` / `preprocess.py`: 训练、渲染、预处理核心代码
- `filter_images_with_vlm.py`: 训练前的图像过滤主脚本
- `filter_brandenburg.sh`: Brandenburg Gate 数据过滤的固定入口
- `tools/filtering/run_filter_pipeline.sh`: 通用过滤入口，适用于任意 PT 风格数据集

## 整理后的目录约定

- `output/`: 训练 checkpoint
- `output/exp_0402/`: 当前重新起跑实验的 checkpoint 根目录
- `eval_result/exp_0402/{index}/`: 当前重新起跑实验的渲染、评估、RAG 对比结果
- `eval_result/filtering/`: 图像过滤相关可视化输出
- `rag_data/`: RAG 缓存与索引
- `JudgeLM`: 当前 RAG 对比默认使用本地 `conda` 环境 `judgelm` 和 `/home/wangyz/project/2past_project/JudgeLM-main`
- `tools/filtering/`: 过滤链路的辅助脚本
- `docs/archive/`: 历史说明文档归档
- `legacy/`: 已不属于当前主流程的旧评估脚本和实验性可视化脚本

## 现在建议你关注的文件

如果你要继续做实验，优先只看这些：

- `run_all.sh`
- `bash_run/exp_0402.sh`
- `train.py`
- `render_lerf_llm.py`
- `visualize_comparison_simple_rgb.py`
- `filter_images_with_vlm.py`
- `filter_brandenburg.sh`
- `tools/filtering/run_filter_pipeline.sh`
- `tools/filtering/test_vlm_filter.py`
- `tools/filtering/visualize_filtered_images.py`

## 图像过滤目前的状态

过滤链路现在已经具备这几部分：

- `filter_images_with_vlm.py`: 逐张读取 TSV 中的图像，调用 Ollama VLM，输出筛选后的 TSV
- `filter_brandenburg.sh`: 做路径配置、Ollama 可用性检查，然后调用上面的主脚本
- `tools/filtering/run_filter_pipeline.sh`: 通用版封装，自动输出 `<xxx>_filtered.tsv`、判定日志和可视化目录
- `tools/filtering/test_vlm_filter.py`: 先对单张图片测 prompt 和模型回复
- `tools/filtering/visualize_filtered_images.py`: 看过滤结果样本和可选的全量总览图

我目前确认到的状态：

- 这几个脚本都能通过 Python 语法检查
- `filter_images_with_vlm.py --help`、`tools/filtering/test_vlm_filter.py --help` 能正常运行
- `tools/filtering/visualize_filtered_images.py --help` 能运行，但在受限环境下会出现 matplotlib/fontconfig 缓存警告；脚本本身已改为把输出统一写到 `eval_result/filtering/`
- 真正能不能完整跑通，取决于两件事：
  - 本机 Ollama 服务是否启动
  - 对应的 `llava`/`llava:34b` 模型是否已安装

## 过滤链路的推荐使用顺序

1. 先用单图测试 prompt 是否合理：

```bash
python tools/filtering/test_vlm_filter.py \
    --image_path /abs/path/to/frame.jpg \
    --model_name llava:34b
```

2. 再跑整套过滤：

```bash
bash tools/filtering/run_filter_pipeline.sh \
    --dataset_dir /abs/path/to/dataset \
    --input_tsv xxx.tsv \
    --landmark_name "Your Landmark Name"
```

3. 最后看过滤可视化：

```bash
python tools/filtering/visualize_filtered_images.py \
    --filtered_tsv /home/wangyz/data/PT/brandenburg_gate/brandenburg_filtered.tsv \
    --image_dir /home/wangyz/data/PT/brandenburg_gate/dense/images \
    --save_full_grid
```

## 已归档的内容

以下内容不是当前核心流程的一部分，已经移出主目录：

- 旧版 `eval_lerf*` 评估脚本
- 多个试验性的特征可视化变体
- 多份历史说明文档
- 历史过滤可视化图片和日志

如果后面需要比对老实现，可以去 `legacy/` 和 `docs/archive/` 查看。

## 一个重要提醒

仓库里旧文档曾提到 `eval_lerf.py`、一些 3584 特征提取脚本和解码脚本，但它们并没有构成你现在的主实验路径。你现在继续做 Brandenburg Gate 实验时，应该把注意力放在 `run_all.sh` 和过滤链路，不要再被旧文档带偏。

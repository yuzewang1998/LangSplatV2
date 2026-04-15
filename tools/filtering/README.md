# Filtering Pipeline

## 目标

给一个 PT 风格的数据集目录和一个输入 TSV，自动完成：

- 生成 `<xxx>_filtered.tsv`
- 生成 `<xxx>_filtered_decisions.jsonl`
- 生成过滤结果可视化图

这里的 PT 风格目录至少需要满足：

```text
<dataset_dir>/
├── dense/
│   └── images/
└── <xxx>.tsv
```

## 推荐入口

使用通用脚本：

```bash
bash tools/filtering/run_filter_pipeline.sh \
    --dataset_dir /abs/path/to/dataset \
    --input_tsv xxx.tsv \
    --landmark_name "Your Landmark Name"
```

## 最常见用法

### 1. 只有一个原始 TSV，直接生成过滤结果

```bash
bash tools/filtering/run_filter_pipeline.sh \
    --dataset_dir /home/wangyz/data/PT/some_scene \
    --input_tsv some_scene.tsv \
    --landmark_name "Some Scene"
```

输出会自动写到：

- `/home/wangyz/data/PT/some_scene/some_scene_filtered.tsv`
- `/home/wangyz/data/PT/some_scene/some_scene_filtered_decisions.jsonl`
- `/home/wangyz/project/0working/LangSplatV2/eval_result/filtering/some_scene/some_scene_filtered/`

### 2. 输入 TSV 用绝对路径也可以

```bash
bash tools/filtering/run_filter_pipeline.sh \
    --dataset_dir /home/wangyz/data/PT/brandenburg_gate \
    --input_tsv /home/wangyz/data/PT/brandenburg_gate/brandenburg.tsv \
    --landmark_name "Brandenburg Gate"
```

## 参数说明

- `--dataset_dir`: 数据集目录，内部必须有 `dense/images`
- `--input_tsv`: 输入 TSV；可传文件名或绝对路径
- `--landmark_name`: 地标名；会直接进入 VLM prompt，建议认真写
- `--ollama_host`: Ollama 地址；默认 `192.168.192.124`
- `--ollama_port`: Ollama 端口；默认 `11434`
- `--model_name`: 默认 `llava:34b`
- `--request_timeout`: 单次请求超时；默认 `180`
- `--resume_from`: 断点续跑起始行号；默认 `0`
- `--num_samples`: 可视化随机样本数；默认 `20`
- `--full_grid_cols`: 全量总览图列数；默认 `15`

## 输出说明

### 1. `<xxx>_filtered.tsv`

最终保留的图片列表。

### 2. `<xxx>_filtered_decisions.jsonl`

逐图结构化判定日志。每行一条 JSON，方便后续排查误判。

### 3. 可视化目录

至少会有：

- `filtered_samples.png`: 过滤结果随机样本
- `filtered_all.png`: 过滤结果全集总览

## 单图调试

在全量跑之前，建议先拿单图试一下：

```bash
python tools/filtering/test_vlm_filter.py \
    --image_path /abs/path/to/image.jpg \
    --ollama_host 192.168.192.124 \
    --model_name llava:34b \
    --landmark_name "Brandenburg Gate" \
    --request_timeout 180
```

## 当前默认策略

当前 prompt 偏保守，优先满足：

- 严格去掉真实人、车、旗子、标牌、横幅、明显遮挡
- 尽量排除滤镜感、明显奇怪外观
- 对 Brandenburg Gate 特别处理：允许门顶的 Quadriga 雕像，不把它当真人或遮挡

如果你换新场景，`--landmark_name` 要改成对应场景名。

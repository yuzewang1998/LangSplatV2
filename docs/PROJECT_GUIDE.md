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

## Codex / 登录切换恢复 Tip

如果以后再次遇到 `codex-switch openai` 还在要求登录，或者 `codex` 打开后 provider / telemetry 看起来不对，按下面流程排查，通常可以快速恢复：

1. **先确认本地认证已经导入**
   - 将已有凭证放到 `~/.codex/auth.json`
   - 再执行 `codex login status`
   - 正常情况下应显示 `Logged in using ChatGPT`

2. **记住入口分工**
   - `codex` 是真正进入交互界面的入口
   - `codex-switch openai` / `codex-switch mirror` 只是切换后端 provider
   - 如果还看到登录页，通常不是“切换命令没跑”，而是本地认证或残留进程有问题

3. **如果切换后 `/status` 还是旧 provider，优先查残留 app-server**
   - 重点看 `~/.codex/app-server-control/app-server.log`
   - 检查是否有旧的 `codex app-server` / proxy 链路还在
   - 这类残留会把 config/auth 缓存在内存里，表现为 `codex-switch mirror` 后 `/status` 仍显示 openai
   - 当前修复在 `~/.local/bin/codex-switch`：切换 provider 后会刷新本用户的 app-server/proxy helper，但不会杀普通交互式 `codex` 会话

4. **避免切换后 conversation 丢失，同时保持“一个文件夹一组会话”**
   - openai/mirror 可以共享同一个 `CODEX_HOME`，但 `/resume` 必须按**精确当前 `cwd`**过滤
   - 不要把子目录合并到项目根目录：`ArchStudio` 和 `ArchStudio/stage1` 必须是两组不同会话
   - `codex-switch` 只能同步“当前精确目录”的 SQLite thread provider 和对应 rollout `session_meta.model_provider`，不能全局改所有项目，也不能把父/子目录归一到一起
   - 不要重建全局 `~/.codex/session_index.jsonl`；它没有 cwd 字段，容易让所有 project 的会话混在一起
   - 当前已把坏的全局索引移到 `~/.codex/session_index.jsonl.global-bad-20260520`，并移除 `~/.codex-live/session_index.jsonl` 链接，让 Codex 走 SQLite 的 cwd 过滤
   - 验证命令：`sqlite3 ~/.codex/state_5.sqlite "SELECT cwd, model_provider, COUNT(*) FROM threads GROUP BY cwd, model_provider;"`

5. **如果需要 daemon 管理但启动失败**
   - 先检查 `~/.codex/packages/standalone/current/codex` 是否存在
   - 不存在时不要卡在 daemon 上，直接用 `codex` 主入口即可

6. **排障时可打开调试**
   - `CODEX_WRAPPER_DEBUG=1 codex ...`
   - 这样更容易确认当前到底用了哪个 binary、home 和 provider

7. **最后提醒**
   - 不要把任何 token 或敏感认证内容写进仓库
   - 只保留流程说明和排障线索即可

## 结果目录约定（重要）

当前推荐的唯一结果目录约定是：

- checkpoint 根目录：`output/exp_0402/`
- 渲染/评估根目录：`eval_result/exp_0402/`

也就是说，默认应该通过：

- `bash_run/exp_0402.sh`

来启动实验。

如果你直接调用 `run_all.sh`，也建议显式传：

```bash
bash run_all.sh --output_root output/exp_0402 --eval_root eval_result/exp_0402 ...
```

避免再次出现：

- `eval_result/{index}`
- `eval_result/exp_0402/{index}`

两棵结果树同时存在、人工复盘时容易拿错的问题。

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

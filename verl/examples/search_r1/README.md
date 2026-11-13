# Search-R1 with verl

Search-R1 风格的 GRPO 训练：本地 E5 + FAISS 维基检索、verl 内置的 `searchR1_*` reward、
以及多轮 `tool_agent` 工具调用循环。目标硬件是 4×B300（GPU0 同时跑检索服务）。

## 目录布局

工作区根目录（`$SEARCH_R1_HOME`，也就是 git 仓库根）只跟踪 `verl/`；数据、实验、日志和
虚拟环境都在同一层，但不进 git。

```
$SEARCH_R1_HOME/                 # 工作区根 = git 仓库根
├── verl/                        # 代码（verl@3255f6b1 + Search-R1 示例，Python 包在 verl/verl/）
├── .venv/  .venv-mig/  .tools/  .cache/
├── data/
│   ├── nq_hotpotqa_train/       # 原始 Search-R1 parquet
│   ├── search_r1/{train,test,test_sub}.parquet
│   ├── retrieval/{e5_Flat.index,wiki-18.jsonl}
│   └── models/{Qwen3.5-2B,e5-base-v2}
├── experiments/<EXP>/
│   ├── meta.json
│   ├── checkpoints/
│   └── runs/<TS>.{metrics.jsonl,log,command.sh,hydra/}
└── logs/retriever.log
```

`SEARCH_R1_HOME` 默认是代码目录（`verl/`）的上一级，也就是工作区/git 仓库根
（脚本在 `verl/examples/search_r1/`，向上三级）。所有路径都可以用环境变量覆盖
（`SEARCH_R1_HOME`、`MODEL_PATH`、`TRAIN_FILE`、`TEST_FILE`、`TOOL_CONFIG`）。

## 一次性准备

```bash
cd "$SEARCH_R1_HOME"
uv pip install --python .venv/bin/python -e verl --no-deps
```

用 venv 的 Python 安装，不要用 `uv run` / `uv sync`（见文末的坑）。

## 预处理

```bash
python3 verl/examples/data_preprocess/search_r1.py \
    --input_dir "$SEARCH_R1_HOME/data/nq_hotpotqa_train" \
    --local_save_dir "$SEARCH_R1_HOME/data/search_r1" \
    --val_samples_per_source 200
```

会写出 `train.parquet`、`test.parquet`，以及分层抽样的 `test_sub.parquet`（每源 200 条）。

## 启动检索服务

```bash
bash verl/examples/search_r1/start_retriever.sh
```

日志在 `$SEARCH_R1_HOME/logs/retriever.log`，脚本会打印 PID 和 curl 自检命令。
GPU 用 `RETRIEVER_GPU=1` 指定（默认 0）。

## 冒烟训练

```bash
SMOKE=1 bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
```

跑 2 步，不做验证、不存 checkpoint，结果在 `experiments/smoke/<TS>-smoke/`。
先 `DRY_RUN=1` 看看解析出来的路径和完整命令：

```bash
EXP=demo DRY_RUN=1 bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
```

## 正式训练

```bash
EXP=search-r1-001 BG=1 bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
```

`EXP` 是唯一的实验名（同时也是 trainer 的 experiment_name、实验目录和 checkpoint 目录）。
`BG=1` 会 nohup 到后台，打印实验目录、log、metrics、PID 和 `tail -f` 命令。
不加 `BG=1` 则前台运行并同时 tee 到 log。

每次训练都会写 `runs/<TS>.command.sh`（可重放）和 `runs/<TS>.hydra/`，
并更新 `meta.json`（`status` 从 `running` 到 `finished` / `failed (exit N)`）。
额外的 Hydra override 直接作为参数追加，优先级最高：

```bash
EXP=search-r1-001 BG=1 bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    actor_rollout_ref.actor.optim.lr=5e-7
```

## 续训

```bash
EXP=search-r1-001 RESUME=1 BG=1 bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
```

`RESUME=1` 要求实验目录已存在，并设置 `trainer.resume_mode=auto`。
不加 `RESUME=1` 时，如果 `experiments/<EXP>` 已存在会直接报错，避免覆盖旧实验。

## 导出 metrics

```bash
bash verl/examples/search_r1/pack_metrics.sh search-r1-001   # 或 smoke/<TS>-smoke
bash verl/examples/search_r1/pack_metrics.sh all             # 所有实验打成一个包
```

生成 `experiments/_export/<name>_<TS>.tar.gz`，只含 `meta.json`、`runs/*.metrics.jsonl`、
`runs/*.command.sh`、`runs/*.hydra/.hydra/*.yaml`，不含 checkpoint 和几百 MB 的 `*.log`。
把这个 tar.gz 下载到本地给 dashboard 用即可。

## 踩过的坑

- **`use_remove_padding=True` 会让 Qwen3.5 崩**：GatedDeltaNet 下 rollout 与训练的
  KL 从 6e-4 涨到 ~0.05，训练在 60 步左右 collapse。本 recipe 固定为 `False`，并把
  `ppo_max_token_len_per_gpu` 降到 12288（不开 rmpad 时 24576 峰值约 219GB）。
- **不要复用已有 EXP**：第二次训练会被拒绝；要接着跑就加 `RESUME=1`，否则换一个 `EXP`。
- **不要让 `uv run` / `uv sync` 动 `.venv`**：它们会按 `uv.lock` 重建环境，把不在 lock 里的
  `faiss-gpu` 删掉。装包用 `uv pip install --python .venv/bin/python ...`。
- **`.venv-mig` 只给 MIG 切片用**，普通 4×B300 训练不要用它。

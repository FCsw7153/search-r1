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
source "$SEARCH_R1_HOME/.venv/bin/activate"
"$SEARCH_R1_HOME/.tools/uv" pip install --python .venv/bin/python -e verl --no-deps
```

`uv` 不在服务器的 PATH 里，用 `.tools/uv`；先激活 `.venv`，脚本里的 `python3` 都来自它
（包括 `start_retriever.sh`）。不要用 `uv run` / `uv sync`（见文末的坑）。

## 预处理

```bash
python3 verl/examples/data_preprocess/search_r1.py \
    --input_dir "$SEARCH_R1_HOME/data/nq_hotpotqa_train" \
    --local_save_dir "$SEARCH_R1_HOME/data/search_r1" \
    --val_samples_per_source 200
```

会写出 `train.parquet`、`test.parquet`，以及分层抽样的 `test_sub.parquet`（每源 200 条）。
服务器上 `data/search_r1/` 已经生成好了，这一步只需要做一次。

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
`GPUS=0,1,2,3` / `VAL=full` 是默认值：`GPUS` 决定 `CUDA_VISIBLE_DEVICES`，并按个数设置
`trainer.n_gpus_per_node`；`VAL=sub` 换成 `test_sub.parquet`（分层抽样，验证更快）。
`BG=1` 会 nohup 到后台，打印实验目录、log、metrics、PID、pid 文件和 `tail -f` 命令。
不加 `BG=1` 则前台运行并同时 tee 到 log。

每次训练都会写 `runs/<TS>.command.sh`（可重放）、`runs/<TS>.hydra/` 和
`runs/<TS>.pid`（仅 `BG=1`），并更新 `meta.json`（`status` 从 `running` 到
`finished` / `failed (exit N)`，另外记录生效后的 `train_batch_size`、`rollout_n` 和
`overrides`，dashboard 用它们算 samples seen）。
额外的 Hydra override 直接作为参数追加，优先级最高：

```bash
EXP=search-r1-001 BG=1 bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    actor_rollout_ref.actor.optim.lr=5e-7
```

## 一个节点并行跑两个实验（每个 2 卡）

```bash
# slot A：GPU0 上有检索服务，显存少 ~35GB
EXP=exp-a GPUS=0,1 VAL=sub BG=1 bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
# slot B
EXP=exp-b GPUS=2,3 VAL=sub BG=1 bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
```

两个 job 各起一个本机 Ray 集群、各占 2 张卡，互不共享进程和资源：

- `GPUS` 设置 `CUDA_VISIBLE_DEVICES`，Ray 的 raylet 由 driver 启动并继承该变量，
  所以每个集群只看得到自己的 2 张卡；`trainer.n_gpus_per_node` 取 `GPUS` 的个数。
- **Ray 的 `ray.init()` 会自动连已有集群**：无 `address` 时它先看 `RAY_ADDRESS`，再找
  `<temp_dir>/ray_current_cluster`，找到就连上去（两个 job 共享 `/tmp/ray` 时，第二个 job
  会静默变成第一个的第二个 driver，GPU/actor 名字/TransferQueue controller 全部串台）。
  所以每个实验用独立的 `_temp_dir=/tmp/ray-<cksum(EXP_DIR)>`，并显式传 `address=local`
  （崩溃残留的 `ray_current_cluster` 和外部 `RAY_ADDRESS` 都不会被连）。用 `/tmp` 下的短路径
  而不是实验目录，是因为 Ray 会在 `<temp_dir>/session_*/sockets/` 下建 AF_UNIX socket：
  实验目录在共享网络/FUSE 盘上时可能不支持 Unix socket、路径容易超过 107 字节上限，而且
  Ray 启动也慢（之前在这块盘上超时过一次）。每个 job 的 Ray 会话/日志都在这个 `/tmp/ray-*`
  目录里，`DRY_RUN=1` 和 `BG=1` 的输出会打印它。
- **权重同步的 IPC socket 也要分开**：colocated 的 FSDP→vLLM 权重传输用
  `/tmp/rl-colocate-zmq-{job_id}-replica-{r}-rank-{l}.sock`，而两个独立 Ray 集群的第一个
  job 都叫 `01000000`，socket 名会撞。本仓库为此做了一处最小代码改动：socket 名优先读
  `VERL_ZMQ_JOB_TOKEN` 环境变量（`vllm_rollout.py`、`vllm_async_server.py`、
  `vllm_pd_replica.py`），launcher 把它设成 `<cksum(EXP_DIR)>-<launcher pid>`（每 job 唯一，
  长度固定，避免长 EXP 把 socket 路径顶到 108 字节上限）。没有这个改动时并行 job 的权重同步
  会互相踩（可能连到另一个 job 的 engine）。
- 端口方面没有会硬失败的固定端口：Ray 的 GCS / raylet / object manager / metrics 端口都是 0
  （随机）。dashboard 关掉后即使还有进程碰 8265，Ray 的 dashboard 端口逻辑会自动换 8266+
  （不致命）；dashboard agent 的固定端口 52365 第二个 job 绑不上时异常会被吞掉，只是降级。
  vLLM 的 HTTP 是 `port=0`，vLLM master/DP 和 torch 分布式 master port 都是现找空闲端口。
  TransferQueue（外部包）的 controller/storage 端口也是动态分配，controller 是各自集群里的
  named actor；只有显式选 Mooncake/Yuanrong 后端时才有固定端口，别用它们。
- `_temp_dir` 的副作用：verl 有几处硬编码 `/tmp/ray/session_latest`
  （nsight profiler、Prometheus 配置、metrics service discovery）。我们不用这些功能；以后要用
  profiler/metrics 时要记得这条。
- `/dev/shm`：两个 Ray 集群各建自己的 object store，但共享同一个 `/dev/shm` 容量；verl 的
  数据走 TransferQueue，object store 用量很小。不要给两个 job 同时开
  `use_shm=True`（`/dev/shm/verl-cache/<md5>` 是共享路径）。
- 两个 Ray 集群各自都会看到节点全部 CPU（`ray.init` 的 `num_cpus` 默认全量），CPU 是超卖的；
  如果 agent loop / tokenizer 抢 CPU 明显，给每个 job 加 `+ray_kwargs.ray_init.num_cpus=<N>`。
- GPU0 还要跑检索服务（~35GB），所以 `GPUS=0,1` 的 job 显存更紧；`gpu_memory_utilization=0.6`
  保持不变（B300 268GB 下单卡预算见 `ablations.md` 的显存估算）。
- 第一次并行时在服务器上确认一下：`nvidia-smi` 看两个 python 各占 2 卡；
  两个日志里不应该出现 Ray 的 "Connecting to existing Ray cluster"；`ray --version` 应是
  2.55.x（GCS 随机端口的依据）；`ls /tmp/rl-colocate-zmq-*.sock` 应该是两个不同的名字。

**只停一个 job**：`BG=1` 的进程组 leader pid 写在
`experiments/<EXP>/runs/<TS>.pid`（`BG=1` 用 `setsid` 启动，python 和它的 Ray 进程都在这个
进程组里）：

```bash
kill -- -$(cat experiments/exp-a/runs/<TS>.pid)
```

不要用 `ray stop --force`：它按节点清理 Ray 进程，不区分是哪个 job 的，会把同节点另一个
job 的 Ray 一起杀掉；而这里的会话目录又在实验目录下、不在 `/tmp/ray`，行为更不可控。
只停一个 job 一律用进程组。

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
- **同时跑两个实验时不要用 `ray stop --force`**：它会连带杀掉同节点另一个 job 的 Ray；
  用 `experiments/<EXP>/runs/<TS>.pid` 杀本 job 的进程组（见上一节）。
- **不要让 `uv run` / `uv sync` 动 `.venv`**：它们会按 `uv.lock` 重建环境，把不在 lock 里的
  `faiss-gpu` 删掉。装包用 `uv pip install --python .venv/bin/python ...`。
- **`.venv-mig` 只给 MIG 切片用**，普通 4×B300 训练不要用它。

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
│   ├── checkpoints/             # 默认目录；CKPT_ROOT 设置时是指向本地盘的符号链接
│   ├── kept_checkpoints/        # keep_checkpoint.sh 归档到共享盘的 checkpoint
│   └── runs/<TS>.{metrics.jsonl,log,command.sh,hydra/}
└── logs/retriever.log
```

`/home/work/search_r1_local`（`LOCAL_ROOT`）是容器本地 overlay 上的加速目录，不在工作区里；
`stage_local.sh` 把 venv 和检索数据放过去，pod 重建后重跑一次即可（见下文）。

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

## 本地盘加速：LOCAL_ROOT 与 stage_local.sh

`$SEARCH_R1_HOME` 挂在 spfs（慢速网络 FUSE，95% 满）；容器本地的 overlay（`/`，约 2.7T
空闲）快得多，但 pod/run 结束就清空。`stage_local.sh` 把大件拷到本地盘：

```bash
bash verl/examples/search_r1/stage_local.sh            # 等价于 all
bash verl/examples/search_r1/stage_local.sh venv       # 只搞 venv
bash verl/examples/search_r1/stage_local.sh retrieval  # 只搞检索数据
```

默认 `LOCAL_ROOT=/home/work/search_r1_local`（故意放在 `mshop-agent-1` 之外，属于本地盘；
可以覆盖，但脚本会拒绝指向工作区）：

- `venv`：`rsync -a --delete`（没有 rsync 就 `cp -a`）把 `$SEARCH_R1_HOME/.venv` 拷到
  `$LOCAL_ROOT/.venv`，然后把写死旧路径的地方改掉——`bin/*` 里以旧 venv 开头的 shebang，
  以及 `bin/activate*` 里的 `VIRTUAL_ENV`（`pyvenv.cfg` 不用改，`home` 指向系统 python）。
  可编辑安装的 `.pth`/finder 仍指向 `$SEARCH_R1_HOME/verl`，不要动；脚本会用
  `"$LOCAL_ROOT/.venv/bin/python" -c "import verl, torch, vllm, faiss"` 验证。验证通过后
  才写完成标记 `$LOCAL_ROOT/.venv.ok`。然后用
  `compileall -j 16` 预热 `__pycache__`，并打印本地/共享 venv 各自
  `python -c 'import torch, vllm'` 的耗时（冷节点上共享 venv 要 2 分钟以上）。
- `retrieval`：把 `data/retrieval/{e5_Flat.index,wiki-18.jsonl}` 和 `data/models/e5-base-v2`
  拷到 `$LOCAL_ROOT/data/...`；大小一致的文件直接跳过，打印吞吐。全部拷完并逐文件校验
  大小后，才写完成标记 `$LOCAL_ROOT/data/.retrieval.ok`。

两处标记都会在开始拷贝对应部分时先删掉，所以拷贝中断、失败或 venv 验证不通过时标记
不存在。脚本幂等，可重复跑。之后 launcher 在未显式设置 `VERL_PYTHON` 且
`$LOCAL_ROOT/.venv.ok` 存在时才用 `$LOCAL_ROOT/.venv/bin/python`（Ray worker 用
`sys.executable`，整个集群跟着走）；`start_retriever.sh` 同样只认标记，`data/.retrieval.ok`
存在才用本地数据和本地 python。本地目录存在但缺标记时，脚本会打印一行提示并回落到共享盘；
显式 `VERL_PYTHON=...` 仍然优先，`LOCAL_ROOT=` 可以关掉本地盘路径。
`DRY_RUN=1` 会打印实际用的 `VERL_PYTHON`/`PYTHON_BIN`，`meta.json` 记录 `python`。

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

## 在这台 pod 上启动长任务（hc run）

从 Mac 上**不要**用 `hc exec` 跑训练：`hc exec` 返回时会把这次调用 spawn 的所有进程杀
掉（`setsid`/`nohup`/`tmux` 都不管用），job 会被 SIGTERM 杀掉（`meta.json` 会写
`failed (signal 15)`）。长任务用 `hc run --bg` 开一个持久终端：

```bash
hc run --on <machine> --bg --no-parent --name search-r1-<EXP> \
    bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh ...
hc read <terminal> --lines 50     # 看输出
hc wait <terminal>                # 等它结束
```

`<machine>` 用 `hc machine ls` 里的 `pod-<runId>-…`。如果直接在交互式 WebShell/Herdr
终端里敲命令，`BG=1` 仍然可用（pid 文件和进程组控制不变）。

## 正式训练

```bash
EXP=search-r1-001 BG=1 bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
```

`EXP` 是唯一的实验名（同时也是 trainer 的 experiment_name、实验目录和默认 checkpoint
目录 `$EXP_DIR/checkpoints`；用 `CKPT_ROOT` 时 checkpoint 实际在本地盘，见下节）。
`GPUS=0,1,2,3` / `VAL=full` 是默认值：`GPUS` 决定 `CUDA_VISIBLE_DEVICES`，并按个数设置
`trainer.n_gpus_per_node`；`VAL=sub` 换成 `test_sub.parquet`（分层抽样，验证更快）。
`BG=1` 会 nohup 到后台，打印实验目录、log、metrics、PID、pid 文件和 `tail -f` 命令。
不加 `BG=1` 则前台运行并同时 tee 到 log。

每次训练都会写 `runs/<TS>.command.sh`（可重放）、`runs/<TS>.hydra/` 和
`runs/<TS>.pid`（仅 `BG=1`），并更新 `meta.json`（`status` 从 `running` 到
`finished` / `failed (exit N)` / `failed (signal N)`；被 SIGTERM/SIGINT/SIGHUP 杀掉时记
`failed (signal N)`，不会写成 `finished`。另外记录生效后的 `train_batch_size`、`rollout_n`、
`overrides`、`checkpoint_dir`、`checkpoint_local` 和实际用的 `python`；
`launches[].log`/`.metrics` 是相对实验目录的路径）。
额外的 Hydra override 直接作为参数追加，优先级最高：

```bash
EXP=search-r1-001 BG=1 bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    actor_rollout_ref.actor.optim.lr=5e-7
```

## Qwen3-4B-Base：原始 Search-R1 tag 协议（search_r1_agent）

Qwen3-4B-Base 是 base 模型，几乎不会输出 native `<tool_call>`（冒烟测试里约 0.15% 的
trajectory），但会自然写 `<search>...</search>` / `<answer>...</answer>`。因此新增了
`search_r1_agent`：纯文本 prompt（**不套 chat template**），协议与
Search-R1（`PeterGriffinJin/Search-R1`）一致——`<think>` 推理、`<search> query </search>`
检索、结果放进 `<information>...</information>`、最后 `<answer> ... </answer>`。

- prompt 是上游 `scripts/data_process/nq_search.py` 里 `make_prefix(..., "base")` 模板的
  逐字复刻；题目从现有 parquet `prompt` 的 user 消息里取，所以**不需要重新预处理数据**，
  data_source 仍是 `searchR1_*`，`search_r1_like_qa_em` 的 EM reward 和验证指标不变。
  prompt 里的示例 `<answer> Beijing </answer>` 只存在于 prompt tokens；reward manager 只
  decode response（`data.batch["responses"]`，不含 prompt），不会匹配到它。
- 检索信息以 token 形式追加进 response，`response_mask=0`（参与 attention、不进 loss）；
  每轮 `stop=["</search>", "</answer>"]` 且 `include_stop_str_in_output=True`，停止标记
  本身保留在 response tokens 里（和 Search-R1 保留 `</search>` 一致）。

用现有 launcher 加下面的 overrides 即可（1 卡、batch 128、800 步；`VAL=sub` 验证更快）：

```bash
EXP=qwen3-4b-base-sr1 VAL=sub GPUS=0 BG=1 \
MODEL_PATH="$SEARCH_R1_HOME/data/models/Qwen3-4B-Base" \
bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
  actor_rollout_ref.rollout.agent.default_agent_loop=search_r1_agent \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  data.train_batch_size=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
  trainer.total_training_steps=800 \
  trainer.rollout_data_dir=/home/work/rollouts/qwen3-4b-base-sr1/train \
  trainer.validation_data_dir=/home/work/rollouts/qwen3-4b-base-sr1/val
```

- 数据文件沿用 launcher 的默认值：`TRAIN_FILE=$SEARCH_R1_HOME/data/search_r1/train.parquet`，
  `VAL=sub` 时 `TEST_FILE=$SEARCH_R1_HOME/data/search_r1/test_sub.parquet`（可用同名环境变量覆盖）。
- `multi_turn.tool_config_path` / `max_tool_response_length=3000` / `max_response_length=3072`
  都沿用现有 recipe；`search_r1_agent` 直接使用其中名为 `search` 的工具（同一检索 URL 和 topk）。
  `max_assistant_turns` 是**生成轮数**（和 `tool_agent` 一致）。基线用 **5**：最多 4 轮
  `<search>` + 最后再生成一次 `<answer>`，与上游 Search-R1 一致。
- **最后一轮不再检索**：如果最后一次允许的 generation 仍以 `</search>` 结束，或者这次
  generation 已用完 `max_response_length`，loop 直接结束，不再调用 retriever、也不追加
  `<information>`（这些 token 只会被 mask 掉）。因此 `num_turns = user_turns +
  assistant_turns + 1 <= 2 * max_assistant_turns`；此前 `max_assistant_turns=4` 时观察到
  `num_turns/max=9` 的问题已修复。
- `use_remove_padding=True` 对 Qwen3 的普通 attention 没问题（README 里的 `False` 是给
  Qwen3.5 GatedDeltaNet 的）。rmpad 下用 `ppo_max_token_len_per_gpu=24576` 的 token 预算。
- `tokenization_sanity_check_mode=disable`：本 loop 不套 chat template，但只要配置了 tool
  config，worker 仍会对 `tool_agent` 跑一次 chat-template sanity check；这一步与本 loop
  无关，关掉即可。
- 停止与 EOS：Qwen3-4B-Base 的 `generation_config.json` 里 eos 只有 `<|endoftext|>`
  （151643）。本 loop 用纯文本 prompt、不套 chat template，vLLM 按 generation config
  处理 EOS 即可；每轮显式设置上面两个 stop strings，所以不依赖 chat 结束符。
- 单卡时 GPU0 默认还跑检索服务（`RETRIEVER_GPU=0`，~35GB），rollout 用
  `gpu_memory_utilization=0.6`；显存紧就把检索服务放到别的卡。

### 每条轨迹的结束原因与检索统计

`search_r1_agent` 为每条 trajectory 写三个 `extra_fields`（经 reward manager 的
`tool_extra_fields` → `extra_info` 传到 `search_r1_like_qa_em.compute_score`，再放进
`reward_extra_info`）：

| extra_fields | 含义 |
| --- | --- |
| `search_r1_end_reason` | `answer` / `turn_cap` / `length` / `no_action` |
| `search_r1_num_searches` | 实际执行的检索次数 |
| `search_r1_queries` | 检索 query 列表（按执行顺序） |

结束原因：`answer` = 以 `</answer>` 结束；`turn_cap` = 达到 `max_assistant_turns` 时还在
检索；`length` = response token 预算耗尽；`no_action` = EOS / 没有 tag / tag 不完整。

reward 返回的 dict 里另有 4 个 0/1 字段 `search_r1_end_{answer,turn_cap,length,no_action}`
（均值就是占比）。数值字段在 logger 里聚合，字符串 `search_r1_end_reason` 和列表
`search_r1_queries` 只进 dump（verl 的验证指标聚合会跳过非标量）。

- **训练指标**（每个 step，console + `runs/<TS>.metrics.jsonl`）：V1 trainer 的
  `_compute_metrics` 现在会读取每条样本 `extra_fields.reward_extra_info` 里的数值字段，
  输出 `training/reward_extra/<key>/{mean,min,max}`，例如
  `training/reward_extra/search_r1_end_turn_cap/mean`、
  `training/reward_extra/search_r1_num_searches/mean`。原有的
  `training/num_turns/{mean,max,min}`、`num_turns/{mean,max,min}` 仍然保留。
- **验证指标**（每次验证）：走已有的 `val-aux/<data_source>/<key>/mean@N`，例如
  `val-aux/searchR1_nq/search_r1_end_answer/mean@1`（`VAL=sub` 时每个 data_source 各自的
  200 条；`n=1` 所以只有 `mean@1`）。
- **rollout dump**（`trainer.rollout_data_dir`，V1 路径）：每个 step 一个
  `<dir>/<global_step>.jsonl`，每行一条 trajectory：
  `input`（decode 后的 Search-R1 prompt）、`output`（decode 后的 response，含
  `<information>`）、`gts`、`score`（EM reward）、`step`、`uid`，以及
  `reward_extra_info` 的所有 key（`acc`、`search_r1_end_reason`、
  `search_r1_end_*`、`search_r1_num_searches`、`search_r1_queries`）。
- **验证 dump**（`trainer.validation_data_dir`）：同样格式的 `<dir>/<global_step>.jsonl`，
  每次验证写一份。
- **大小**：800 步 × (128 prompts × `rollout.n=5`) = 512,000 条。单条约 5–8 KB
  （prompt ~0.5 KB，response 最多 3072 tokens ≈ 12 KB），整轮训练 dump 约 3–4 GB；全部
  trajectory 都接近长度上限时约 6–7 GB。放 pod 本地盘（`/home/work/rollouts/...`，pod
  重建即清空），不要放共享盘。

## Checkpoint 放本地盘：CKPT_ROOT

checkpoint 是 FSDP 模型 + fp32 optimizer，每个 25GB（4 rank），写共享盘很慢。设置
`CKPT_ROOT` 后写到本地盘：

```bash
EXP=search-r1-001 CKPT_ROOT=/home/work/ckpt BG=1 \
    bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
```

实际目录是 `$CKPT_ROOT/<EXP>`（smoke 是 `$CKPT_ROOT/smoke/<TS>-smoke`，和 experiments/
下的相对路径一致，两个实验不会互相覆盖），并建符号链接
`$EXP_DIR/checkpoints -> $CKPT_ROOT/<EXP>`：`RESUME=1` 和平常找 checkpoint 都用这个固定
路径。已有的符号链接会被替换；真实的非空目录会被拒绝（避免把旧 checkpoint 藏起来）。
`meta.json` 记录 `checkpoint_dir`（真实路径）和 `checkpoint_local`。

pod 重建后如果**不带** `CKPT_ROOT` 启动，`$EXP_DIR/checkpoints` 可能还是指向已清空本地盘
的失效符号链接；脚本会明确报错退出（不会自动删除），按提示设置 `CKPT_ROOT` 重跑或手动
删除该链接即可。

**风险**：本地盘在 pod/run 重建时清空，checkpoint 一起丢，**RESUME 不可能**（`RESUME=1`
但本地目录为空时脚本会警告，trainer 从第 0 步开始）。200 步的消融丢了重跑可以接受；
长训练不要用 `CKPT_ROOT`，或者定期用下面的脚本往共享盘归档。

## 归档 checkpoint：keep_checkpoint.sh

```bash
bash verl/examples/search_r1/keep_checkpoint.sh <EXP> latest
bash verl/examples/search_r1/keep_checkpoint.sh <EXP> 200
bash verl/examples/search_r1/keep_checkpoint.sh <EXP> best          # 推荐
bash verl/examples/search_r1/keep_checkpoint.sh <EXP> best --move   # 校验后删本地副本
```

把 `$EXP_DIR/checkpoints/global_step_<N>`（CKPT_ROOT 时通过符号链接读本地盘）拷到**共享盘**
`$EXP_DIR/kept_checkpoints/global_step_<N>`，打印源/目标大小、文件数和拷贝耗时，并校验
文件数 + 字节数一致。`best` 从 `runs/*.metrics.jsonl` 里取每个验证 step 的 7 个
`val-core/searchR1_*/acc/mean@1` 做未加权平均，挑平均最高的验证 step，再取**已保存的、
最接近且 ≤ 它的 step**（会打印 val best 和实际保存的是哪个 step）。`--move` 在校验通过后
删掉本地副本——注意如果删的是最新 checkpoint，`RESUME=1` 会找不到它（脚本会警告）。

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
用 `CKPT_ROOT` 时，续训从 `$EXP_DIR/checkpoints`（本地盘符号链接）找 checkpoint；
pod 重建后本地 checkpoint 已经丢了，脚本会警告并从第 0 步开始——要跨 pod 续训，提前用
`keep_checkpoint.sh` 把需要的 step 归档到共享盘。

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
- **不要用 `hc exec` 跑训练**：`hc exec` 返回时会杀掉它 spawn 的所有进程（`setsid`/
  `nohup`/`tmux` 也不行），job 以 `failed (signal 15)` 结束。长任务用
  `hc run --on <machine> --bg --no-parent --name <n> bash <script>`。
- **`CKPT_ROOT` 是本地盘，pod/run 重建就没了**：checkpoint 丢了就没法 `RESUME`；
  长训练要么不用 `CKPT_ROOT`，要么定期 `keep_checkpoint.sh` 归档。
- **不要复用已有 EXP**：第二次训练会被拒绝；要接着跑就加 `RESUME=1`，否则换一个 `EXP`。
- **同时跑两个实验时不要用 `ray stop --force`**：它会连带杀掉同节点另一个 job 的 Ray；
  用 `experiments/<EXP>/runs/<TS>.pid` 杀本 job 的进程组（见上一节）。
- **不要让 `uv run` / `uv sync` 动 `.venv`**：它们会按 `uv.lock` 重建环境，把不在 lock 里的
  `faiss-gpu` 删掉。装包用 `uv pip install --python .venv/bin/python ...`。
- **`.venv-mig` 只给 MIG 切片用**，普通 4×B300 训练不要用它。

# Search-R1 GRPO 消融实验（Qwen3.5-2B，4×B300）

本目录的消融实验都基于 `run_qwen3_5_2b_search_r1_grpo.sh` 的默认配置，每个 job 用 2 张卡，
同节点同时跑两个：`GPUS=0,1` 和 `GPUS=2,3`。GPU0 还跑着检索服务（~35GB），所以 `GPUS=0,1`
的 job 实际可用显存少一些；`gpu_memory_utilization` 保持 0.6 不变（见文末的显存估算）。

## 约定

- 命令都在 `$SEARCH_R1_HOME` 下执行，先 `source "$SEARCH_R1_HOME/.venv/bin/activate"`。
- 所有实验共享：

  ```bash
  trainer.total_training_steps=200 trainer.test_freq=25 trainer.save_freq=50
  VAL=sub          # $SEARCH_R1_HOME/data/search_r1/test_sub.parquet，每 25 步验证
  ```

  （C1 例外，见下。）`VAL=full` 会换成完整 7 数据集 `test.parquet`。
- 两组 GPU 用 `GPUS=0,1` / `GPUS=2,3`，launcher 会设置 `CUDA_VISIBLE_DEVICES` 和
  `trainer.n_gpus_per_node=2`；`EXP` 决定实验目录、checkpoint、Ray 的 `_temp_dir` 和权重同步
  IPC socket 的 token。并行的隔离细节见 README。
- 每步“看过的样本数”= `step × data.train_batch_size × actor_rollout_ref.rollout.n`
  （512×5=2560/步；C1 是 64×5=320/步）。`meta.json` 现在记录了 `train_batch_size` 和
  `rollout_n`，dashboard 可以直接算；**A1 开了动态采样，实际生成的样本比这个多**，
  横向比较时以 x 轴的 samples seen 为准，不要用 step。
- 停一个 job 用它的 pid 文件杀进程组（`kill -- -$(cat .../runs/<TS>.pid)`），
  **不要用 `ray stop --force`**，那会把同节点另一个 job 的 Ray 一起杀掉。详见 README。

## 实验总表

| ID | EXP | 变量 | 相对 A0 的改动 | 建议配对 |
|---|---|---|---|---|
| A0 | `20261114-abl-baseline-noTIS` | 关闭 TIS 的基线 | launcher 默认 + `rollout_is=null` | 与 A1 |
| A1 | `20261114-abl-dapo` | DAPO 三件套 | A0 + clip-higher + 动态采样 + overlong 奖励惩罚 | 与 A0 |
| B0 | `20261114-abl-rmpad-nocorr` | rmpad 不修正（可选） | A0 + `use_remove_padding=True` + `ppo_max_token_len_per_gpu=24576` | 与 D1 |
| B1 | `20261114-abl-rmpad-tis` | token 级 IS | B0 + `rollout_is=token` | 与 B2 |
| B2 | `20261114-abl-rmpad-seqmis` | sequence IS + RS | B0 + `rollout_is=sequence` + `rollout_rs=seq_sum_k1` | 与 B1 |
| B3 | `20261114-abl-rmpad-sc` | score centering | B0 + bypass + REINFORCE + token TIS + score centering | **当前多轮 recipe 跑不了**，见下 |
| B3' | `20261114-abl-rmpad-bypass-rf` | bypass + REINFORCE | B0 + bypass + REINFORCE + token TIS（无 score centering） | 与 C1 |
| C1 | `20261114-abl-batch64` | 全 on-policy 小 batch | A0 + `train_batch_size=64 ppo_mini_batch_size=64`，步数 ×8 | 与 B3' |
| D1 | `20261114-abl-gspo` | GSPO | A0 + GSPO loss / clip / agg | 与 B0 |

## 逐条命令

### A0：no-TIS 基线（对照组）

launcher 默认是 `rollout_is=token`（当前正式 run 的配置）。A0 把它关掉，其余完全一致。

```bash
EXP=20261114-abl-baseline-noTIS GPUS=0,1 VAL=sub BG=1 \
bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    algorithm.rollout_correction.rollout_is=null \
    trainer.total_training_steps=200 trainer.test_freq=25 trainer.save_freq=50
```

### A1：DAPO（clip-higher + 动态采样 + overlong）

```bash
EXP=20261114-abl-dapo GPUS=2,3 VAL=sub BG=1 \
bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    algorithm.rollout_correction.rollout_is=null \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_inflight_gen_batches=1 \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=True \
    +reward.reward_kwargs.max_resp_len=3072 \
    trainer.total_training_steps=200 trainer.test_freq=25 trainer.save_freq=50
```

要点（V1 sync + reward-loop 路径的实测结论）：

- `filter_groups.metric=acc`：我们的 `searchR1_*` reward 返回 float，naive reward manager 只写
  `reward_extra_info["acc"]`；写 `score` 会在第一个完成的 group 上报
  `RuntimeError: Finished groups are missing DAPO metric 'score'`。`reward` 也能用（读
  `rm_scores`），但 A1 开了 overlong 惩罚后 `reward` 会被长度惩罚污染，所以选 `acc`。
- `max_num_gen_batches` 在 V1 内置 ReplayBuffer 里**被忽略**（`trainer_base.py` 里明确 warn），
  用 `max_inflight_gen_batches` 限制并发；group 被淘汰后会自动补采样，batch size 不缩水。
- 被过滤的是“组内 5 条 reward 完全相同（全对或全错）”的 prompt，不是只丢 reward=0 的组。
- overlong 处理只有**奖励惩罚**：`reward.reward_manager.name=dapo` 读
  `reward.reward_kwargs.max_resp_len` / `overlong_buffer_cfg`，超过
  `max_resp_len - len` 的部分线性扣分。`max_resp_len` 不会自动取
  `data.max_response_length`，这里手工对齐成 3072；多轮场景里 response 含检索/tool token，
  惩罚按这些 token 一起算。**本版本不支持把截断样本从 loss 里 mask 掉**（没有对应配置项）。
- 同一个 reward manager 也作用于验证集：A1 的 `val/reward` 含 overlong 惩罚，和 A0 不可直接
  比较；看 `val/acc`（纯正确率）或者统一用训练曲线里的 `acc`。

### B0：rmpad 不修正（可选，复现 mismatch）

`use_remove_padding=True` 是已知的 mismatch 来源（Qwen3.5 GatedDeltaNet 下
`rollout_corr/kl≈0.05`，约 60 步 collapse）。rmpad 省显存，因此把
`ppo_max_token_len_per_gpu` 恢复到 24576。

```bash
EXP=20261114-abl-rmpad-nocorr GPUS=0,1 VAL=sub BG=1 \
bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    algorithm.rollout_correction.rollout_is=null \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    trainer.total_training_steps=200 trainer.test_freq=25 trainer.save_freq=50
```

### B1：rmpad + token 级 IS

```bash
EXP=20261114-abl-rmpad-tis GPUS=0,1 VAL=sub BG=1 \
bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    algorithm.rollout_correction.rollout_is=token \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    trainer.total_training_steps=200 trainer.test_freq=25 trainer.save_freq=50
```

### B2：rmpad + Seq-MIS（sequence IS + rejection sampling）

`RolloutCorrectionConfig.decoupled_seq_is_rs()` 的等价 YAML：
`rollout_is=sequence`、`rollout_rs=seq_sum_k1`、RS 阈值 `0.5_2.0`、IS 阈值 2.0。
decoupled 模式（默认 `bypass_mode=false`）会用 π_old 重算 old_log_prob，三个策略。

```bash
EXP=20261114-abl-rmpad-seqmis GPUS=2,3 VAL=sub BG=1 \
bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    algorithm.rollout_correction.rollout_is=sequence \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_rs=seq_sum_k1 \
    algorithm.rollout_correction.rollout_rs_threshold=0.5_2.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    trainer.total_training_steps=200 trainer.test_freq=25 trainer.save_freq=50
```

### B3：rmpad + score centering —— 当前多轮 recipe 跑不起来

**结论：不要按 B3 原样启动。** V1 sync trainer 本身支持 score centering（有完整的
bypass/REINFORCE/score-centering 路径），但它要求 `rollout.topk_log_probs>0`，而这个版本
只在**单轮** agent loop 下允许记录 top-k head：

```python
# verl/experimental/agent_loop/agent_loop.py:169
if self.num_turns > 2:
    raise ValueError("rollout.topk_log_probs supports the single-turn agent loop only.")
```

`tool_agent` 的 `num_turns = user_turns + assistant_turns + 1`
（`tool_agent_loop.py:205`），只要模型调用过一次工具就 >2，会在第一条这样的轨迹做
postprocess 时直接抛 `ValueError`，训练起不来。也就是说：**score centering 与
Search-R1 的多轮 tool agent 不兼容**，不是配置能绕过去的。

另外 `score_centering` 的配置约束是：必须 `bypass_mode=true` + `loss_type=reinforce` +
`rollout_is` 为 null 或 token + `topk_log_probs>0`，且 V1 必须在
`actor_rollout_ref.actor.policy_loss.rollout_correction` 下镜像 algorithm 侧的值
（两侧不一致时 `validate_config` 直接报错；V1 也不会像 V0 那样自动把 `loss_mode` 改成
`bypass_mode`）。

**替代方案 B3'（推荐）**：保留这套修正栈里可运行的部分——bypass + REINFORCE + token TIS，
去掉 score centering（也就不需要 topk head）。它和 score-centering preset 是同一个
bypass/REINFORCE 系列，只是少一个 centering 项。注意 bypass 模式下 actor 侧必须显式镜像
algorithm 的 `rollout_correction`，否则 actor 会用 dataclass 默认值
（`rollout_is=sequence`、`loss_type=ppo_clip`）——这是会静默发生的，不会报错。

```bash
EXP=20261114-abl-rmpad-bypass-rf GPUS=0,1 VAL=sub BG=1 \
bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.bypass_mode=True \
    algorithm.rollout_correction.loss_type=reinforce \
    actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=True \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=reinforce \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=token \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=2.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    trainer.total_training_steps=200 trainer.test_freq=25 trainer.save_freq=50
```

如果一定要 score centering：只能把 agent loop 换成 `single_turn_agent`（丢掉检索，不是这个
任务），或者改代码给多轮 top-k head 做逐轮对齐（不是最小改动，不建议在消融里做）。

### C1：batch 64（完全 on-policy）

`ppo_mini_batch_size=64` = 整个 `train_batch_size`（乘 `rollout.n` 后），也就是每个 batch
只做一次更新。学习率保持 1e-6 不变。**步数必须 ×8**，让 samples seen 和 A0 对齐：

- 200 → 1600 步（512×200 = 64×1600 = 102400 prompts）
- test_freq 25 → 200（每 12800 prompts 验证一次）
- save_freq 50 → 400（每 25600 prompts 存一次）

```bash
EXP=20261114-abl-batch64 GPUS=2,3 VAL=sub BG=1 \
bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    algorithm.rollout_correction.rollout_is=null \
    data.train_batch_size=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    trainer.total_training_steps=1600 trainer.test_freq=200 trainer.save_freq=400
```

注意：用 step 做 x 轴会得出完全错误的结论；一律用 samples seen。步数多 8 倍也意味着
wall-clock 更长（每步的 fixed overhead 更多），排期时留够时间。

### D1：GSPO

GSPO 的推荐配置来自 `examples/gspo_trainer/`：
`policy_loss.loss_mode=gspo`、`loss_agg_mode=seq-mean-token-mean`、
`clip_ratio_low/high=3e-4/4e-4`（`clip_ratio_c=10.0` 对 GSPO 是 no-op，选项里带着无害）。

```bash
EXP=20261114-abl-gspo GPUS=2,3 VAL=sub BG=1 \
bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    algorithm.rollout_correction.rollout_is=null \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    trainer.total_training_steps=200 trainer.test_freq=25 trainer.save_freq=50
```

D1 保持基线的 `use_kl_loss=True kl_loss_coef=0.001`（作为单变量对比）；GSPO 原 recipe 是
`use_kl_loss=False`，如果要贴论文，再加这个 override，但那就不是“A0 + GSPO”的干净对比了。

## 配对与执行顺序

每个 slot 对（`GPUS=0,1` / `GPUS=2,3`）同时跑两个。GPU0 上有检索服务，两个 slot 显存条件
不同，只影响速度/OOM 风险，不影响优化本身的数值；但同一个 pair 的两个实验尽量固定 slot，
不要中途换。

1. **A0 + A1**：先跑。两者都是非 rmpad、显存最轻，也给出“基线与 DAPO”这一对主结论。
2. **B1 + B2**：rmpad 下的 token IS vs sequence IS+RS，直接对比修正粒度。
3. **B3' + C1**：bypass/REINFORCE 修正栈与 batch 组成；两者互不依赖，放在一起跑。
4. **B0 + D1**：B0 复现 mismatch collapse（也可复用已有的 collapse run）；D1 是 GSPO。

如果某个 job 提前结束（尤其是 B0 的 collapse），空出的 slot 可以提前启动下一个实验；
`EXP` 目录是唯一的，两个 job 的实验目录/checkpoint/Ray 临时目录互不干扰。

## 显存估算（0.6 保持不变）

B300 单卡 268GB。`GPUS=0,1` 的 job 在 GPU0 上要和检索服务（~35GB）共享：

- rollout 阶段：vLLM 预算 0.6×268≈161GB + 检索 35GB ≈ 196GB，空闲余量还有 ~70GB；
- 训练阶段（A0/A1/C1/D1）：当前设置（不开 rmpad、12288 token/GPU）实测 FSDP 峰值 ~107GB，
  + 检索 35GB ≈ 142GB；
- 训练阶段（B 组）：rmpad 省 activation，但 `ppo_max_token_len_per_gpu` 恢复到 24576。
  不开 rmpad 的 24576 峰值约 219GB（README 记录），rmpad 下会显著低于这个数；具体值没有在
  2 卡 + 检索的环境下量过，**这是唯一需要第一次跑时盯 `nvidia-smi` 的地方**。

按上面数字 0.6 不用改；如果 B 组在 GPU0/1 上 OOM，优先降 `ppo_max_token_len_per_gpu`
（B 组内部保持同一个值），其次才是 `gpu_memory_utilization`（那会改变所有实验的 rollout 预算）。

## V1 sync 路径的已知限制（写结论时别踩）

- `algorithm.filter_groups.max_num_gen_batches` 被忽略；用 `max_inflight_gen_batches`。
- 动态采样 metric 用 `acc`；`score` 不可用（我们的 reward 返回 float，只有 `acc` 这个
  `reward_extra_info` key）。
- overlong 只有奖励惩罚，没有“从 loss 中 mask 截断样本”的开关。
- **score centering 不能和多轮 tool agent 一起用**：它需要 `topk_log_probs>0`，而
  `AgentLoopOutput.as_dict` 对 `num_turns>2` 直接 raise（`agent_loop.py:169`），
  `tool_agent` 只要调用一次工具就超过 2 轮。B3 因此换成 B3'。
- GSPO 与 `rollout_correction.bypass_mode=true` 不兼容（bypass loss 只支持
  `ppo_clip`/`reinforce`）；D1 没开 bypass，不冲突。
- bypass 模式（B3'）下 actor 侧 `policy_loss.rollout_correction` 必须显式镜像 algorithm 侧，
  否则 actor 用 dataclass 默认值（sequence + ppo_clip），不报错但跑的不是你以为的配置。
- 所有 `+` 前缀的 override 是新增 key（`reward.reward_kwargs.*`、
  `actor_rollout_ref.actor.policy_loss.rollout_correction.*`），漏掉 `+` Hydra 会报
  `Key ... is not in struct`。

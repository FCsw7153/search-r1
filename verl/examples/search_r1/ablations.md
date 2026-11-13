# Search-R1 GRPO 消融实验（Qwen3.5-2B，4×B300）

本目录的消融实验都基于 `run_qwen3_5_2b_search_r1_grpo.sh` 的默认配置，每个 job 用 2 张卡，
同节点同时跑两个：`GPUS=0,1` 和 `GPUS=2,3`。GPU0 还跑着检索服务（~35GB），所以 `GPUS=0,1`
的 job 实际可用显存少一些；`gpu_memory_utilization` 保持 0.6 不变（见文末的显存估算）。

## 约定

- 命令都在 `$SEARCH_R1_HOME` 下执行，先 `source "$SEARCH_R1_HOME/.venv/bin/activate"`。
- 除 C1/C2 外，所有实验共享：

  ```bash
  trainer.total_training_steps=200 trainer.test_freq=25 trainer.save_freq=50
  VAL=sub          # $SEARCH_R1_HOME/data/search_r1/test_sub.parquet，每 25 步验证
  ```

  C1/C2 的步数按各自 batch size 换算，见下。`VAL=full` 会换成完整 7 数据集 `test.parquet`。
- 两组 GPU 用 `GPUS=0,1` / `GPUS=2,3`，launcher 会设置 `CUDA_VISIBLE_DEVICES` 和
  `trainer.n_gpus_per_node=2`；`EXP` 决定实验目录、checkpoint、Ray 的 `_temp_dir` 和权重同步
  IPC socket 的 token。并行的隔离细节见 README。
- 每步“看过的样本数”= `step × data.train_batch_size × actor_rollout_ref.rollout.n`：
  A0/A1/B1/B2 是 512×5=2560/步，C2 是 128×5=640/步，C1 是 32×5=160/步。
  `meta.json` 记录了生效后的 `train_batch_size` 和 `rollout_n`，dashboard 可以直接算；
  **A1 开了动态采样，实际生成的样本比这个多**，横向比较时以 x 轴的 samples seen 为准，
  不要用 step。
- 停一个 job 用它的 pid 文件杀进程组（`kill -- -$(cat .../runs/<TS>.pid)`），
  **不要用 `ray stop --force`**，那会把同节点另一个 job 的 Ray 一起杀掉。详见 README。

## 排期（3 轮，每轮两个 job）

| 轮次 | `GPUS=0,1` | `GPUS=2,3` | 对比内容 | 预计耗时 |
|---|---|---|---|---|
| 1 | A0 baseline-noTIS | A1 DAPO | 基线 vs DAPO 三件套 | 待第一轮实测后更新 |
| 2 | B1 rmpad + token IS | B2 rmpad + sequence IS + RS | IS 修正粒度 | 待第一轮实测后更新 |
| 3 | C2 batch 128 | C1 batch 32（先跑到 1600 步） | batch 组成 / on-policy 程度 | 待第一轮实测后更新 |

时间估算先留空：第一轮跑完后用 A0/A1 的实际 wall-clock 填表（C1 计划跑 1600 步，是这一轮
里最长的；C2 只有 800 步但每步更重）。GPU0 上有检索服务，两个 slot 的显存条件不同，只影响
速度/OOM 风险，不影响优化本身的数值；同一个 pair 尽量固定 slot，不要中途换。

## 实验总表

| ID | EXP | 变量 | 相对 A0 的改动 | 训练步数 |
|---|---|---|---|---|
| A0 | `20261114-abl-baseline-noTIS` | 关闭 TIS 的基线 | launcher 默认 + `rollout_is=null` | 200 |
| A1 | `20261114-abl-dapo` | DAPO 三件套 | A0 + clip-higher + 动态采样 + overlong 奖励惩罚 | 200 |
| B1 | `20261114-abl-rmpad-tis` | token 级 IS | A0 + `use_remove_padding=True` + `ppo_max_token_len_per_gpu=24576` + `rollout_is=token` | 200 |
| B2 | `20261114-abl-rmpad-seqmis` | sequence IS + RS | 同 B1，但 `rollout_is=sequence` + `rollout_rs=seq_sum_k1` | 200 |
| C2 | `20261114-abl-c2-bs128` | batch 128 | A0 + `train_batch_size=128 ppo_mini_batch_size=128` | 800 |
| C1 | `20261114-abl-c1-bs32` | batch 32（小 batch） | A0 + `train_batch_size=32 ppo_mini_batch_size=32`，先跑到 1600 | 3200（计划停在 1600） |

B 组的“不修正”参照不用重跑：已有的 collapse run `20261113-rmpad-on-collapsed` 就是
rmpad + 不修正的对照组（注意它当时用了 **4 卡和完整验证集**，和这里 2 卡 + `VAL=sub` 的训练
数值可比、吞吐/验证口径不完全一致）。

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

### B1：rmpad + token 级 IS

`use_remove_padding=True` 是已知的 mismatch 来源（Qwen3.5 GatedDeltaNet 下
`rollout_corr/kl≈0.05`，约 60 步 collapse）。rmpad 省显存，因此把
`ppo_max_token_len_per_gpu` 恢复到 24576。

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

### C2：batch 128

`ppo_mini_batch_size=128` = 整个 `train_batch_size`（乘 `rollout.n` 后），每个 batch 只做一次
更新。步数按 samples seen 对齐 A0：`128×800 = 512×200 = 102400` prompts。

- 200 → 800 步
- test_freq 25 → 100（每 12800 prompts 验证一次）
- save_freq 50 → 200（每 25600 prompts 存一次）

```bash
EXP=20261114-abl-c2-bs128 GPUS=0,1 VAL=sub BG=1 \
bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    algorithm.rollout_correction.rollout_is=null \
    data.train_batch_size=128 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    trainer.total_training_steps=800 trainer.test_freq=100 trainer.save_freq=200
```

### C1：batch 32（先跑到 1600 步，按需续到 3200）

`ppo_mini_batch_size=32` = 整个 `train_batch_size`，每个 batch 一次更新。完整长度按 samples
seen 对齐 A0：`32×3200 = 512×200 = 102400` prompts；**计划先跑到 1600 步**（A0 的一半样本，
`32×1600 = 51200` prompts），看完结果再决定要不要 `RESUME=1` 续到 3200。

- 200 → **3200 步**（先跑到 1600）
- test_freq 25 → 400（每 12800 prompts 验证一次；1600 之前有 400/800/1200/1600 四个点）
- save_freq 50 → 800（每 25600 prompts 存一次；800 和 1600 各有一个 checkpoint）

**为什么 total_training_steps 保持 3200、而不是直接设 1600**：`optim.lr_warmup_steps_ratio=0.285`
的 warmup 长度是 `0.285 × total_training_steps`。设 3200 时 warmup 是 912 步 ×32 =
29184 prompts，正好等于 A0 的 `0.285×200×512`；如果设 1600，warmup 只有一半样本，
LR 曲线在 samples seen 的 x 轴上就和 A0 对不齐了。所以保持 3200（LR 曲线按样本对齐），
用“跑到 1600 手动停”控制计算量。

**首次启动**：

```bash
EXP=20261114-abl-c1-bs32 GPUS=2,3 VAL=sub BG=1 \
bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    algorithm.rollout_correction.rollout_is=null \
    data.train_batch_size=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    trainer.total_training_steps=3200 trainer.test_freq=400 trainer.save_freq=800
```

**停止方式**（务必等 step 1600 的 checkpoint 写完再停，否则续训只能从 800 恢复）：

```bash
kill -- -$(cat $SEARCH_R1_HOME/experiments/20261114-abl-c1-bs32/runs/<TS>.pid)
```

**续训**：如果决定继续，用同样的 override 加 `RESUME=1`：

```bash
EXP=20261114-abl-c1-bs32 RESUME=1 GPUS=2,3 VAL=sub BG=1 \
bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    algorithm.rollout_correction.rollout_is=null \
    data.train_batch_size=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    trainer.total_training_steps=3200 trainer.test_freq=400 trainer.save_freq=800
```

**batch 消融的固有 confound**：学习率保持 1e-6 不变，但 batch 越小，每个样本对应的优化器
更新次数越多（A0：200 步 × 2 次更新 = 400 次；C2：800 次；C1 到 1600 步时已经 1600 次）。
所以 C1/C2 与 A0 的差异同时包含“batch 大小/on-policy 程度”和“更新次数”，结论要按这个口径写，
不要只说成 batch size 的因果效应。

## 显存估算（0.6 保持不变）

B300 单卡 268GB。`GPUS=0,1` 的 job 在 GPU0 上要和检索服务（~35GB）共享：

- rollout 阶段：vLLM 预算 0.6×268≈161GB + 检索 35GB ≈ 196GB，空闲余量还有 ~70GB；
- 训练阶段（A0/A1/C1/C2）：当前设置（不开 rmpad、12288 token/GPU）实测 FSDP 峰值 ~107GB，
  + 检索 35GB ≈ 142GB；batch size 变化不改变 per-GPU token 上限，峰值基本不变；
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
  `tool_agent` 只要调用一次工具就超过 2 轮。因此 B3 未排期（见附录），可运行的 B3' 也未排期。
- GSPO 与 `rollout_correction.bypass_mode=true` 不兼容（bypass loss 只支持
  `ppo_clip`/`reinforce`）；附录里的 D1 没开 bypass，不冲突。
- 所有 `+` 前缀的 override 是新增 key（`reward.reward_kwargs.*`、
  `actor_rollout_ref.actor.policy_loss.rollout_correction.*`），漏掉 `+` Hydra 会报
  `Key ... is not in struct`。

## 未安排的实验（保留配置）

### B0：rmpad 不修正

**不重跑**：`20261113-rmpad-on-collapsed` 就是 rmpad + 不修正的 collapse 参照，
`rollout_corr/kl≈0.05`、约 60 步崩。注意它用的是 4 卡和完整验证集（`VAL=full`），
和本次 2 卡 + `VAL=sub` 的口径不同。保留配置以便以后复现：

```bash
EXP=20261114-abl-rmpad-nocorr GPUS=0,1 VAL=sub BG=1 \
bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh \
    algorithm.rollout_correction.rollout_is=null \
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

score centering 的配置约束（供以后版本参考）：必须 `bypass_mode=true` + `loss_type=reinforce` +
`rollout_is` 为 null 或 token + `topk_log_probs>0`，且 V1 必须在
`actor_rollout_ref.actor.policy_loss.rollout_correction` 下镜像 algorithm 侧的值
（两侧不一致时 `validate_config` 直接报错；V1 也不会像 V0 那样自动把 `loss_mode` 改成
`bypass_mode`）。

如果一定要 score centering：只能把 agent loop 换成 `single_turn_agent`（丢掉检索，不是这个
任务），或者改代码给多轮 top-k head 做逐轮对齐（不是最小改动，不建议在消融里做）。

### B3'：rmpad + bypass + REINFORCE + token TIS（无 score centering）

score-centering 系列里可运行的部分，未排期。注意 bypass 模式下 actor 侧必须显式镜像
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

### D1：GSPO

未排期。推荐配置来自 `examples/gspo_trainer/`：
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

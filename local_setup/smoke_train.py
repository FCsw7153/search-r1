"""One-step PPO smoke test of the repository's Ray/FSDP/HF training path."""
import ray
from hydra import compose, initialize_config_dir
from pathlib import Path
from verl.trainer.main_ppo import main
root = Path(__file__).resolve().parents[1]
with initialize_config_dir(config_dir=str(root / 'verl/trainer/config'), version_base=None):
    cfg = compose(config_name='ppo_trainer', overrides=[
        'data.train_files=data/nq_hotpotqa_train/smoke_train.parquet',
        'data.val_files=data/nq_hotpotqa_train/smoke_test.parquet',
        'data.train_batch_size=2','data.val_batch_size=2',
        'data.max_prompt_length=512','data.max_start_length=384',
        'data.max_response_length=32','data.max_obs_length=64',
        'actor_rollout_ref.model.path=data/models/Qwen2.5-0.5B-Instruct',
        'critic.model.path=data/models/Qwen2.5-0.5B-Instruct',
        'actor_rollout_ref.rollout.name=hf',
        'actor_rollout_ref.rollout.tensor_model_parallel_size=1',
        'actor_rollout_ref.actor.ppo_mini_batch_size=2',
        'actor_rollout_ref.actor.ppo_micro_batch_size=1',
        'actor_rollout_ref.rollout.log_prob_micro_batch_size=1',
        'actor_rollout_ref.ref.log_prob_micro_batch_size=1',
        'critic.ppo_mini_batch_size=2','critic.ppo_micro_batch_size=1',
        'critic.forward_micro_batch_size=1',
        'trainer.n_gpus_per_node=1','trainer.nnodes=1',
        'trainer.total_training_steps=1','trainer.total_epochs=1',
        'trainer.project_name=search_r1_smoke','trainer.experiment_name=b300_hf_ppo','trainer.logger=[console]','trainer.save_freq=-1','trainer.test_freq=-1',
        'trainer.default_hdfs_dir=null','+trainer.val_before_train=false',
        'max_turns=1', 'retriever.url=http://127.0.0.1:18080/retrieve', 'retriever.topk=3',
    ])
ray.init(num_cpus=4, num_gpus=1, object_store_memory=268435456,
         include_dashboard=False, runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2'}})
try:
    main(cfg)
finally:
    ray.shutdown()

"""External V1 sync recipe. Upstream verl files remain unchanged."""
import os
from pathlib import Path
import hydra
import ray
import torch
from omegaconf import OmegaConf
import transfer_queue as tq
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.v1 import PPOTrainerSync, AgentLoopManagerTQ
from verl.utils.config import validate_config
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.device import auto_set_device
from verl.utils.logging_utils import configure_verl_logging
from experiment_metrics import pass_metrics


class SearchTrainer(PPOTrainerSync):
    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict,
                            sample_turns, expected_acc_counts=None):
        metrics = super()._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict,
                                              sample_turns, expected_acc_counts)
        metrics.update(pass_metrics(data_sources, sample_uids, reward_extra_infos_dict['acc'],
                                    expected_acc_counts,
                                    ks=tuple(sorted({1, int(self.config.actor_rollout_ref.rollout.val_kwargs.n)}))))
        return metrics

    def _compute_ref_log_prob(self, batch, metrics):
        batch = super()._compute_ref_log_prob(batch, metrics)
        fields = ['old_log_probs', 'rollout_log_probs', 'ref_log_prob', 'response_mask']
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        # Jagged tensors all share response offsets after the upstream conversion.
        mask = data['response_mask'].values().bool()
        ref = data['ref_log_prob'].values().double()[mask]
        sampler = data['rollout_log_probs'].values().double()[mask]
        trainer = data['old_log_probs'].values().double()[mask]
        # Samples come from sampler. Importance weighting estimates trainer || reference.
        weight = torch.exp(trainer-sampler)
        for name, logp, w in (('sampler', sampler, torch.ones_like(sampler)),
                             ('trainer', trainer, weight)):
            delta = logp-ref
            metrics[f'{name}/ref_kl_k1'] = (w*delta).mean().item()
            metrics[f'{name}/ref_kl_k3'] = (w*(torch.exp(-delta)-1+delta)).mean().item()
        metrics['trainer/ref_kl_importance_weight_mean'] = weight.mean().item()
        return batch

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        if 'actor/entropy' in metrics:
            metrics['trainer/entropy'] = metrics['actor/entropy']


@ray.remote(num_cpus=1)
class SearchTaskRunner:
    def run(self, config):
        configure_verl_logging()
        torch.set_num_threads(int(os.environ.get('TRAIN_CPU_THREADS', '4')))
        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        trainer, succeeded = None, False
        try:
            trainer = SearchTrainer(config=config)
            trainer.init()
            manager = AgentLoopManagerTQ.create(
                config=config, llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles())
            trainer.fit(manager)
            succeeded = True
        finally:
            try:
                if trainer is not None:
                    trainer.close_dataloaders()
            finally:
                try:
                    tracking = getattr(trainer, 'logger', None)
                    if tracking is not None:
                        tracking.finish(exit_code=0 if succeeded else 1)
                finally:
                    tq.close()


@hydra.main(config_path=str(Path(__file__).resolve().parents[1]/'verl/verl/trainer/config'),
            config_name='ppo_trainer', version_base=None)
def main(config):
    auto_set_device(config)
    if not config.trainer.use_v1 or config.trainer.v1.trainer_mode != 'sync':
        raise ValueError('This recipe requires the synchronous V1 trainer')
    if config.actor_rollout_ref.rollout.val_kwargs.n < 2:
        raise ValueError('Use at least 2 validation samples for pass@k')
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    run_ppo(config, task_runner_class=SearchTaskRunner)


if __name__ == '__main__':
    main()

"""CPU-only configuration and extension checks; never starts Ray or loads model weights."""
import asyncio
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from train_grpo import SearchTrainer
from paper_agent import PaperToolAgent, ToolAgentLoop, AgentState
from paper_reward import compute_score
from verl.trainer.ppo.v1 import PPOTrainerSync
from verl.utils.config import validate_config
from verl.trainer.ppo.utils import need_reference_policy, need_critic
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
configs = list((ROOT/'logs').glob('qwen35-2B-search-grpo-paper/*/config.yaml'))
config = OmegaConf.load(max(configs, key=lambda p:p.stat().st_mtime))
validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
assert config.trainer.n_gpus_per_node == 4 and config.trainer.total_training_steps == 500
assert config.actor_rollout_ref.rollout.n == 5
assert config.actor_rollout_ref.rollout.calculate_log_probs
assert config.actor_rollout_ref.actor.calculate_entropy
assert compute_score('nq', '<answer>Paris</answer>', {'target':['Paris']}) == {'score':1.0,'acc':1.0}
assert compute_score('nq', 'Paris', {'target':['Paris']})['score'] == 0

# Exercise actual nested tensor layout used by TransferQueue, with masked tool tokens.
def nested(values):
    return torch.nested.as_nested_tensor([torch.tensor(values)], layout=torch.jagged)
data = {'old_log_probs':nested([math.log(.5), -999]),
        'rollout_log_probs':nested([math.log(.5), -999]),
        'ref_log_prob':nested([math.log(.25), -1]), 'response_mask':nested([1,0])}
trainer = object.__new__(SearchTrainer)
metrics = {}
batch = SimpleNamespace(keys=['sample'], partition_id='train')
with patch.object(PPOTrainerSync, '_compute_ref_log_prob', return_value=batch), \
     patch('train_grpo.tq.kv_batch_get', return_value=data):
    trainer._compute_ref_log_prob(batch, metrics)
assert abs(metrics['sampler/ref_kl_k1']-math.log(2)) < 1e-6
assert abs(metrics['trainer/ref_kl_k3']-(math.log(2)-.5)) < 1e-6

async def check_agent():
    agent = object.__new__(PaperToolAgent)
    agent.rollout_config = SimpleNamespace(max_model_len=4096)
    agent.response_length = 4096
    state = SimpleNamespace(prompt_ids=[1]*4000, response_mask=[1]*100, response_logprobs=[-.5]*100)
    async def generate(_, value, params, ignore):
        assert params['max_tokens'] == 96
        return AgentState.PROCESSING_TOOLS
    with patch.object(ToolAgentLoop, '_handle_generating_state', generate):
        assert await agent._handle_generating_state(state, {}) == AgentState.PROCESSING_TOOLS
    async def tool(_, value):
        value.prompt_ids += [2]*200
        value.response_mask += [0]*200
        value.response_logprobs += [0.0]*200
        return AgentState.GENERATING
    with patch.object(ToolAgentLoop, '_handle_processing_tools_state', tool):
        assert await agent._handle_processing_tools_state(state) == AgentState.TERMINATED
    assert len(state.prompt_ids)==4096 and len(state.response_mask)==196
    assert len(state.response_logprobs)==196
asyncio.run(check_agent())
for split, expected in [('train',169615),('test',51713)]:
    path = ROOT/f'data/grpo_qwen35/{split}.parquet'
    assert pq.read_metadata(path).num_rows == expected
    table = pq.read_table(path, columns=['data_source'])
    print(split, expected, sorted(set(table.column(0).to_pylist())))
print('PASS: four-GPU config, binary reward, masked nested KL, context limits, full datasets. No GPU training launched.')

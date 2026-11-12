"""Native Qwen tools, paper limits: 500 generated tokens/action, 4096 total tokens."""
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop, AgentState


class PaperToolAgent(ToolAgentLoop):
    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        remaining = self.rollout_config.max_model_len - len(agent_data.prompt_ids)
        if remaining <= 0:
            return AgentState.TERMINATED
        params = dict(sampling_params)
        params['max_tokens'] = min(500, remaining, self.response_length-len(agent_data.response_mask))
        if params['max_tokens'] <= 0:
            return AgentState.TERMINATED
        state = await super()._handle_generating_state(agent_data, params, ignore_termination)
        if len(agent_data.prompt_ids) >= self.rollout_config.max_model_len:
            return AgentState.TERMINATED
        return state

    async def _handle_processing_tools_state(self, agent_data):
        state = await super()._handle_processing_tools_state(agent_data)
        # Only truncate newly appended tool/context tokens; keep token/mask/logprob alignment.
        overflow = len(agent_data.prompt_ids) - self.rollout_config.max_model_len
        if overflow > 0:
            agent_data.prompt_ids = agent_data.prompt_ids[:-overflow]
            agent_data.response_mask = agent_data.response_mask[:-overflow]
            if agent_data.response_logprobs:
                agent_data.response_logprobs = agent_data.response_logprobs[:-overflow]
        if overflow >= 0:
            return AgentState.TERMINATED
        return state

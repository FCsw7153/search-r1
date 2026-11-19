# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Search-R1 tag protocol agent loop, for base models without chat ability.

Mirrors ``search_r1/llm_agent/generation.py`` from PeterGriffinJin/Search-R1:
the prompt is the original plain-text instruction (no chat template), the model
reasons inside ``<think>...</think>``, searches with ``<search> query </search>``,
reads results inside ``<information>...</information>``, and finishes with
``<answer> ... </answer>``. Retrieved passages are appended to the response with
``response_mask = 0`` so they are attended to but excluded from the loss.
"""

import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    ToolListWrap,
    register,
)
from verl.experimental.agent_loop.tool_agent_loop import SPEC_DECODE_EXTRA_KEYS
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Verbatim from Search-R1's scripts/data_process/nq_search.py (make_prefix, template_type="base").
# The upstream source uses backslash line continuations, so the template is one flat line.
SEARCH_R1_BASE_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
    "Question: {question}\n"
)

SEARCH_R1_STOP_STRINGS = ("</search>", "</answer>")
# Why a trajectory stopped; reported per sample and aggregated as one 0/1 metric each.
SEARCH_R1_END_REASONS = ("answer", "turn_cap", "length", "no_action")
_SEARCH_ACTION_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)


def build_search_r1_prompt(question: str) -> str:
    """Render the original Search-R1 instruction for one question.

    The upstream preprocessing also appends ``?`` when the question lacks one.
    """
    question = question.strip()
    if question and not question.endswith("?"):
        question += "?"
    return SEARCH_R1_BASE_TEMPLATE.format(question=question)


def extract_question(messages: list[dict[str, Any]]) -> str:
    """Return the text of the last user message (the agent loop replaces the dataset's chat prompt)."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            texts = [
                part["text"]
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip()
            ]
            if texts:
                return " ".join(texts)
    raise ValueError("search_r1_agent expects a user message with non-empty text content")


def parse_search_r1_action(text: str) -> tuple[Optional[str], Optional[str]]:
    """Map one assistant turn to ``(action, query)``.

    ``</search>`` takes precedence over ``</answer>`` (same order as upstream), and the last
    complete ``<search>...</search>`` block supplies the query. An empty query is still a search
    turn (the tool rejects it with a text observation). A closing search tag without a complete
    block, or no tag at all, terminates the trajectory.
    """
    if "</search>" in text:
        matches = _SEARCH_ACTION_PATTERN.findall(text)
        if not matches:
            return None, None
        return "search", matches[-1].strip()
    if "</answer>" in text:
        return "answer", None
    return None, None


def extract_gold_answers(reward_model: Any) -> list[str]:
    """Return the non-empty gold answer aliases from a dataset row's ``reward_model``.

    The Search-R1 parquet stores ``reward_model = {"style": "rule", "ground_truth":
    {"target": [...]}}``. The agent loop receives the same dict through ``kwargs``; returning an
    empty list means the aliases are unavailable and the hit diagnostics stay at their defaults.
    """
    if not isinstance(reward_model, dict):
        return []
    ground_truth = reward_model.get("ground_truth")
    if isinstance(ground_truth, dict):
        targets = ground_truth.get("target")
    else:
        targets = ground_truth
    if isinstance(targets, str):
        targets = [targets]
    if not isinstance(targets, (list, tuple)):
        return []
    return [str(target) for target in targets if target is not None and str(target).strip()]


def response_contains_gold_alias(text: str, gold_answers: list[str]) -> bool:
    """Paper-style evidence-hit test: ``lower(strip(alias)) in lower(response)``.

    This is the ``m(a, o)`` containment test from GCA (arXiv 2610.23148, Eq. 1). It is
    deliberately a raw case-insensitive substring test rather than the EM normalizer: aliases
    are already short answer strings, and normalization would make short aliases over-match.
    """
    lowered = text.lower()
    return any(alias.strip().lower() in lowered for alias in gold_answers if alias.strip())


def truncate_tool_response(text: str, max_length: Optional[int], truncate_side: str) -> str:
    """Truncate a tool response by characters, matching ToolAgentLoop's behaviour."""
    if not text or not max_length or len(text) <= max_length:
        return text
    if truncate_side == "left":
        return "(truncated)..." + text[-max_length:]
    if truncate_side == "right":
        return text[:max_length] + "...(truncated)"
    half = max_length // 2
    return text[:half] + "...(truncated)..." + text[-half:]


@register("search_r1_agent")
class SearchR1AgentLoop(AgentLoopBase):
    """Multi-turn Search-R1 loop that uses text tags instead of native tool calls.

    Works for base models: the prompt is built directly from the user question with
    :func:`build_search_r1_prompt`, bypassing the chat template. The search tool is looked
    up by name (``search``) from ``multi_turn.tool_config_path``.
    """

    def __init__(self, *args, tools: Optional[ToolListWrap] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        tool_list = tools.tools if tools else []
        self.search_tool = next((tool for tool in tool_list if tool.name == "search"), None)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        if self.search_tool is None:
            raise ValueError(
                "search_r1_agent requires a tool named 'search'; set "
                "actor_rollout_ref.rollout.multi_turn.tool_config_path to a config that defines it."
            )
        # priority may arrive as np.int64 from non_tensor_batch; normalize to Python int.
        priority = int(priority)

        question = extract_question(list(kwargs["raw_prompt"]))
        gold_answers = extract_gold_answers(kwargs.get("reward_model"))
        prompt_ids = await self.loop.run_in_executor(None, self._build_prompt_ids, question)
        prompt_ids = self._cap_text_prompt_length(prompt_ids)

        request_id = f"det-{priority}" if self.rollout_config.full_determinism else uuid4().hex
        want_logprobs = bool(sampling_params.get("logprobs"))
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: Optional[list[float]] = [] if want_logprobs else None
        routed_experts = None
        extra_fields: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        assistant_turns = 0
        user_turns = 0
        search_queries: list[str] = []
        hit = 0
        first_hit_turn = -1
        end_reason = "no_action"

        while True:
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                end_reason = "turn_cap"
                break
            remaining = self.response_length - len(response_ids)
            if remaining <= 0:
                end_reason = "length"
                break

            # vLLM includes the matched stop string in the returned token ids with
            # include_stop_str_in_output, so </search> / </answer> stay in the response
            # (Search-R1 keeps </search>). max_tokens keeps this turn inside the total budget.
            turn_sampling_params = {
                **sampling_params,
                "stop": list(SEARCH_R1_STOP_STRINGS),
                "include_stop_str_in_output": True,
                "max_tokens": remaining,
            }
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids + response_ids,
                    sampling_params=turn_sampling_params,
                    priority=priority,
                )
            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            else:
                metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

            self._merge_extra_fields(extra_fields, output.extra_fields)
            if output.routed_experts is not None:
                routed_experts = output.routed_experts

            assistant_turns += 1
            turn_ids = list(output.token_ids[:remaining])
            if not turn_ids:
                end_reason = "no_action"
                break
            response_ids.extend(turn_ids)
            response_mask.extend([1] * len(turn_ids))
            if response_logprobs is not None:
                turn_logprobs = list(output.log_probs or [])
                turn_logprobs.extend([0.0] * (len(turn_ids) - len(turn_logprobs)))
                response_logprobs.extend(turn_logprobs[: len(turn_ids)])

            turn_text = await self.loop.run_in_executor(
                None, lambda ids=turn_ids: self.tokenizer.decode(ids, skip_special_tokens=True)
            )
            action, query = parse_search_r1_action(turn_text)
            if action != "search":
                end_reason = "answer" if action == "answer" else "no_action"
                break

            # Do not search after the last allowed generation or with no response budget left:
            # the loop ends on the next iteration anyway, so an appended information block would
            # only add masked tokens. This also keeps num_turns <= 2 * max_assistant_turns.
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                end_reason = "turn_cap"
                break
            if len(response_ids) >= self.response_length:
                end_reason = "length"
                break

            with simple_timer("tool_calls", metrics):
                documents = await self._search(query)
            search_queries.append(query)
            # Record the first search turn whose (truncated) retrieved text contains a gold alias.
            # This is GCA's H_i / k_i^* (arXiv 2610.23148, Eq. 1) and is also a plain-GRPO diagnostic.
            if not hit and gold_answers and response_contains_gold_alias(documents, gold_answers):
                hit = 1
                first_hit_turn = len(search_queries)
            observation = f"\n\n<information>{documents}</information>\n\n"
            observation_ids = await self.loop.run_in_executor(
                None, lambda text=observation: self.tokenizer(text, add_special_tokens=False)["input_ids"]
            )
            observation_ids = list(observation_ids[: self.response_length - len(response_ids)])
            if not observation_ids:
                end_reason = "length" if len(response_ids) >= self.response_length else "no_action"
                break
            response_ids.extend(observation_ids)
            response_mask.extend([0] * len(observation_ids))
            if response_logprobs is not None:
                response_logprobs.extend([0.0] * len(observation_ids))
            user_turns += 1

        extra_fields.update(
            {
                "search_r1_end_reason": end_reason,
                "search_r1_num_searches": len(search_queries),
                "search_r1_queries": search_queries,
                "search_r1_hit": hit,
                "search_r1_first_hit_turn": first_hit_turn,
                "search_r1_has_answer": int(end_reason == "answer"),
            }
        )

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs is not None else None,
            routed_experts=(
                routed_experts[: len(prompt_ids) + self.response_length] if routed_experts is not None else None
            ),
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
            extra_fields=extra_fields,
        )
        # Keep the schema consistent with tool_agent / single_turn agent loops.
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return output

    def _build_prompt_ids(self, question: str) -> list[int]:
        """Tokenize the plain-text Search-R1 prompt (base models: no chat template, no special tokens)."""
        return self.tokenizer(build_search_r1_prompt(question), add_special_tokens=False)["input_ids"]

    async def _search(self, query: str) -> str:
        """Run the configured search tool and return its (truncated) text."""
        instance_id = None
        try:
            instance_id, _ = await self.search_tool.create()
            tool_response, _, _ = await self.search_tool.execute(instance_id, {"query": query})
            text = tool_response.text or ""
        except Exception as exc:
            # A retrieval failure should not lose the whole trajectory; show it as the observation.
            logger.warning("search_r1_agent: search failed for query %r: %s", query, exc)
            text = f"Search failed: {exc}"
        finally:
            if instance_id is not None:
                await self.search_tool.release(instance_id)
        return truncate_tool_response(text, self.max_tool_response_length, self.tool_response_truncate_side)

    @staticmethod
    def _merge_extra_fields(extra_fields: dict[str, Any], turn_extra_fields: dict[str, Any]) -> None:
        """Accumulate server extra fields across turns (same contract as ToolAgentLoop)."""
        if not extra_fields:
            extra_fields.update(turn_extra_fields)
            return
        # Multi-round calls, only update the maximum max_global_steps.
        max_global_steps = turn_extra_fields.get("max_global_steps", None)
        if max_global_steps:
            extra_fields["max_global_steps"] = max_global_steps
        for key in SPEC_DECODE_EXTRA_KEYS:
            if key in turn_extra_fields and key in extra_fields:
                extra_fields[key] = int(extra_fields[key]) + int(turn_extra_fields[key])

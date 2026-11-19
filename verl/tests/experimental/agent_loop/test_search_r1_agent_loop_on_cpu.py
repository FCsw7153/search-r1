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
"""CPU tests for the Search-R1 tag-protocol agent loop (no GPU, no retriever).

Set ``SEARCH_R1_TEST_TOKENIZER`` to a local tokenizer path/HF id to run the same tests with a
real tokenizer (e.g. ``Qwen/Qwen3-4B-Base``); otherwise a deterministic character tokenizer is used.
"""

import asyncio
import os
import unittest
from types import SimpleNamespace

from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
from verl.experimental.agent_loop.search_r1_agent_loop import (
    SEARCH_R1_BASE_TEMPLATE,
    SearchR1AgentLoop,
    build_search_r1_prompt,
    extract_question,
    parse_search_r1_action,
)
from verl.tools.schemas import ToolResponse
from verl.utils.reward_score import search_r1_like_qa_em

EXPECTED_PROMPT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new "
    "information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query "
    "</search> and it will return the top searched results between <information> and </information>. You can search as "
    "many times as your want. If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: "
    "What is the capital of France?\n"
)


class CharTokenizer:
    """One token per character; deterministic and dependency-free."""

    def __init__(self):
        self._token_to_id = {}
        self._id_to_token = {}

    def __call__(self, text, add_special_tokens=True, **kwargs):
        assert add_special_tokens is False, "Search-R1 base prompts must not add special tokens"
        return {"input_ids": [self._token_id(char) for char in text]}

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self._id_to_token[token_id] for token_id in token_ids)

    def _token_id(self, char):
        if char not in self._token_to_id:
            token_id = len(self._token_to_id) + 1
            self._token_to_id[char] = token_id
            self._id_to_token[token_id] = char
        return self._token_to_id[char]


class FakeLLMServer:
    """Returns canned assistant turns and records the requests."""

    def __init__(self, tokenizer, turns, logprob=-0.5):
        self.tokenizer = tokenizer
        self.turns = list(turns)
        self.logprob = logprob
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, priority=0, **kwargs):
        self.calls.append({"prompt_ids": list(prompt_ids), "sampling_params": dict(sampling_params)})
        text = self.turns.pop(0)
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        log_probs = [self.logprob] * len(token_ids) if sampling_params.get("logprobs") else None
        return SimpleNamespace(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=None,
            num_preempted=0,
            stop_reason="completed",
            extra_fields={"global_steps": 3},
        )


class FakeSearchTool:
    name = "search"

    def __init__(self, text="Doc 1(Title: Paris) Paris is the capital of France."):
        self.text = text
        self.queries = []
        self.released = []

    async def create(self, instance_id=None, **kwargs):
        return "instance-1", ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        self.queries.append(parameters["query"])
        return ToolResponse(text=self.text), 0.0, {"search_calls": 1}

    async def release(self, instance_id, **kwargs):
        self.released.append(instance_id)


class FailingSearchTool(FakeSearchTool):
    async def execute(self, instance_id, parameters, **kwargs):
        raise RuntimeError("retriever down")


def _find_subsequence(haystack, needle):
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


def _make_loop(
    tokenizer,
    server,
    tool,
    *,
    response_length=4096,
    max_assistant_turns=4,
    max_tool_response_length=3000,
):
    loop = object.__new__(SearchR1AgentLoop)
    loop.loop = asyncio.get_running_loop()
    loop.tokenizer = tokenizer
    loop.server_manager = server
    loop.search_tool = tool
    loop.response_length = response_length
    loop.prompt_length = 1024
    loop.max_assistant_turns = max_assistant_turns
    loop.max_tool_response_length = max_tool_response_length
    loop.tool_response_truncate_side = "middle"
    loop.rollout_config = SimpleNamespace(full_determinism=False, prompt_length=1024)
    return loop


def _load_tokenizer():
    tokenizer_id = os.environ.get("SEARCH_R1_TEST_TOKENIZER")
    if not tokenizer_id:
        return CharTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_id)


RAW_PROMPT = [
    {"role": "system", "content": "ignored by search_r1_agent"},
    {"role": "user", "content": "What is the capital of France?"},
]


class TestSearchR1Helpers(unittest.TestCase):
    def test_template_matches_upstream(self):
        assert SEARCH_R1_BASE_TEMPLATE.format(question="What is the capital of France?") == EXPECTED_PROMPT
        assert build_search_r1_prompt("What is the capital of France?") == EXPECTED_PROMPT

    def test_question_gets_question_mark_like_upstream_preprocessing(self):
        assert build_search_r1_prompt("Who wrote Hamlet").endswith("Question: Who wrote Hamlet?\n")

    def test_extract_question_skips_system_message(self):
        assert extract_question(RAW_PROMPT) == "What is the capital of France?"

    def test_parse_actions(self):
        assert parse_search_r1_action("<think>x</think><search> a b </search>") == ("search", "a b")
        assert parse_search_r1_action("a<search>first</search>b<search>second</search>") == ("search", "second")
        assert parse_search_r1_action("<think>x</think><answer> Paris </answer>") == ("answer", None)
        assert parse_search_r1_action("<think>no closing tag") == (None, None)
        assert parse_search_r1_action("</search>") == (None, None)
        assert parse_search_r1_action("<search>   </search>") == ("search", "")

    def test_registered(self):
        import verl.experimental.agent_loop  # noqa: F401  (import side effect registers built-in loops)

        entry = _agent_loop_registry["search_r1_agent"]
        assert entry["_target_"].endswith("SearchR1AgentLoop")


class TestSearchR1AgentLoop(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tokenizer = _load_tokenizer()

    async def test_search_information_answer_flow(self):
        tool = FakeSearchTool()
        turns = [
            "<think>I should search.</think><search> capital of France </search>",
            "<think>The result says Paris.</think><answer> Paris </answer>",
        ]
        server = FakeLLMServer(self.tokenizer, turns)
        loop = _make_loop(self.tokenizer, server, tool)

        output = await loop.run({"logprobs": True}, raw_prompt=RAW_PROMPT)

        # One search turn plus one answer turn; num_turns matches the tool_agent formula.
        assert len(server.calls) == 2
        assert output.num_turns == 4  # two assistant turns + one information turn + 1
        assert tool.queries == ["capital of France"]
        assert tool.released == ["instance-1"]

        # Stop strings are requested for every turn and the stop text stays in the response.
        for call in server.calls:
            assert call["sampling_params"]["stop"] == ["</search>", "</answer>"]
            assert call["sampling_params"]["include_stop_str_in_output"] is True
        response_text = self.tokenizer.decode(output.response_ids, skip_special_tokens=True)
        assert "</search>" in response_text
        assert "<information>Doc 1(Title: Paris)" in response_text

        # The information block is appended to the response and masked out of the loss exactly.
        observation_ids = self.tokenizer(
            "\n\n<information>Doc 1(Title: Paris) Paris is the capital of France.</information>\n\n",
            add_special_tokens=False,
        )["input_ids"]
        start = _find_subsequence(output.response_ids, observation_ids)
        assert start >= 0
        end = start + len(observation_ids)
        assert all(mask == 0 for mask in output.response_mask[start:end])
        assert all(mask == 1 for mask in output.response_mask[:start])
        assert all(mask == 1 for mask in output.response_mask[end:])
        assert len(output.response_mask) == len(output.response_ids)

        # The second turn sees the prompt + first turn + information tokens.
        assert server.calls[1]["prompt_ids"][: len(output.prompt_ids)] == output.prompt_ids
        assert (
            server.calls[1]["prompt_ids"][len(output.prompt_ids) :][: len(observation_ids) + start]
            == (output.response_ids[: start + len(observation_ids)])
        )

        # Logprobs cover every response token; information tokens get a 0 placeholder.
        assert len(output.response_logprobs) == len(output.response_ids)
        assert all(value == 0.0 for value in output.response_logprobs[start:end])

        # The prompt's example answer is never part of the response, so the EM reward cannot see it.
        assert "Beijing" not in response_text
        score = search_r1_like_qa_em.compute_score(solution_str=response_text, ground_truth={"target": ["Paris"]})
        assert score == 1.0
        assert search_r1_like_qa_em.compute_score(solution_str=response_text, ground_truth={"target": ["Beijing"]}) == 0

        # Termination diagnostics ride along in extra_fields for the reward/metrics path.
        assert output.extra_fields["search_r1_end_reason"] == "answer"
        assert output.extra_fields["search_r1_num_searches"] == 1
        assert output.extra_fields["search_r1_queries"] == ["capital of France"]

    async def test_no_tag_stops_without_search(self):
        server = FakeLLMServer(self.tokenizer, ["<think>I am not sure, no tag at all"])
        loop = _make_loop(self.tokenizer, server, FakeSearchTool())

        output = await loop.run({"logprobs": False}, raw_prompt=RAW_PROMPT)

        assert len(server.calls) == 1
        assert output.num_turns == 2
        assert set(output.response_mask) == {1}
        assert output.response_logprobs is None
        assert output.extra_fields["search_r1_end_reason"] == "no_action"
        assert output.extra_fields["search_r1_num_searches"] == 0
        assert output.extra_fields["search_r1_queries"] == []

    async def test_max_assistant_turns_caps_generations_without_a_trailing_search(self):
        tool = FakeSearchTool()
        server = FakeLLMServer(
            self.tokenizer,
            [
                "<think>one</think><search> q1 </search>",
                "<think>two</think><search> q2 </search>",
                "<answer> never reached </answer>",
            ],
        )
        loop = _make_loop(self.tokenizer, server, tool, max_assistant_turns=2)

        output = await loop.run({"logprobs": False}, raw_prompt=RAW_PROMPT)

        assert len(server.calls) == 2
        # The final allowed generation still ends with </search>, but it must not search.
        assert tool.queries == ["q1"]
        assert output.num_turns == 4  # two assistant turns + one information turn + 1
        assert output.num_turns <= 2 * loop.max_assistant_turns
        response_text = self.tokenizer.decode(output.response_ids, skip_special_tokens=True)
        assert "<search> q2 </search>" in response_text
        assert "never reached" not in response_text
        assert "<information>" in response_text
        assert output.extra_fields["search_r1_end_reason"] == "turn_cap"
        assert output.extra_fields["search_r1_num_searches"] == 1
        assert output.extra_fields["search_r1_queries"] == ["q1"]

    async def test_single_turn_cap_does_not_search(self):
        tool = FakeSearchTool()
        server = FakeLLMServer(self.tokenizer, ["<think>x</think><search> q </search>"])
        loop = _make_loop(self.tokenizer, server, tool, max_assistant_turns=1)

        output = await loop.run({"logprobs": False}, raw_prompt=RAW_PROMPT)

        assert len(server.calls) == 1
        assert tool.queries == []
        assert output.num_turns == 2
        assert set(output.response_mask) == {1}
        assert output.extra_fields["search_r1_end_reason"] == "turn_cap"
        assert output.extra_fields["search_r1_num_searches"] == 0

    async def test_no_budget_after_generation_does_not_search(self):
        turn_text = "<search> q </search>"
        turn_ids = self.tokenizer(turn_text, add_special_tokens=False)["input_ids"]
        tool = FakeSearchTool()
        server = FakeLLMServer(self.tokenizer, [turn_text, "<answer> unreachable </answer>"])
        loop = _make_loop(self.tokenizer, server, tool, response_length=len(turn_ids), max_assistant_turns=4)

        output = await loop.run({"logprobs": False}, raw_prompt=RAW_PROMPT)

        assert len(server.calls) == 1
        assert tool.queries == []
        assert output.num_turns == 2
        assert len(output.response_ids) == len(turn_ids)
        assert output.extra_fields["search_r1_end_reason"] == "length"
        assert output.extra_fields["search_r1_num_searches"] == 0

    async def test_total_response_budget_truncates_information(self):
        turn_text = "<search> q </search>"
        turn_ids = self.tokenizer(turn_text, add_special_tokens=False)["input_ids"]
        # Leave only 12 tokens for the observation, which is longer than that.
        response_length = len(turn_ids) + 12
        tool = FakeSearchTool(text="x" * 200)
        server = FakeLLMServer(self.tokenizer, [turn_text, "<answer> unreachable </answer>"])
        loop = _make_loop(self.tokenizer, server, tool, response_length=response_length)

        output = await loop.run({"logprobs": True}, raw_prompt=RAW_PROMPT)

        assert len(server.calls) == 1
        assert len(output.response_ids) == response_length
        assert len(output.response_mask) == response_length
        assert len(output.response_logprobs) == response_length
        assert all(mask == 1 for mask in output.response_mask[: len(turn_ids)])
        assert all(mask == 0 for mask in output.response_mask[len(turn_ids) :])
        observation_ids = self.tokenizer(
            "\n\n<information>" + "x" * 200 + "</information>\n\n", add_special_tokens=False
        )["input_ids"]
        assert output.response_ids[len(turn_ids) :] == observation_ids[:12]
        assert output.extra_fields["search_r1_end_reason"] == "length"
        assert output.extra_fields["search_r1_num_searches"] == 1
        assert output.extra_fields["search_r1_queries"] == ["q"]

    async def test_generation_is_capped_by_remaining_budget(self):
        # The fake server ignores max_tokens and returns more tokens than the remaining budget.
        response_length = 5
        long_turn = "<think>thinking hard about this question</think>"
        server = FakeLLMServer(self.tokenizer, [long_turn, "<answer> unreachable </answer>"])
        loop = _make_loop(self.tokenizer, server, FakeSearchTool(), response_length=response_length)

        output = await loop.run({"logprobs": False}, raw_prompt=RAW_PROMPT)

        assert len(server.calls) == 1
        assert server.calls[0]["sampling_params"]["max_tokens"] == response_length
        assert len(output.response_ids) == response_length
        assert set(output.response_mask) == {1}
        assert output.extra_fields["search_r1_end_reason"] == "no_action"

    async def test_search_failure_becomes_observation(self):
        server = FakeLLMServer(
            self.tokenizer,
            [
                "<think>search</think><search> q </search>",
                "<think>no results</think><answer> unknown </answer>",
            ],
        )
        loop = _make_loop(self.tokenizer, server, FailingSearchTool())

        output = await loop.run({"logprobs": False}, raw_prompt=RAW_PROMPT)

        response_text = self.tokenizer.decode(output.response_ids, skip_special_tokens=True)
        assert "Search failed: retriever down" in response_text
        assert len(server.calls) == 2
        assert output.extra_fields["search_r1_end_reason"] == "answer"
        assert output.extra_fields["search_r1_num_searches"] == 1
        assert output.extra_fields["search_r1_queries"] == ["q"]

    def test_reward_reports_termination_metrics(self):
        extra_info = {
            "search_r1_end_reason": "turn_cap",
            "search_r1_num_searches": 2,
            "search_r1_queries": ["first", "second"],
        }
        result = search_r1_like_qa_em.compute_score(
            solution_str="<answer> Paris </answer>",
            ground_truth={"target": ["Paris"]},
            extra_info=extra_info,
        )
        assert isinstance(result, dict)
        assert result["score"] == 1.0
        assert result["acc"] == 1.0
        assert result["search_r1_end_reason"] == "turn_cap"
        assert result["search_r1_end_turn_cap"] == 1.0
        assert result["search_r1_end_answer"] == 0.0
        assert result["search_r1_num_searches"] == 2
        assert result["search_r1_queries"] == ["first", "second"]

        # Without agent-loop diagnostics the reward keeps its scalar contract.
        assert (
            search_r1_like_qa_em.compute_score(
                solution_str="<answer> Paris </answer>", ground_truth={"target": ["Paris"]}
            )
            == 1.0
        )


if __name__ == "__main__":
    unittest.main()

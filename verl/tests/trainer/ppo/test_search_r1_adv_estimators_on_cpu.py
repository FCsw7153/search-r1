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
"""CPU unit tests for the Search-R1 advantage estimators.

Covers Stratified GRPO (arXiv 2510.06214) and GCA (arXiv 2610.23148) on hand-built batches:
stratification, singleton strata, the four GCA cases, the 10% cap, sign preservation, and the
requirement that plain GRPO (and alpha=0 / single-stratum Stratified GRPO) is unchanged.
"""

import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    compute_gca_outcome_advantage,
    compute_grpo_outcome_advantage,
    compute_stratified_grpo_outcome_advantage,
    get_adv_estimator_fn,
)
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories

# response layout for a two-search trajectory with an answer turn:
#   search #1 | information | search #2 | information | answer
SEARCH_SEARCH_ANSWER_MASK = [1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1]
# terminal timeout / no-answer trajectory: search #1 | information | search #2
SEARCH_SEARCH_MASK = [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0]


def _make_batch(masks, scores, index, num_searches, hit, first_hit_turn, has_answer):
    masks = torch.tensor(masks, dtype=torch.long)
    rewards = torch.zeros(masks.shape, dtype=torch.float32)
    for row, score in enumerate(scores):
        last = int(masks[row].nonzero().max())
        rewards[row, last] = float(score)
    return {
        "token_level_rewards": rewards,
        "response_mask": masks,
        "index": np.array(index, dtype=object),
        "num_searches": np.asarray(num_searches, dtype=np.int64),
        "hit": np.asarray(hit, dtype=np.int64),
        "first_hit_turn": np.asarray(first_hit_turn, dtype=np.int64),
        "has_answer": np.asarray(has_answer, dtype=np.int64),
    }


def _constant_salience_config(**overrides):
    config = {"gca_salience_mode": "constant"}
    config.update(overrides)
    return config


class TestRegistration:
    def test_both_estimators_are_registered(self):
        assert get_adv_estimator_fn("stratified_grpo") is compute_stratified_grpo_outcome_advantage
        assert get_adv_estimator_fn("gca") is compute_gca_outcome_advantage
        assert AdvantageEstimator.STRATIFIED_GRPO == "stratified_grpo"
        assert AdvantageEstimator.GCA == "gca"


class TestStratifiedGRPO:
    def test_single_stratum_is_exactly_grpo(self):
        batch = _make_batch(
            [SEARCH_SEARCH_ANSWER_MASK] * 4,
            scores=[1.0, 0.0, 1.0, 0.0],
            index=["g"] * 4,
            num_searches=[2, 2, 2, 2],
            hit=[1, 1, 1, 1],
            first_hit_turn=[1, 1, 1, 1],
            has_answer=[1, 1, 1, 1],
        )
        grpo_adv, _ = compute_grpo_outcome_advantage(
            batch["token_level_rewards"], batch["response_mask"], batch["index"]
        )
        strat_adv, _ = compute_stratified_grpo_outcome_advantage(
            batch["token_level_rewards"],
            batch["response_mask"],
            batch["index"],
            num_searches=batch["num_searches"],
        )
        assert torch.allclose(strat_adv, grpo_adv, atol=1e-6)

    def test_alpha_zero_is_exactly_grpo(self):
        batch = _make_batch(
            [SEARCH_SEARCH_ANSWER_MASK, SEARCH_SEARCH_MASK, SEARCH_SEARCH_ANSWER_MASK, SEARCH_SEARCH_MASK],
            scores=[1.0, 0.0, 1.0, 0.0],
            index=["g"] * 4,
            num_searches=[0, 0, 2, 2],
            hit=[1, 0, 1, 0],
            first_hit_turn=[1, -1, 1, -1],
            has_answer=[1, 0, 1, 0],
        )
        grpo_adv, _ = compute_grpo_outcome_advantage(
            batch["token_level_rewards"], batch["response_mask"], batch["index"]
        )
        strat_adv, _ = compute_stratified_grpo_outcome_advantage(
            batch["token_level_rewards"],
            batch["response_mask"],
            batch["index"],
            num_searches=batch["num_searches"],
            alpha=0.0,
        )
        assert torch.equal(strat_adv, grpo_adv)

    def test_mixed_strata_and_singletons(self):
        scores = [1.0, 0.0, 1.0, 0.0]
        batch = _make_batch(
            [SEARCH_SEARCH_ANSWER_MASK] * 4,
            scores=scores,
            index=["g"] * 4,
            num_searches=[0, 0, 1, 2],
            hit=[1, 0, 1, 0],
            first_hit_turn=[1, -1, 1, -1],
            has_answer=[1, 0, 1, 0],
        )
        metrics = {}
        # alpha=1 isolates SAN: count-0 stratum has two rows, the others are singletons.
        san_adv, _ = compute_stratified_grpo_outcome_advantage(
            batch["token_level_rewards"],
            batch["response_mask"],
            batch["index"],
            num_searches=batch["num_searches"],
            alpha=1.0,
            metrics=metrics,
        )
        stratum = torch.tensor(scores[:2])
        expected = (stratum - stratum.mean()) / (stratum.std() + 1e-6)
        mask0 = batch["response_mask"][0].bool()
        mask1 = batch["response_mask"][1].bool()
        assert torch.allclose(san_adv[0][mask0], expected[0].expand(int(mask0.sum())))
        assert torch.allclose(san_adv[1][mask1], expected[1].expand(int(mask1.sum())))
        # Singleton strata have SAN = 0.
        assert torch.equal(san_adv[2], torch.zeros_like(san_adv[2]))
        assert torch.equal(san_adv[3], torch.zeros_like(san_adv[3]))

        # Diagnostics: three strata in one group, two of them singletons, two singleton rows,
        # and the count-0 stratum has nonzero std.
        assert metrics["training/adv_estimator/stratified_grpo/num_strata"] == pytest.approx(3.0)
        assert metrics["training/adv_estimator/stratified_grpo/mean_stratum_size"] == pytest.approx(4 / 3)
        assert metrics["training/adv_estimator/stratified_grpo/singleton_strata_frac"] == pytest.approx(2 / 3)
        assert metrics["training/adv_estimator/stratified_grpo/zero_san_traj_frac"] == pytest.approx(0.5)

    def test_singletons_fall_back_to_global_advantage(self):
        batch = _make_batch(
            [SEARCH_SEARCH_ANSWER_MASK] * 3,
            scores=[1.0, 0.0, 0.0],
            index=["g"] * 3,
            num_searches=[0, 1, 2],
            hit=[1, 1, 0],
            first_hit_turn=[1, 1, -1],
            has_answer=[1, 1, 0],
        )
        grpo_adv, _ = compute_grpo_outcome_advantage(
            batch["token_level_rewards"], batch["response_mask"], batch["index"]
        )
        adv, _ = compute_stratified_grpo_outcome_advantage(
            batch["token_level_rewards"],
            batch["response_mask"],
            batch["index"],
            num_searches=batch["num_searches"],
            alpha=0.6,
        )
        # Every stratum here is a singleton, so SAN=0 and the blend is 0.4 * A_GN.
        assert torch.allclose(adv, 0.4 * grpo_adv, atol=1e-6)

    def test_population_std_option_matches_paper_algorithm(self):
        batch = _make_batch(
            [SEARCH_SEARCH_ANSWER_MASK] * 4,
            scores=[1.0, 0.0, 1.0, 0.0],
            index=["g"] * 4,
            num_searches=[2, 2, 2, 2],
            hit=[1, 1, 1, 1],
            first_hit_turn=[1, 1, 1, 1],
            has_answer=[1, 1, 1, 1],
        )
        adv, _ = compute_stratified_grpo_outcome_advantage(
            batch["token_level_rewards"],
            batch["response_mask"],
            batch["index"],
            num_searches=batch["num_searches"],
            alpha=1.0,
            config={"stratified_grpo_population_std": True},
        )
        scores = batch["token_level_rewards"].sum(dim=-1)
        expected = (scores - scores.mean()) / (scores.std(unbiased=False) + 1e-6)
        for row in range(4):
            mask = batch["response_mask"][row].bool()
            assert torch.allclose(adv[row][mask], expected[row].expand(int(mask.sum())))

    def test_missing_num_searches_raises(self):
        batch = _make_batch(
            [SEARCH_SEARCH_ANSWER_MASK],
            scores=[1.0],
            index=["g"],
            num_searches=[1],
            hit=[1],
            first_hit_turn=[1],
            has_answer=[1],
        )
        with pytest.raises(ValueError, match="search_r1_num_searches"):
            compute_stratified_grpo_outcome_advantage(
                batch["token_level_rewards"], batch["response_mask"], batch["index"]
            )


class TestGCA:
    def _mixed_case_batch(self):
        # A, B1, B2, C share one prompt group: scores [1, 0, 0, 0].
        return _make_batch(
            [SEARCH_SEARCH_ANSWER_MASK, SEARCH_SEARCH_ANSWER_MASK, SEARCH_SEARCH_ANSWER_MASK, SEARCH_SEARCH_MASK],
            scores=[1.0, 0.0, 0.0, 0.0],
            index=["g"] * 4,
            num_searches=[2, 2, 2, 2],
            hit=[1, 1, 1, 0],
            first_hit_turn=[1, 1, 1, -1],
            has_answer=[1, 1, 0, 0],
        )

    def test_four_gca_cases(self):
        batch = self._mixed_case_batch()
        metrics = {}
        adv, returns = compute_gca_outcome_advantage(
            batch["token_level_rewards"],
            batch["response_mask"],
            batch["index"],
            num_searches=batch["num_searches"],
            hit=batch["hit"],
            first_hit_turn=batch["first_hit_turn"],
            has_answer=batch["has_answer"],
            config=_constant_salience_config(),
            metrics=metrics,
        )
        assert torch.equal(adv, returns)
        base, _ = compute_grpo_outcome_advantage(
            batch["token_level_rewards"], batch["response_mask"], batch["index"]
        )

        # Type A: pre-hit search +, redundant search -, answer +.
        vsearch, rsearch, answer = 0, 4, 9
        assert adv[0, vsearch] > base[0, vsearch]
        assert adv[0, rsearch] < base[0, rsearch]
        assert adv[0, answer] > base[0, answer]

        # Type B1: vSearch +, rSearch -, wrong answer -, with the answer penalty bigger.
        assert adv[1, vsearch] > base[1, vsearch]
        assert adv[1, rsearch] < base[1, rsearch]
        assert adv[1, answer] < base[1, answer]
        assert (adv[1, answer] - base[1, answer]).abs() > (adv[1, rsearch] - base[1, rsearch]).abs()

        # Type B2: timeout/no valid answer; the terminal turn is untouched (lambda=0).
        assert adv[2, vsearch] > base[2, vsearch]
        assert adv[2, rsearch] < base[2, rsearch]
        assert torch.equal(adv[2, 9:], base[2, 9:])

        # Type C: no hit and wrong; no correction at all.
        assert torch.equal(adv[3], base[3])

        # No sign flips and the 10% cap holds on every response token.
        for row in range(4):
            mask = batch["response_mask"][row].bool()
            baseline = base[row, mask]
            corrected = adv[row, mask]
            nonzero = baseline != 0
            assert torch.all(corrected[nonzero].sign() == baseline[nonzero].sign())
            assert torch.all((corrected - baseline).abs() <= 0.1 * baseline.abs() + 1e-6)
            assert torch.all(adv[row][~mask] == 0)

        assert metrics["training/adv_estimator/gca/type_a_frac"] == pytest.approx(0.25)
        assert metrics["training/adv_estimator/gca/type_b1_frac"] == pytest.approx(0.25)
        assert metrics["training/adv_estimator/gca/type_b2_frac"] == pytest.approx(0.25)
        assert metrics["training/adv_estimator/gca/type_c_frac"] == pytest.approx(0.25)
        assert metrics["training/adv_estimator/gca/corrected_traj_frac"] == pytest.approx(0.75)
        assert 0.0 < metrics["training/adv_estimator/gca/mean_abs_correction_ratio"] <= 0.1

    def test_constant_salience_hits_the_cap_exactly(self):
        batch = self._mixed_case_batch()
        adv, _ = compute_gca_outcome_advantage(
            batch["token_level_rewards"],
            batch["response_mask"],
            batch["index"],
            num_searches=batch["num_searches"],
            hit=batch["hit"],
            first_hit_turn=batch["first_hit_turn"],
            has_answer=batch["has_answer"],
            config=_constant_salience_config(gca_c_max=0.5),
        )
        base, _ = compute_grpo_outcome_advantage(
            batch["token_level_rewards"], batch["response_mask"], batch["index"]
        )
        # Type-A answer segment: eta=0.2, lambda=1.0, c=0.5 -> exactly 10% of |A|.
        assert torch.allclose(adv[0, 9:], base[0, 9:] * 1.1)
        # Type-A pre-hit search: 0.2 * 0.5 * 0.5 = 5%.
        assert torch.allclose(adv[0, 0:3], base[0, 0:3] * 1.05)

    def test_d1_and_d0(self):
        batch = _make_batch(
            [SEARCH_SEARCH_ANSWER_MASK, SEARCH_SEARCH_ANSWER_MASK],
            scores=[1.0, 0.0],
            index=["g"] * 2,
            num_searches=[2, 2],
            hit=[0, 0],
            first_hit_turn=[-1, -1],
            has_answer=[1, 0],
        )
        base, _ = compute_grpo_outcome_advantage(
            batch["token_level_rewards"], batch["response_mask"], batch["index"]
        )
        d1, _ = compute_gca_outcome_advantage(
            batch["token_level_rewards"],
            batch["response_mask"],
            batch["index"],
            num_searches=batch["num_searches"],
            hit=batch["hit"],
            first_hit_turn=batch["first_hit_turn"],
            has_answer=batch["has_answer"],
            config=_constant_salience_config(),
            metrics=(d1_metrics := {}),
        )
        # Correct without evidence: both search and answer get negative residual.
        assert d1[0, 0] < base[0, 0]
        assert d1[0, 9] < base[0, 9]
        assert d1_metrics["training/adv_estimator/gca/type_d1_frac"] == pytest.approx(0.5)

        d0, _ = compute_gca_outcome_advantage(
            batch["token_level_rewards"],
            batch["response_mask"],
            batch["index"],
            num_searches=batch["num_searches"],
            hit=batch["hit"],
            first_hit_turn=batch["first_hit_turn"],
            has_answer=batch["has_answer"],
            config=_constant_salience_config(gca_assume_retrievable=False),
            metrics=(d0_metrics := {}),
        )
        assert torch.equal(d0[0], torch.zeros_like(d0[0]))
        assert d0_metrics["training/adv_estimator/gca/type_d0_frac"] == pytest.approx(0.5)

    def test_zero_variance_group_skips_correction(self):
        batch = _make_batch(
            [SEARCH_SEARCH_ANSWER_MASK, SEARCH_SEARCH_ANSWER_MASK],
            scores=[1.0, 1.0],
            index=["g"] * 2,
            num_searches=[2, 2],
            hit=[1, 1],
            first_hit_turn=[1, 1],
            has_answer=[1, 1],
        )
        metrics = {}
        adv, _ = compute_gca_outcome_advantage(
            batch["token_level_rewards"],
            batch["response_mask"],
            batch["index"],
            num_searches=batch["num_searches"],
            hit=batch["hit"],
            first_hit_turn=batch["first_hit_turn"],
            has_answer=batch["has_answer"],
            config=_constant_salience_config(),
            metrics=metrics,
        )
        # All-equal rewards give A_hat = 0; the algorithm skips the correction entirely.
        assert torch.equal(adv, torch.zeros_like(adv))
        assert metrics["training/adv_estimator/gca/skipped_zero_var_frac"] == pytest.approx(1.0)

    def test_token_salience_approximation_is_bounded(self):
        batch = self._mixed_case_batch()
        torch.manual_seed(0)
        old_log_probs = -torch.rand(batch["response_mask"].shape) * 3.0
        entropy = torch.rand(batch["response_mask"].shape) * 4.0
        metrics = {}
        adv, _ = compute_gca_outcome_advantage(
            batch["token_level_rewards"],
            batch["response_mask"],
            batch["index"],
            num_searches=batch["num_searches"],
            hit=batch["hit"],
            first_hit_turn=batch["first_hit_turn"],
            has_answer=batch["has_answer"],
            old_log_probs=old_log_probs,
            entropy=entropy,
            config={"gca_salience_mode": "approx"},
            metrics=metrics,
        )
        base, _ = compute_grpo_outcome_advantage(
            batch["token_level_rewards"], batch["response_mask"], batch["index"]
        )
        for row in range(4):
            mask = batch["response_mask"][row].bool()
            baseline = base[row, mask]
            corrected = adv[row, mask]
            nonzero = baseline != 0
            assert torch.all(corrected[nonzero].sign() == baseline[nonzero].sign())
            assert torch.all((corrected - baseline).abs() <= 0.1 * baseline.abs() + 1e-6)
        assert 0.0 <= metrics["training/adv_estimator/gca/mean_salience"] <= 0.5

    def test_missing_evidence_fields_raise(self):
        batch = self._mixed_case_batch()
        with pytest.raises(ValueError, match="search_r1_hit"):
            compute_gca_outcome_advantage(
                batch["token_level_rewards"],
                batch["response_mask"],
                batch["index"],
                num_searches=batch["num_searches"],
            )

    def test_invalid_salience_mode_raises(self):
        batch = self._mixed_case_batch()
        with pytest.raises(ValueError, match="gca_salience_mode"):
            compute_gca_outcome_advantage(
                batch["token_level_rewards"],
                batch["response_mask"],
                batch["index"],
                num_searches=batch["num_searches"],
                hit=batch["hit"],
                first_hit_turn=batch["first_hit_turn"],
                has_answer=batch["has_answer"],
                config={"gca_salience_mode": "bogus"},
            )


class TestTrainerPlumbing:
    """The registry dispatch and the V1 multi-trajectory scatter must preserve token-level GCA."""

    def _dataproto(self):
        masks = torch.tensor(
            [
                [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # partial row of session 0
                [1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1],  # final row of session 0 (type A)
                [1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1],  # type B1
                [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # type C
            ],
            dtype=torch.long,
        )
        rewards = torch.zeros(masks.shape, dtype=torch.float32)
        for row, score in enumerate([0.0, 1.0, 0.0, 0.0]):
            rewards[row, int(masks[row].nonzero().max())] = score
        return DataProto.from_dict(
            tensors={
                "token_level_rewards": rewards,
                "response_mask": masks,
                "old_log_probs": -torch.rand(masks.shape),
                "entropy": torch.rand(masks.shape),
            },
            non_tensors={
                "uid": np.array(["g"] * 4, dtype=object),
                "search_r1_num_searches": np.array([2, 2, 2, 2], dtype=np.int64),
                "search_r1_hit": np.array([1, 1, 1, 0], dtype=np.int64),
                "search_r1_first_hit_turn": np.array([1, 1, 1, -1], dtype=np.int64),
                "search_r1_has_answer": np.array([1, 1, 1, 0], dtype=np.int64),
            },
        )

    def test_compute_advantage_dispatches_gca_with_metrics(self):
        data = self._dataproto()
        result = compute_advantage(
            data,
            adv_estimator="gca",
            config={"gca_salience_mode": "constant"},
        )
        assert "adv_estimator_metrics" in result.meta_info
        assert result.meta_info["adv_estimator_metrics"]["training/adv_estimator/gca/type_a_frac"] == 0.25
        # GCA keeps per-token corrections instead of one scalar per trajectory.
        row = 1
        mask = result.batch["response_mask"][row].bool()
        assert result.batch["advantages"][row][mask].unique().numel() > 1

    def test_multi_trajectory_scatter_keeps_full_gca_rows(self):
        data = self._dataproto()
        batch_keys = ["g_0_0", "g_0_1", "g_1_0", "g_2_0"]
        result = compute_advantage_for_multi_trajectories(
            data,
            batch_keys=batch_keys,
            adv_estimator="gca",
            config={"gca_salience_mode": "constant"},
        )
        # Rows 0 and 1 belong to the same session: row 0 must receive the final row's full
        # token-level advantage, masked to its own (shorter) response.
        assert torch.equal(
            result.batch["advantages"][0],
            result.batch["advantages"][1] * data.batch["response_mask"][0],
        )
        assert "adv_estimator_metrics" in result.meta_info
        assert result.batch["advantages"][1, 0] != result.batch["advantages"][1, 9]

    def test_stratified_dispatch(self):
        data = self._dataproto()
        # Stratified GRPO does not need the GCA fields.
        for key in ("search_r1_hit", "search_r1_first_hit_turn", "search_r1_has_answer"):
            data.non_tensor_batch.pop(key)
        result = compute_advantage(data, adv_estimator="stratified_grpo", config={"stratified_grpo_alpha": 0.6})
        assert "adv_estimator_metrics" in result.meta_info
        metrics = result.meta_info["adv_estimator_metrics"]
        assert metrics["training/adv_estimator/stratified_grpo/alpha"] == pytest.approx(0.6)
        # All four rows have two searches: one stratum, so the estimator is a scaled GRPO.
        assert metrics["training/adv_estimator/stratified_grpo/num_strata"] == pytest.approx(1.0)
        assert metrics["training/adv_estimator/stratified_grpo/singleton_strata_frac"] == pytest.approx(0.0)

    def test_multi_trajectory_stratified_uses_group_relative_path(self):
        data = self._dataproto()
        result = compute_advantage_for_multi_trajectories(
            data,
            batch_keys=["g_0_0", "g_0_1", "g_1_0", "g_2_0"],
            adv_estimator="stratified_grpo",
            config={"stratified_grpo_alpha": 0.6},
        )
        # Rows 0 and 1 are the same session, so both get the final row's scalar advantage,
        # each masked to its own response.
        assert torch.equal(
            result.batch["advantages"][0],
            result.batch["advantages"][1] * data.batch["response_mask"][0],
        )
        assert "adv_estimator_metrics" in result.meta_info

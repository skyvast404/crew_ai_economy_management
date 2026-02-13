"""Tests for lib_custom/team_generator.py."""

import pytest

from lib_custom.personality_types import PERSONALITY_TYPES
from lib_custom.team_generator import (
    DiversityMetrics,
    TeamComposition,
    calculate_blau_index,
    calculate_diversity_metrics,
    classify_diversity_stratum,
    composition_to_members,
    generate_team_compositions,
)


# ---------------------------------------------------------------------------
# calculate_blau_index
# ---------------------------------------------------------------------------
class TestBlauIndex:
    """Tests for calculate_blau_index()."""

    def test_empty_list_returns_zero(self):
        assert calculate_blau_index([]) == 0.0

    def test_all_same_returns_zero(self):
        assert calculate_blau_index(["a", "a", "a", "a"]) == 0.0

    def test_two_equal_groups(self):
        # Blau = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5
        result = calculate_blau_index(["a", "b", "a", "b"])
        assert abs(result - 0.5) < 0.001

    def test_three_equal_groups(self):
        # Blau = 1 - 3*(1/3)² = 1 - 1/3 ≈ 0.667
        result = calculate_blau_index(["a", "b", "c"])
        assert abs(result - 0.6667) < 0.01

    def test_single_element(self):
        assert calculate_blau_index(["x"]) == 0.0

    def test_unequal_distribution(self):
        # 3 a's, 1 b → Blau = 1 - (0.75² + 0.25²) = 1 - 0.625 = 0.375
        result = calculate_blau_index(["a", "a", "a", "b"])
        assert abs(result - 0.375) < 0.001

    def test_range_between_zero_and_one(self):
        result = calculate_blau_index(["a", "b", "c", "d", "e"])
        assert 0.0 <= result <= 1.0

    def test_maximum_diversity(self):
        # All different = maximum Blau = 1 - k*(1/k)² = 1 - 1/k
        values = ["a", "b", "c", "d", "e"]
        result = calculate_blau_index(values)
        expected = 1.0 - 5 * (1 / 5) ** 2  # = 0.8
        assert abs(result - expected) < 0.001


# ---------------------------------------------------------------------------
# calculate_diversity_metrics
# ---------------------------------------------------------------------------
class TestDiversityMetrics:
    """Tests for calculate_diversity_metrics()."""

    def test_all_same_type(self):
        # All strategic_charger → same urgency, action, time
        metrics = calculate_diversity_metrics(["strategic_charger"] * 5)
        assert metrics.urgency_blau == 0.0
        assert metrics.action_pattern_blau == 0.0
        assert metrics.time_orientation_blau == 0.0
        assert metrics.composite == 0.0

    def test_mixed_types(self):
        # strategic_charger: high/early/future
        # zen_deadline: low/deadline/present
        metrics = calculate_diversity_metrics([
            "strategic_charger", "zen_deadline",
        ])
        assert metrics.urgency_blau == 0.5  # high/low
        assert metrics.action_pattern_blau == 0.5  # early/deadline
        assert metrics.time_orientation_blau == 0.5  # future/present

    def test_composite_is_average(self):
        metrics = calculate_diversity_metrics([
            "strategic_charger", "zen_deadline",
        ])
        expected = (metrics.urgency_blau + metrics.action_pattern_blau + metrics.time_orientation_blau) / 3
        assert abs(metrics.composite - expected) < 0.001

    def test_unknown_type_ignored(self):
        metrics = calculate_diversity_metrics(["strategic_charger", "nonexistent"])
        # Only 1 valid type → all Blau = 0
        assert metrics.composite == 0.0

    def test_empty_list(self):
        metrics = calculate_diversity_metrics([])
        assert metrics.composite == 0.0


# ---------------------------------------------------------------------------
# classify_diversity_stratum
# ---------------------------------------------------------------------------
class TestClassifyStratum:
    """Tests for classify_diversity_stratum()."""

    def test_low(self):
        assert classify_diversity_stratum(0.0) == "low"
        assert classify_diversity_stratum(0.2) == "low"
        assert classify_diversity_stratum(0.47) == "low"

    def test_medium(self):
        assert classify_diversity_stratum(0.48) == "medium"
        assert classify_diversity_stratum(0.49) == "medium"
        assert classify_diversity_stratum(0.50) == "medium"

    def test_high(self):
        assert classify_diversity_stratum(0.51) == "high"
        assert classify_diversity_stratum(0.8) == "high"
        assert classify_diversity_stratum(1.0) == "high"


# ---------------------------------------------------------------------------
# generate_team_compositions
# ---------------------------------------------------------------------------
class TestGenerateCompositions:
    """Tests for generate_team_compositions()."""

    def test_returns_three_strata(self):
        result = generate_team_compositions(n_per_stratum=3, seed=42)
        assert "low" in result
        assert "medium" in result
        assert "high" in result

    def test_correct_count_per_stratum(self):
        n = 4
        result = generate_team_compositions(n_per_stratum=n, seed=42)
        for stratum, comps in result.items():
            assert len(comps) <= n

    def test_team_size_is_five(self):
        result = generate_team_compositions(n_per_stratum=2, seed=42)
        for stratum, comps in result.items():
            for comp in comps:
                assert len(comp.type_ids) == 5

    def test_deterministic_with_seed(self):
        r1 = generate_team_compositions(n_per_stratum=3, seed=123)
        r2 = generate_team_compositions(n_per_stratum=3, seed=123)
        for stratum in ["low", "medium", "high"]:
            ids1 = [c.composition_id for c in r1[stratum]]
            ids2 = [c.composition_id for c in r2[stratum]]
            assert ids1 == ids2

    def test_different_seeds_produce_different_results(self):
        r1 = generate_team_compositions(n_per_stratum=3, seed=1)
        r2 = generate_team_compositions(n_per_stratum=3, seed=2)
        # At least one stratum should differ
        any_diff = False
        for stratum in ["low", "medium", "high"]:
            types1 = [tuple(c.type_ids) for c in r1.get(stratum, [])]
            types2 = [tuple(c.type_ids) for c in r2.get(stratum, [])]
            if types1 != types2:
                any_diff = True
        assert any_diff

    def test_all_type_ids_are_valid(self):
        result = generate_team_compositions(n_per_stratum=5, seed=42)
        for comps in result.values():
            for comp in comps:
                for tid in comp.type_ids:
                    assert tid in PERSONALITY_TYPES

    def test_composition_has_correct_stratum(self):
        result = generate_team_compositions(n_per_stratum=3, seed=42)
        for stratum, comps in result.items():
            for comp in comps:
                assert comp.stratum == stratum


# ---------------------------------------------------------------------------
# composition_to_members
# ---------------------------------------------------------------------------
class TestCompositionToMembers:
    """Tests for composition_to_members()."""

    def test_returns_correct_count(self):
        comp = TeamComposition(
            composition_id="test_01",
            type_ids=["strategic_charger", "zen_deadline", "light_explorer",
                       "planned_pusher", "stable_implementer"],
            diversity=DiversityMetrics(
                urgency_blau=0.5, action_pattern_blau=0.5,
                time_orientation_blau=0.5, composite=0.5,
            ),
            stratum="medium",
        )
        members = composition_to_members(comp)
        assert len(members) == 5

    def test_members_have_correct_type_ids(self):
        type_ids = ["strategic_charger", "zen_deadline", "light_explorer",
                     "planned_pusher", "stable_implementer"]
        comp = TeamComposition(
            composition_id="test_02",
            type_ids=type_ids,
            diversity=DiversityMetrics(
                urgency_blau=0.5, action_pattern_blau=0.5,
                time_orientation_blau=0.5, composite=0.5,
            ),
            stratum="medium",
        )
        members = composition_to_members(comp)
        result_ids = [m.personality_type_id for m in members]
        assert result_ids == type_ids

    def test_members_have_names(self):
        comp = TeamComposition(
            composition_id="test_03",
            type_ids=["strategic_charger", "zen_deadline"],
            diversity=DiversityMetrics(
                urgency_blau=0.5, action_pattern_blau=0.5,
                time_orientation_blau=0.5, composite=0.5,
            ),
            stratum="low",
        )
        members = composition_to_members(comp)
        for m in members:
            assert len(m.name) > 0

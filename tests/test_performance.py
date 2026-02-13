"""Tests for lib_custom/engine/performance.py."""

from lib_custom.engine.performance import (
    PROJECT_TYPES,
    MemberFitness,
    PerformancePrediction,
    predict_performance,
)
from lib_custom.personality_types import TeamConfig, TeamMember


def _cfg(boss: str, type_ids: list[str]) -> TeamConfig:
    return TeamConfig(
        boss_type_id=boss,
        members=[
            TeamMember(id=f"m{i}", name=f"M{i}", personality_type_id=tid)
            for i, tid in enumerate(type_ids)
        ],
    )


class TestProjectTypes:
    def test_three_project_types(self):
        assert len(PROJECT_TYPES) == 3
        assert "urgent_launch" in PROJECT_TYPES
        assert "long_term_platform" in PROJECT_TYPES
        assert "exploratory_prototype" in PROJECT_TYPES


class TestPerformancePrediction:
    def test_urgent_launch_prefers_high_early(self):
        """High urgency, early action should score well for urgent launch."""
        config = _cfg("time_master", ["strategic_charger", "blitz_executor"])
        pred = predict_performance(config, "urgent_launch")
        assert isinstance(pred, PerformancePrediction)
        # Both are high-urgency early-action → should be good for urgent
        for s in pred.individual_scores:
            assert s.score >= 60

    def test_long_term_prefers_steady_future(self):
        """Steady, future-oriented should score well for long term."""
        config = _cfg("time_master", ["long_term_builder", "planned_pusher"])
        pred = predict_performance(config, "long_term_platform")
        for s in pred.individual_scores:
            assert s.score >= 60

    def test_exploratory_prefers_early_future(self):
        """Early action, future → good for exploration."""
        config = _cfg("time_master", ["calm_preemptive", "light_explorer"])
        pred = predict_performance(config, "exploratory_prototype")
        for s in pred.individual_scores:
            assert s.score >= 60

    def test_overall_score_in_range(self):
        config = _cfg("time_master", ["strategic_charger", "zen_deadline", "stable_implementer"])
        pred = predict_performance(config, "urgent_launch")
        assert 0 <= pred.overall_score <= 100
        assert 0 <= pred.team_synergy_score <= 100

    def test_empty_team(self):
        config = _cfg("time_master", [])
        pred = predict_performance(config, "urgent_launch")
        assert pred.individual_scores == []
        assert pred.overall_score >= 0

    def test_summary_not_empty(self):
        config = _cfg("time_master", ["strategic_charger"])
        pred = predict_performance(config, "urgent_launch")
        assert len(pred.summary) > 0

    def test_different_boss_affects_synergy(self):
        """time_master vs time_chaos should give different synergy scores for same team."""
        members = ["strategic_charger", "blitz_executor"]
        pred_master = predict_performance(_cfg("time_master", members), "urgent_launch")
        pred_chaos = predict_performance(_cfg("time_chaos", members), "urgent_launch")
        assert pred_master.team_synergy_score != pred_chaos.team_synergy_score

    def test_individual_fitness_detail(self):
        config = _cfg("time_master", ["strategic_charger"])
        pred = predict_performance(config, "urgent_launch")
        assert len(pred.individual_scores) == 1
        assert isinstance(pred.individual_scores[0], MemberFitness)
        assert pred.individual_scores[0].detail != ""

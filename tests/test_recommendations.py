"""Tests for lib_custom/engine/recommendations.py."""

from lib_custom.engine.recommendations import Recommendation, generate_recommendations
from lib_custom.personality_types import TeamConfig, TeamMember


def _cfg(boss: str, type_ids: list[str]) -> TeamConfig:
    return TeamConfig(
        boss_type_id=boss,
        members=[
            TeamMember(id=f"m{i}", name=f"M{i}", personality_type_id=tid)
            for i, tid in enumerate(type_ids)
        ],
    )


class TestRecommendations:
    def test_returns_list(self):
        config = _cfg("time_master", ["strategic_charger", "zen_deadline"])
        recs = generate_recommendations(config, "urgent_launch")
        assert isinstance(recs, list)
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_sorted_by_priority(self):
        config = _cfg("time_master", [
            "strategic_charger",
            "zen_deadline",
            "firefighter_closer",
        ])
        recs = generate_recommendations(config, "urgent_launch")
        for i in range(len(recs) - 1):
            assert recs[i].priority <= recs[i + 1].priority

    def test_task_assignment_for_early_future(self):
        """Early + future member should get planning task suggestion."""
        config = _cfg("time_master", ["strategic_charger"])
        recs = generate_recommendations(config, "urgent_launch")
        task_recs = [r for r in recs if r.category == "任务分配"]
        assert any("规划" in r.title or "前期" in r.title for r in task_recs)

    def test_task_assignment_for_steady(self):
        config = _cfg("time_master", ["stable_implementer"])
        recs = generate_recommendations(config, "urgent_launch")
        task_recs = [r for r in recs if r.category == "任务分配"]
        assert any("持续" in r.title for r in task_recs)

    def test_task_assignment_for_deadline_high(self):
        """High urgency deadline → checkpoint suggestion."""
        config = _cfg("time_master", ["firefighter_closer"])
        recs = generate_recommendations(config, "urgent_launch")
        task_recs = [r for r in recs if r.category == "任务分配"]
        assert any("checkpoint" in r.title.lower() or "检查点" in r.description for r in task_recs)

    def test_task_assignment_for_deadline_low(self):
        """Low urgency deadline → pre-deadline suggestion."""
        config = _cfg("time_master", ["zen_deadline"])
        recs = generate_recommendations(config, "urgent_launch")
        task_recs = [r for r in recs if r.category == "任务分配"]
        assert any("提前" in r.title or "截止" in r.title or "截止" in r.description for r in task_recs)

    def test_communication_for_time_master(self):
        config = _cfg("time_master", ["strategic_charger", "zen_deadline"])
        recs = generate_recommendations(config, "urgent_launch")
        comm_recs = [r for r in recs if r.category == "沟通策略"]
        assert len(comm_recs) >= 1

    def test_communication_for_time_chaos(self):
        config = _cfg("time_chaos", ["strategic_charger", "blitz_executor"])
        recs = generate_recommendations(config, "urgent_launch")
        comm_recs = [r for r in recs if r.category == "沟通策略"]
        assert any("缓冲" in r.title or "缓冲" in r.description for r in comm_recs)

    def test_risk_warnings_included(self):
        """Conflicts should surface as risk recommendations."""
        config = _cfg("time_chaos", [
            "goal_driven_deadline",
            "firefighter_closer",
            "zen_deadline",
        ])
        recs = generate_recommendations(config, "urgent_launch")
        risk_recs = [r for r in recs if r.category == "风险预警"]
        assert len(risk_recs) >= 1

    def test_empty_team(self):
        config = _cfg("time_master", [])
        recs = generate_recommendations(config, "urgent_launch")
        # No task assignment or specific recs, but communication recs may still exist
        assert isinstance(recs, list)

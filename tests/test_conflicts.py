"""Tests for lib_custom/engine/conflicts.py."""

from lib_custom.engine.conflicts import detect_conflicts
from lib_custom.personality_types import TeamConfig, TeamMember


def _cfg(boss: str, type_ids: list[str]) -> TeamConfig:
    return TeamConfig(
        boss_type_id=boss,
        members=[
            TeamMember(id=f"m{i}", name=f"M{i}", personality_type_id=tid)
            for i, tid in enumerate(type_ids)
        ],
    )


class TestConflictDetection:
    def test_empty_team_no_conflicts(self):
        config = _cfg("time_master", [])
        assert detect_conflicts(config) == []

    def test_deadline_chaos_high_risk(self):
        """3+ deadline members + chaos boss → high severity."""
        config = _cfg("time_chaos", [
            "goal_driven_deadline",
            "firefighter_closer",
            "zen_deadline",
        ])
        alerts = detect_conflicts(config)
        high_alerts = [a for a in alerts if a.severity == "high"]
        assert any("压线" in a.title or "deadline" in a.title.lower() for a in high_alerts)

    def test_no_future_strategic_blind_spot(self):
        """All present-oriented → strategic blind spot."""
        config = _cfg("time_master", [
            "blitz_executor",       # present
            "high_pressure_steady", # present
            "firefighter_closer",   # present
        ])
        alerts = detect_conflicts(config)
        assert any("战略盲区" in a.title for a in alerts)

    def test_groupthink(self):
        """80%+ same type → groupthink."""
        config = _cfg("time_master", [
            "strategic_charger",
            "strategic_charger",
            "strategic_charger",
            "strategic_charger",
            "blitz_executor",
        ])
        alerts = detect_conflicts(config)
        assert any("群体思维" in a.title for a in alerts)

    def test_all_high_urgency_burnout(self):
        """All high urgency → burnout warning."""
        config = _cfg("time_master", [
            "strategic_charger",    # high
            "blitz_executor",       # high
            "planned_pusher",       # high
        ])
        alerts = detect_conflicts(config)
        assert any("紧迫感" in a.title or "紧迫" in a.description for a in alerts)

    def test_early_vs_deadline_pacing(self):
        """2+ early + 2+ deadline → pacing conflict."""
        config = _cfg("time_master", [
            "strategic_charger",    # early
            "blitz_executor",       # early
            "goal_driven_deadline", # deadline
            "firefighter_closer",   # deadline
        ])
        alerts = detect_conflicts(config)
        assert any("节奏" in a.title for a in alerts)

    def test_sorted_by_severity(self):
        """Results should be sorted high → medium → low."""
        config = _cfg("time_chaos", [
            "blitz_executor",
            "firefighter_closer",
            "zen_deadline",
            "goal_driven_deadline",
        ])
        alerts = detect_conflicts(config)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        for i in range(len(alerts) - 1):
            assert severity_order[alerts[i].severity] <= severity_order[alerts[i + 1].severity]

    def test_no_conflict_balanced_team(self):
        """A balanced team should have fewer alerts."""
        config = _cfg("time_master", [
            "strategic_charger",    # high, early, future
            "stable_implementer",   # low, steady, present
        ])
        alerts = detect_conflicts(config)
        # No groupthink, no all-high-urgency, has future+present
        high_alerts = [a for a in alerts if a.severity == "high"]
        assert len(high_alerts) == 0

    def test_single_member_no_burnout(self):
        """Single member should not trigger all-high-urgency rule."""
        config = _cfg("time_master", ["strategic_charger"])
        alerts = detect_conflicts(config)
        assert not any("紧迫" in a.title for a in alerts)

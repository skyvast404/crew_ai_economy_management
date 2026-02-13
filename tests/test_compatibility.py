"""Tests for lib_custom/engine/compatibility.py."""


from lib_custom.engine.compatibility import (
    CompatibilityScore,
    boss_compatibility_for_team,
    calculate_boss_compatibility,
    calculate_peer_compatibility,
    peer_compatibility_matrix,
)
from lib_custom.personality_types import PersonalityDimensions, TeamConfig, TeamMember


class TestBossCompatibility:
    def test_time_master_high_early_future(self):
        """Best fit for time_master: high urgency, early action, future."""
        dims = PersonalityDimensions(urgency="high", action_pattern="early", time_orientation="future")
        score = calculate_boss_compatibility("time_master", dims)
        # 50 + 15 + 20 + 10 = 95
        assert score == 95

    def test_time_master_low_deadline_present(self):
        """Worst fit for time_master: low urgency, deadline, present."""
        dims = PersonalityDimensions(urgency="low", action_pattern="deadline", time_orientation="present")
        score = calculate_boss_compatibility("time_master", dims)
        # 50 + 5 + (-10) + 5 = 50
        assert score == 50

    def test_time_chaos_low_deadline_present(self):
        """Best fit for time_chaos: low urgency, deadline, present."""
        dims = PersonalityDimensions(urgency="low", action_pattern="deadline", time_orientation="present")
        score = calculate_boss_compatibility("time_chaos", dims)
        # 50 + 10 + 15 + 15 = 90
        assert score == 90

    def test_time_chaos_high_early_future(self):
        """Worst fit for time_chaos."""
        dims = PersonalityDimensions(urgency="high", action_pattern="early", time_orientation="future")
        score = calculate_boss_compatibility("time_chaos", dims)
        # 50 + (-5) + (-10) + (-5) = 30
        assert score == 30

    def test_score_clamped_0_100(self):
        """Score should never go below 0 or above 100."""
        dims = PersonalityDimensions(urgency="high", action_pattern="early", time_orientation="future")
        score = calculate_boss_compatibility("time_master", dims)
        assert 0 <= score <= 100

    def test_unknown_boss_type(self):
        """Unknown boss type → base score 50."""
        dims = PersonalityDimensions(urgency="high", action_pattern="early", time_orientation="future")
        score = calculate_boss_compatibility("unknown", dims)
        assert score == 50


class TestPeerCompatibility:
    def test_identical_dims(self):
        """Same dims → max bonus."""
        d = PersonalityDimensions(urgency="high", action_pattern="steady", time_orientation="future")
        score = calculate_peer_compatibility(d, d)
        # 50 + 10 + 15 + 10 = 85
        assert score == 85

    def test_early_vs_deadline(self):
        """Early vs deadline → penalty."""
        a = PersonalityDimensions(urgency="high", action_pattern="early", time_orientation="future")
        b = PersonalityDimensions(urgency="high", action_pattern="deadline", time_orientation="future")
        score = calculate_peer_compatibility(a, b)
        # 50 + 10 + (-15) + 10 = 55
        assert score == 55

    def test_completely_different(self):
        a = PersonalityDimensions(urgency="high", action_pattern="early", time_orientation="future")
        b = PersonalityDimensions(urgency="low", action_pattern="deadline", time_orientation="present")
        score = calculate_peer_compatibility(a, b)
        # 50 + 0 + (-15) + 0 = 35
        assert score == 35

    def test_symmetric(self):
        """Score(a,b) == Score(b,a)."""
        a = PersonalityDimensions(urgency="high", action_pattern="early", time_orientation="present")
        b = PersonalityDimensions(urgency="low", action_pattern="steady", time_orientation="future")
        assert calculate_peer_compatibility(a, b) == calculate_peer_compatibility(b, a)


class TestBatchHelpers:
    def _make_config(self, boss: str, members: list[tuple[str, str, str]]) -> TeamConfig:
        return TeamConfig(
            boss_type_id=boss,
            members=[
                TeamMember(id=f"m{i}", name=f"Member{i}", personality_type_id=pid)
                for i, (pid, _, _) in enumerate(members)
            ],
        )

    def test_boss_compatibility_for_team(self):
        config = TeamConfig(
            boss_type_id="time_master",
            members=[
                TeamMember(id="a", name="Alice", personality_type_id="strategic_charger"),
                TeamMember(id="b", name="Bob", personality_type_id="zen_deadline"),
            ],
        )
        results = boss_compatibility_for_team(config)
        assert len(results) == 2
        assert all(isinstance(r, CompatibilityScore) for r in results)

    def test_peer_matrix_empty(self):
        assert peer_compatibility_matrix([]) == []

    def test_peer_matrix_two_members(self):
        members = [
            TeamMember(id="a", name="A", personality_type_id="strategic_charger"),
            TeamMember(id="b", name="B", personality_type_id="blitz_executor"),
        ]
        results = peer_compatibility_matrix(members)
        assert len(results) == 1  # upper-triangle only

    def test_unknown_boss_returns_empty(self):
        config = TeamConfig(
            boss_type_id="nonexistent",
            members=[TeamMember(id="a", name="A", personality_type_id="strategic_charger")],
        )
        assert boss_compatibility_for_team(config) == []

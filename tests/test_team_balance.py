"""Tests for lib_custom/engine/team_balance.py."""

from lib_custom.engine.team_balance import calculate_team_balance
from lib_custom.personality_types import TeamMember


class TestTeamBalance:
    def test_empty_team(self):
        result = calculate_team_balance([])
        assert result.diversity_score == 0.0
        # Still returns 3 dimension distributions, all with zero counts
        assert len(result.distributions) == 3

    def test_single_member(self):
        members = [TeamMember(id="a", name="A", personality_type_id="strategic_charger")]
        result = calculate_team_balance(members)
        # Single member → entropy 0 for 2-value dims, 0 for 3-value dim
        assert result.diversity_score == 0.0

    def test_two_complementary_members(self):
        """Two members covering different values should have higher diversity."""
        members = [
            TeamMember(id="a", name="A", personality_type_id="strategic_charger"),  # high, early, future
            TeamMember(id="b", name="B", personality_type_id="zen_deadline"),         # low, deadline, present
        ]
        result = calculate_team_balance(members)
        assert result.diversity_score > 50.0  # Should be fairly diverse

    def test_identical_members(self):
        """All same type → entropy 0."""
        members = [
            TeamMember(id=f"m{i}", name=f"M{i}", personality_type_id="strategic_charger")
            for i in range(5)
        ]
        result = calculate_team_balance(members)
        assert result.diversity_score == 0.0

    def test_missing_values_detected(self):
        """If no member covers 'present', it should appear in missing_values."""
        members = [
            TeamMember(id="a", name="A", personality_type_id="strategic_charger"),   # future
            TeamMember(id="b", name="B", personality_type_id="planned_pusher"),       # future
        ]
        result = calculate_team_balance(members)
        assert "present" in result.missing_values.get("time_orientation", [])

    def test_groupthink_warning(self):
        """80%+ same type → warning."""
        members = [
            TeamMember(id=f"m{i}", name=f"M{i}", personality_type_id="strategic_charger")
            for i in range(5)
        ]
        result = calculate_team_balance(members)
        assert any("群体思维" in w for w in result.warnings)

    def test_no_future_warning(self):
        """All present-oriented → strategic blind spot warning."""
        members = [
            TeamMember(id="a", name="A", personality_type_id="blitz_executor"),        # present
            TeamMember(id="b", name="B", personality_type_id="high_pressure_steady"),   # present
        ]
        result = calculate_team_balance(members)
        assert any("战略盲区" in w for w in result.warnings)

    def test_full_coverage_no_missing(self):
        """Team covering all dims → no missing."""
        members = [
            TeamMember(id="a", name="A", personality_type_id="strategic_charger"),    # high, early, future
            TeamMember(id="b", name="B", personality_type_id="high_pressure_steady"), # high, steady, present
            TeamMember(id="c", name="C", personality_type_id="zen_deadline"),          # low, deadline, present
            TeamMember(id="d", name="D", personality_type_id="calm_preemptive"),      # low, early, future
        ]
        result = calculate_team_balance(members)
        assert result.missing_values == {}

    def test_unknown_personality_ignored(self):
        members = [
            TeamMember(id="a", name="A", personality_type_id="nonexistent"),
            TeamMember(id="b", name="B", personality_type_id="strategic_charger"),
        ]
        result = calculate_team_balance(members)
        # Should not crash, just skip unknown
        assert result.diversity_score >= 0

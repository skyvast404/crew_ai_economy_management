"""Tests for default team configuration."""

from lib_custom.default_team import (
    DEFAULT_TEAM_MEMBERS,
    create_default_team,
    get_all_personality_type_ids,
    get_member_personality_name,
)
from lib_custom.personality_types import PERSONALITY_TYPES


class TestDefaultTeamMembers:
    def test_twelve_members(self):
        assert len(DEFAULT_TEAM_MEMBERS) == 12

    def test_unique_ids(self):
        ids = [m.id for m in DEFAULT_TEAM_MEMBERS]
        assert len(ids) == len(set(ids))

    def test_unique_names(self):
        names = [m.name for m in DEFAULT_TEAM_MEMBERS]
        assert len(names) == len(set(names))

    def test_unique_personality_types(self):
        type_ids = [m.personality_type_id for m in DEFAULT_TEAM_MEMBERS]
        assert len(type_ids) == len(set(type_ids))

    def test_all_twelve_personality_types_covered(self):
        covered = get_all_personality_type_ids()
        assert covered == set(PERSONALITY_TYPES.keys())

    def test_all_personality_types_valid(self):
        for member in DEFAULT_TEAM_MEMBERS:
            assert member.personality_type_id in PERSONALITY_TYPES, (
                f"Unknown personality type: {member.personality_type_id}"
            )

    def test_orders_are_sequential(self):
        orders = [m.order for m in DEFAULT_TEAM_MEMBERS]
        assert orders == list(range(1, 13))


class TestCreateDefaultTeam:
    def test_create_with_time_master(self):
        team = create_default_team("time_master")
        assert team.boss_type_id == "time_master"
        assert len(team.members) == 12

    def test_create_with_time_chaos(self):
        team = create_default_team("time_chaos")
        assert team.boss_type_id == "time_chaos"
        assert len(team.members) == 12

    def test_immutability(self):
        team_a = create_default_team("time_master")
        team_b = create_default_team("time_chaos")
        assert team_a.boss_type_id != team_b.boss_type_id
        assert team_a.members == team_b.members


class TestGetMemberPersonalityName:
    def test_known_member(self):
        member = DEFAULT_TEAM_MEMBERS[0]
        name = get_member_personality_name(member)
        assert name == "战略冲锋手"

    def test_all_members_have_names(self):
        for member in DEFAULT_TEAM_MEMBERS:
            name = get_member_personality_name(member)
            assert len(name) > 0

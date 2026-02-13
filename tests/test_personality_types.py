"""Tests for lib_custom/personality_types.py — model validation & 12-type completeness."""

from lib_custom.personality_types import (
    BOSS_TYPES,
    PERSONALITY_TYPES,
    BossType,
    PersonalityDimensions,
    PersonalityType,
    TeamConfig,
    TeamMember,
    get_boss_type,
    get_member_dimensions,
    get_personality_type,
)
from pydantic import ValidationError
import pytest


class TestPersonalityDimensions:
    def test_valid_dimensions(self):
        dim = PersonalityDimensions(urgency="high", action_pattern="early", time_orientation="future")
        assert dim.urgency == "high"
        assert dim.action_pattern == "early"
        assert dim.time_orientation == "future"

    def test_invalid_urgency(self):
        with pytest.raises(ValidationError):
            PersonalityDimensions(urgency="medium", action_pattern="early", time_orientation="future")

    def test_invalid_action_pattern(self):
        with pytest.raises(ValidationError):
            PersonalityDimensions(urgency="high", action_pattern="fast", time_orientation="future")


class TestPersonalityType:
    def test_valid_type(self):
        pt = PersonalityType(
            id="test",
            name_zh="测试",
            dimensions=PersonalityDimensions(urgency="high", action_pattern="early", time_orientation="future"),
            description="测试性格类型描述",
            strengths=["快速"],
            weaknesses=["粗心"],
        )
        assert pt.id == "test"
        assert pt.name_zh == "测试"

    def test_id_required(self):
        with pytest.raises(ValidationError):
            PersonalityType(
                id="",
                name_zh="测试",
                dimensions=PersonalityDimensions(urgency="high", action_pattern="early", time_orientation="future"),
                description="测试性格类型描述",
                strengths=["快速"],
                weaknesses=["粗心"],
            )


class TestBossType:
    def test_valid_boss(self):
        bt = BossType(
            id="test_boss",
            name_zh="测试老板",
            description="一个测试用的老板类型",
            traits=["清晰"],
            management_style="结构化管理, 按部就班",
        )
        assert bt.id == "test_boss"

    def test_traits_required(self):
        with pytest.raises(ValidationError):
            BossType(
                id="test_boss",
                name_zh="测试老板",
                description="一个测试用的老板类型",
                traits=[],  # min_length=1
                management_style="结构化管理风格描述",
            )


class TestTeamMemberAndConfig:
    def test_team_member(self):
        m = TeamMember(id="a1", name="Alice", personality_type_id="strategic_charger")
        assert m.order == 0

    def test_team_config(self):
        cfg = TeamConfig(
            boss_type_id="time_master",
            members=[
                TeamMember(id="a1", name="Alice", personality_type_id="strategic_charger"),
            ],
        )
        assert len(cfg.members) == 1


class TestPreDefinedTypes:
    def test_twelve_personality_types(self):
        assert len(PERSONALITY_TYPES) == 12

    def test_unique_ids(self):
        ids = [pt.id for pt in PERSONALITY_TYPES.values()]
        assert len(ids) == len(set(ids))

    def test_all_types_have_required_fields(self):
        for pid, pt in PERSONALITY_TYPES.items():
            assert pt.id == pid
            assert len(pt.name_zh) > 0
            assert len(pt.description) >= 5
            assert len(pt.strengths) >= 1
            assert len(pt.weaknesses) >= 1

    def test_dimension_coverage(self):
        """All 12 combos of 2x3x2 should exist."""
        combos = set()
        for pt in PERSONALITY_TYPES.values():
            d = pt.dimensions
            combos.add((d.urgency, d.action_pattern, d.time_orientation))
        assert len(combos) == 12

    def test_two_boss_types(self):
        assert len(BOSS_TYPES) == 2
        assert "time_master" in BOSS_TYPES
        assert "time_chaos" in BOSS_TYPES


class TestLookupHelpers:
    def test_get_personality_type_found(self):
        pt = get_personality_type("strategic_charger")
        assert pt is not None
        assert pt.name_zh == "战略冲锋手"

    def test_get_personality_type_not_found(self):
        assert get_personality_type("nonexistent") is None

    def test_get_boss_type_found(self):
        bt = get_boss_type("time_master")
        assert bt is not None

    def test_get_boss_type_not_found(self):
        assert get_boss_type("nonexistent") is None

    def test_get_member_dimensions(self):
        m = TeamMember(id="x", name="X", personality_type_id="blitz_executor")
        dims = get_member_dimensions(m)
        assert dims is not None
        assert dims.urgency == "high"
        assert dims.action_pattern == "early"
        assert dims.time_orientation == "present"

    def test_get_member_dimensions_invalid_type(self):
        m = TeamMember(id="x", name="X", personality_type_id="nonexistent")
        assert get_member_dimensions(m) is None

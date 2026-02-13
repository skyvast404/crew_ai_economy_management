"""Tests for thesis crew builder — personality-to-role bridging."""


from lib_custom.default_team import DEFAULT_TEAM_MEMBERS
from lib_custom.leadership_styles import get_leadership_styles_for_boss
from lib_custom.okr_models import DEFAULT_OKRS
from lib_custom.personality_types import BOSS_TYPES, PERSONALITY_TYPES
from lib_custom.thesis_crew_builder import (
    build_boss_role,
    build_evaluator_prompt,
    build_evaluator_role,
    personality_to_role,
)


class TestPersonalityToRole:
    def test_converts_all_twelve_types(self):
        """Every personality type should produce a valid RoleConfig."""
        for member in DEFAULT_TEAM_MEMBERS:
            ptype = PERSONALITY_TYPES[member.personality_type_id]
            role = personality_to_role(member, ptype)
            assert role.role_id == f"member_{member.id}"
            assert role.role_type == "conversation"
            assert member.name in role.role_name
            assert ptype.name_zh in role.role_name

    def test_role_has_personality_fields(self):
        member = DEFAULT_TEAM_MEMBERS[0]
        ptype = PERSONALITY_TYPES[member.personality_type_id]
        role = personality_to_role(member, ptype)
        assert role.personality is not None
        assert role.communication_style is not None
        assert role.emotional_tendency is not None
        assert role.values is not None

    def test_role_goal_reflects_dimensions(self):
        member = DEFAULT_TEAM_MEMBERS[0]
        ptype = PERSONALITY_TYPES["strategic_charger"]
        role = personality_to_role(member, ptype)
        # strategic_charger is high urgency → goal should mention urgency
        assert "紧迫感强" in role.goal

    def test_role_backstory_contains_strengths(self):
        member = DEFAULT_TEAM_MEMBERS[0]
        ptype = PERSONALITY_TYPES[member.personality_type_id]
        role = personality_to_role(member, ptype)
        assert "优势" in role.backstory
        assert "劣势" in role.backstory

    def test_role_preserves_personality_type_id(self):
        member = DEFAULT_TEAM_MEMBERS[0]
        ptype = PERSONALITY_TYPES[member.personality_type_id]
        role = personality_to_role(member, ptype)
        assert role.personality_type_id == member.personality_type_id

    def test_avatar_matches_personality_icon(self):
        member = DEFAULT_TEAM_MEMBERS[0]
        ptype = PERSONALITY_TYPES[member.personality_type_id]
        role = personality_to_role(member, ptype)
        assert role.avatar == ptype.icon


class TestBuildBossRole:
    def test_time_master_boss(self):
        boss_type = BOSS_TYPES["time_master"]
        styles = get_leadership_styles_for_boss("time_master")
        role = build_boss_role(boss_type, styles[0])
        assert role.role_id == "boss"
        assert role.role_type == "conversation"
        assert boss_type.name_zh in role.role_name

    def test_time_chaos_boss(self):
        boss_type = BOSS_TYPES["time_chaos"]
        styles = get_leadership_styles_for_boss("time_chaos")
        role = build_boss_role(boss_type, styles[0])
        assert role.role_id == "boss"
        assert "混乱" in role.backstory or "time_chaos" in role.backstory or boss_type.name_zh in role.backstory

    def test_boss_has_leadership_style_attributes(self):
        boss_type = BOSS_TYPES["time_master"]
        styles = get_leadership_styles_for_boss("time_master")
        style = styles[0]
        role = build_boss_role(boss_type, style)
        assert role.personality == style.boss_personality
        assert role.communication_style == style.boss_communication_style


class TestBuildEvaluatorRole:
    def test_evaluator_role_type(self):
        role = build_evaluator_role()
        assert role.role_type == "analyst"
        assert role.role_id == "evaluator"

    def test_evaluator_has_goal(self):
        role = build_evaluator_role()
        assert "评估" in role.goal

    def test_evaluator_has_backstory(self):
        role = build_evaluator_role()
        assert "组织行为学" in role.backstory


class TestBuildEvaluatorPrompt:
    def test_prompt_contains_okr(self):
        okrs = DEFAULT_OKRS["urgent_launch"]
        prompt = build_evaluator_prompt(okrs, "测试对话记录")
        assert okrs.objective in prompt

    def test_prompt_contains_conversation(self):
        okrs = DEFAULT_OKRS["urgent_launch"]
        prompt = build_evaluator_prompt(okrs, "这是一段测试对话")
        assert "这是一段测试对话" in prompt

    def test_prompt_contains_all_dimensions(self):
        okrs = DEFAULT_OKRS["urgent_launch"]
        prompt = build_evaluator_prompt(okrs, "对话")
        for dim in ["任务完成质量", "协作效率", "决策质量", "创新表现",
                     "团队士气", "沟通有效性", "风险应对", "目标对齐度"]:
            assert dim in prompt

    def test_prompt_requests_json(self):
        okrs = DEFAULT_OKRS["urgent_launch"]
        prompt = build_evaluator_prompt(okrs, "对话")
        assert "JSON" in prompt
        assert "dimensions" in prompt
        assert "overall_score" in prompt

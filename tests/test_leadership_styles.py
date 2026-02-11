"""Unit tests for leadership_styles.py"""

import pytest
from pydantic import ValidationError

from lib_custom.leadership_styles import (
    LEADERSHIP_STYLES,
    LeadershipStyle,
    apply_style_to_roles,
    DEFAULT_COMPARISON_ANALYST_PROMPT,
)
from lib_custom.role_models import create_default_roles


class TestLeadershipStyle:
    """Test LeadershipStyle model validation."""

    def test_valid_style(self):
        """Test creating a valid leadership style."""
        style = LeadershipStyle(
            style_id="test",
            style_name="测试风格",
            description="这是一个测试用的领导风格描述",
            boss_goal="测试目标描述内容，确保长度足够",
            boss_backstory="测试背景故事描述内容，确保长度足够",
            boss_personality="测试性格",
            boss_communication_style="测试沟通",
            boss_emotional_tendency="测试情绪",
            boss_values="测试价值观",
        )
        assert style.style_id == "test"
        assert style.style_name == "测试风格"

    def test_style_id_required(self):
        """Test style_id is required and non-empty."""
        with pytest.raises(ValidationError):
            LeadershipStyle(
                style_id="",
                style_name="测试风格",
                description="这是一个测试用的领导风格描述",
                boss_goal="测试目标描述内容",
                boss_backstory="测试背景故事描述内容",
                boss_personality="测试性格",
                boss_communication_style="测试沟通",
                boss_emotional_tendency="测试情绪",
                boss_values="测试价值观",
            )

    def test_description_min_length(self):
        """Test description must be at least 10 characters."""
        with pytest.raises(ValidationError):
            LeadershipStyle(
                style_id="test",
                style_name="测试风格",
                description="短",
                boss_goal="测试目标描述内容",
                boss_backstory="测试背景故事描述内容",
                boss_personality="测试性格",
                boss_communication_style="测试沟通",
                boss_emotional_tendency="测试情绪",
                boss_values="测试价值观",
            )


class TestLeadershipStylePresets:
    """Test the 4 preset leadership styles."""

    def test_four_styles_defined(self):
        """Test that exactly 4 styles are defined."""
        assert len(LEADERSHIP_STYLES) == 4

    def test_expected_style_ids(self):
        """Test expected style IDs exist."""
        expected = {"transformational", "transactional", "servant", "authoritative"}
        assert set(LEADERSHIP_STYLES.keys()) == expected

    def test_all_styles_have_required_fields(self):
        """Test all preset styles have non-empty required fields."""
        for style_id, style in LEADERSHIP_STYLES.items():
            assert style.style_id == style_id
            assert len(style.style_name) > 0
            assert len(style.description) >= 10
            assert len(style.boss_goal) >= 10
            assert len(style.boss_backstory) >= 10

    def test_styles_are_distinct(self):
        """Test that style names are unique."""
        names = [s.style_name for s in LEADERSHIP_STYLES.values()]
        assert len(names) == len(set(names))


class TestApplyStyleToRoles:
    """Test apply_style_to_roles immutability and correctness."""

    @pytest.fixture
    def base_db(self):
        return create_default_roles()

    def test_returns_new_database(self, base_db):
        """Test that a new RolesDatabase is returned, not mutated."""
        style = LEADERSHIP_STYLES["transformational"]
        new_db = apply_style_to_roles(base_db, style)
        assert new_db is not base_db

    def test_boss_attributes_overridden(self, base_db):
        """Test boss role gets style attributes."""
        style = LEADERSHIP_STYLES["transformational"]
        new_db = apply_style_to_roles(base_db, style)
        boss = next(r for r in new_db.roles if r.role_id == "boss")
        assert boss.goal == style.boss_goal
        assert boss.backstory == style.boss_backstory
        assert boss.personality == style.boss_personality
        assert boss.communication_style == style.boss_communication_style
        assert boss.emotional_tendency == style.boss_emotional_tendency
        assert boss.values == style.boss_values

    def test_original_db_unchanged(self, base_db):
        """Test original database is not mutated."""
        original_boss = next(r for r in base_db.roles if r.role_id == "boss")
        original_goal = original_boss.goal

        style = LEADERSHIP_STYLES["authoritative"]
        apply_style_to_roles(base_db, style)

        boss_after = next(r for r in base_db.roles if r.role_id == "boss")
        assert boss_after.goal == original_goal

    def test_non_boss_roles_unchanged(self, base_db):
        """Test non-boss roles are not affected."""
        style = LEADERSHIP_STYLES["servant"]
        new_db = apply_style_to_roles(base_db, style)

        for orig, new in zip(base_db.roles, new_db.roles):
            if orig.role_id != "boss":
                assert orig.goal == new.goal
                assert orig.backstory == new.backstory

    def test_role_count_preserved(self, base_db):
        """Test number of roles stays the same."""
        style = LEADERSHIP_STYLES["transactional"]
        new_db = apply_style_to_roles(base_db, style)
        assert len(new_db.roles) == len(base_db.roles)


class TestComparisonPrompt:
    """Test the comparison analyst prompt template."""

    def test_prompt_has_placeholders(self):
        """Test prompt contains required placeholders."""
        assert "{topic}" in DEFAULT_COMPARISON_ANALYST_PROMPT
        assert "{style_conversations}" in DEFAULT_COMPARISON_ANALYST_PROMPT

    def test_prompt_can_be_formatted(self):
        """Test prompt can be formatted with sample data."""
        result = DEFAULT_COMPARISON_ANALYST_PROMPT.format(
            topic="测试话题",
            style_conversations="对话内容",
        )
        assert "测试话题" in result
        assert "对话内容" in result

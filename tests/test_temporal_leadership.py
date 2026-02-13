"""Tests for lib_custom/temporal_leadership.py."""

import pytest

from lib_custom.temporal_leadership import (
    TTL_CODE_MAP,
    TTL_LEVELS,
    TemporalLeadershipConfig,
    build_ttl_boss_role,
    get_ttl_config,
)


# ---------------------------------------------------------------------------
# get_ttl_config
# ---------------------------------------------------------------------------
class TestGetTtlConfig:
    """Tests for get_ttl_config()."""

    def test_returns_config_for_each_level(self):
        for level in TTL_LEVELS:
            config = get_ttl_config(level)
            assert isinstance(config, TemporalLeadershipConfig)
            assert config.level == level

    def test_code_matches_map(self):
        for level in TTL_LEVELS:
            config = get_ttl_config(level)
            assert config.code == TTL_CODE_MAP[level]

    def test_high_has_code_1(self):
        config = get_ttl_config("high")
        assert config.code == 1.0

    def test_medium_has_code_05(self):
        config = get_ttl_config("medium")
        assert config.code == 0.5

    def test_low_has_code_0(self):
        config = get_ttl_config("low")
        assert config.code == 0.0

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError, match="Unknown TTL level"):
            get_ttl_config("extreme")

    def test_all_configs_have_nonempty_fields(self):
        for level in TTL_LEVELS:
            config = get_ttl_config(level)
            assert len(config.label) > 0
            assert len(config.time_framing) > 0
            assert len(config.milestone_mgmt) > 0
            assert len(config.pacing_sync) > 0
            assert len(config.overall_instruction) > 0

    def test_high_mentions_time(self):
        config = get_ttl_config("high")
        assert "时间" in config.time_framing or "deadline" in config.time_framing.lower()

    def test_low_mentions_not_time(self):
        config = get_ttl_config("low")
        assert "不" in config.time_framing or "几乎" in config.time_framing


# ---------------------------------------------------------------------------
# build_ttl_boss_role
# ---------------------------------------------------------------------------
class TestBuildTtlBossRole:
    """Tests for build_ttl_boss_role()."""

    def test_returns_role_config(self):
        config = get_ttl_config("high")
        role = build_ttl_boss_role(config)
        assert role.role_id == "boss"
        assert role.role_type == "conversation"

    def test_role_name_includes_ttl_label(self):
        config = get_ttl_config("medium")
        role = build_ttl_boss_role(config)
        assert "中 TTL" in role.role_name

    def test_backstory_includes_ttl_behaviors(self):
        config = get_ttl_config("high")
        role = build_ttl_boss_role(config)
        assert "时间领导力行为" in role.backstory

    def test_custom_base_name(self):
        config = get_ttl_config("low")
        role = build_ttl_boss_role(config, base_name="测试负责人")
        assert "测试负责人" in role.role_name

    def test_neutral_personality_no_leadership_style(self):
        config = get_ttl_config("high")
        role = build_ttl_boss_role(config)
        # Should NOT contain leadership style references
        assert "变革型" not in role.backstory
        assert "交易型" not in role.backstory
        assert "服务型" not in role.backstory
        assert "权威型" not in role.backstory

    def test_order_is_zero(self):
        config = get_ttl_config("medium")
        role = build_ttl_boss_role(config)
        assert role.order == 0

    def test_immutability(self):
        config = get_ttl_config("high")
        role1 = build_ttl_boss_role(config)
        role2 = build_ttl_boss_role(config)
        assert role1.role_name == role2.role_name
        assert role1.backstory == role2.backstory

    def test_all_levels_produce_valid_roles(self):
        for level in TTL_LEVELS:
            config = get_ttl_config(level)
            role = build_ttl_boss_role(config)
            assert role.role_id == "boss"
            assert len(role.goal) >= 10
            assert len(role.backstory) >= 10


# ---------------------------------------------------------------------------
# TTL_LEVELS / TTL_CODE_MAP
# ---------------------------------------------------------------------------
class TestTtlConstants:
    """Tests for TTL constants."""

    def test_three_levels(self):
        assert len(TTL_LEVELS) == 3

    def test_levels_are_low_medium_high(self):
        assert set(TTL_LEVELS) == {"low", "medium", "high"}

    def test_code_map_values(self):
        assert TTL_CODE_MAP["low"] == 0.0
        assert TTL_CODE_MAP["medium"] == 0.5
        assert TTL_CODE_MAP["high"] == 1.0

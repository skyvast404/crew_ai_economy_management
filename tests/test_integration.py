"""Integration tests for role management system."""

import tempfile
from pathlib import Path

import pytest

from lib_custom.crew_builder import CrewBuilder
from lib_custom.role_models import RoleConfig
from lib_custom.role_repository import RoleRepository


class TestRoleManagementIntegration:
    """Test integration of role management components."""

    @pytest.fixture
    def temp_config(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name
        Path(temp_path).unlink()
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
        Path(f"{temp_path}.backup").unlink(missing_ok=True)

    def test_load_and_build_crew(self, temp_config):
        """Test loading roles and building crew."""
        repo = RoleRepository(temp_config)
        db = repo.load_roles()
        builder = CrewBuilder(db)

        crew = builder.build_crew("测试主题", num_rounds=2)

        assert len(crew.agents) == 4
        assert len(crew.tasks) == 7

    def test_add_custom_role_and_build(self, temp_config):
        """Test adding custom role and building crew."""
        repo = RoleRepository(temp_config)

        # Add custom role
        custom_role = RoleConfig(
            role_id="custom",
            role_name="自定义角色",
            goal="测试自定义角色的目标描述",
            backstory="测试自定义角色的背景故事描述",
            avatar="🎪",
            role_type="conversation",
            order=10,
        )
        repo.add_role(custom_role)

        # Build crew with custom role
        db = repo.load_roles()
        builder = CrewBuilder(db)
        crew = builder.build_crew("测试主题", num_rounds=1)

        # Should have 4 conversation agents + analyst
        assert len(crew.agents) == 5
        assert len(crew.tasks) == 5

    def test_custom_prompt_templates(self, temp_config):
        """Test custom prompt templates are used."""
        repo = RoleRepository(temp_config)

        # Update role with custom prompt
        custom_prompt = "自定义提示词: {topic}"
        repo.update_role("boss", {"round_1_prompt": custom_prompt})

        # Build crew
        db = repo.load_roles()
        builder = CrewBuilder(db)
        crew = builder.build_crew("测试主题", num_rounds=1)

        # First task should use custom prompt
        first_task = crew.tasks[0]
        assert "自定义提示词" in first_task.description

    def test_delete_and_rebuild(self, temp_config):
        """Test deleting custom role and rebuilding crew."""
        repo = RoleRepository(temp_config)

        # Add and then delete custom role
        custom_role = RoleConfig(
            role_id="temp_role",
            role_name="临时角色",
            goal="临时角色的测试目标描述",
            backstory="临时角色的测试背景故事",
            avatar="🎯",
            role_type="conversation",
        )
        repo.add_role(custom_role)
        repo.delete_role("temp_role")

        # Should still be able to build crew
        db = repo.load_roles()
        builder = CrewBuilder(db)
        crew = builder.build_crew("测试主题", num_rounds=1)

        assert len(crew.agents) == 4

    def test_reset_and_rebuild(self, temp_config):
        """Test resetting to defaults and rebuilding crew."""
        repo = RoleRepository(temp_config)

        # Modify and then reset
        repo.update_role("boss", {"goal": "修改后的目标描述内容"})
        repo.reset_to_defaults()

        # Should have default configuration
        db = repo.load_roles()
        builder = CrewBuilder(db)
        crew = builder.build_crew("测试主题", num_rounds=1)

        assert len(crew.agents) == 4
        assert len(crew.tasks) == 4

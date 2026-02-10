"""Unit tests for crew_builder.py"""

import pytest

from lib_custom.crew_builder import CrewBuilder
from lib_custom.role_models import RoleConfig, create_default_roles


class TestCrewBuilder:
    """Test CrewBuilder functionality."""

    @pytest.fixture
    def roles_db(self):
        """Create default roles database."""
        return create_default_roles()

    @pytest.fixture
    def builder(self, roles_db):
        """Create CrewBuilder instance."""
        return CrewBuilder(roles_db)

    def test_build_agent(self, builder, roles_db):
        """Test building an agent from role config."""
        role = roles_db.roles[0]
        agent = builder.build_agent(role)

        assert agent.role == role.role_name
        assert agent.goal == role.goal
        assert agent.backstory == role.backstory
        assert agent.verbose is False
        assert agent.allow_delegation is False

    def test_create_conversation_task_round_1(self, builder, roles_db):
        """Test creating conversation task for round 1."""
        role = roles_db.get_conversation_roles()[0]
        agent = builder.build_agent(role)

        task = builder.create_conversation_task(
            agent, role, "测试主题", 1, []
        )

        assert task.agent == agent
        assert "测试主题" in task.description
        assert task.expected_output.startswith("第1轮")

    def test_create_conversation_task_followup(self, builder, roles_db):
        """Test creating conversation task for followup rounds."""
        role = roles_db.get_conversation_roles()[0]
        agent = builder.build_agent(role)

        task = builder.create_conversation_task(
            agent, role, "测试主题", 2, []
        )

        assert task.agent == agent
        assert "测试主题" in task.description
        assert task.expected_output.startswith("第2轮")

    def test_create_analyst_task(self, builder, roles_db):
        """Test creating analyst task."""
        analyst_role = roles_db.get_analyst_role()
        analyst_agent = builder.build_agent(analyst_role)

        task = builder.create_analyst_task(
            analyst_agent, analyst_role, "测试主题", 3, []
        )

        assert task.agent == analyst_agent
        assert "测试主题" in task.description
        assert "结构化" in task.expected_output

    def test_build_crew(self, builder):
        """Test building complete crew."""
        crew = builder.build_crew("测试主题", num_rounds=2)

        # Should have conversation agents + analyst
        assert len(crew.agents) == 4

        # Should have 2 rounds × 3 conv agents + 1 analyst task
        assert len(crew.tasks) == 7

    def test_build_crew_no_analyst_fails(self, roles_db):
        """Test building crew fails without analyst role."""
        # Remove analyst role
        conv_only_db = type(roles_db)(
            version=roles_db.version,
            roles=[r for r in roles_db.roles if r.role_type == "conversation"]
        )
        builder = CrewBuilder(conv_only_db)

        with pytest.raises(ValueError, match="No analyst role"):
            builder.build_crew("测试主题")

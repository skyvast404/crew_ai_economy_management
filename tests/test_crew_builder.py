"""Unit tests for crew_builder.py"""

import pytest

from lib_custom.crew_builder import CrewBuilder, build_comparison_crew
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

    def test_personality_in_task_description(self, builder, roles_db):
        """Test personality fields appear in task descriptions."""
        role = roles_db.get_conversation_roles()[0]  # boss
        agent = builder.build_agent(role)

        task = builder.create_conversation_task(agent, role, "测试主题", 1, [])

        assert role.personality in task.description
        assert role.communication_style in task.description

    def test_none_personality_becomes_empty(self):
        """Test None personality fields become empty strings in prompts."""
        role = RoleConfig(
            role_id="no_personality",
            role_name="No Personality Role",
            goal="测试没有性格属性的角色",
            backstory="这个角色没有设置性格属性",
            avatar="🎭",
            role_type="conversation",
            order=0,
        )
        analyst = RoleConfig(
            role_id="analyst",
            role_name="Analyst",
            goal="从组织行为学角度分析会议讨论",
            backstory="你是组织行为学研究者，擅长分析职场互动。",
            avatar="📊",
            role_type="analyst",
            order=999,
        )
        from lib_custom.role_models import RolesDatabase
        db = RolesDatabase(roles=[role, role.model_copy(update={"role_id": "r2", "order": 1}), analyst])
        builder = CrewBuilder(db)
        agent = builder.build_agent(role)

        # Should not raise KeyError for missing personality fields
        task = builder.create_conversation_task(agent, role, "测试", 1, [])
        assert "测试" in task.description


class TestBuildComparisonCrew:
    """Test build_comparison_crew function."""

    def test_returns_crew(self):
        """Test that a Crew object is returned."""
        crew = build_comparison_crew("测试话题", {"风格A": "对话A", "风格B": "对话B"})
        assert len(crew.agents) == 1
        assert len(crew.tasks) == 1

    def test_task_contains_topic(self):
        """Test task description contains the topic."""
        crew = build_comparison_crew("项目延期", {"变革型": "内容1", "交易型": "内容2"})
        assert "项目延期" in crew.tasks[0].description

    def test_task_contains_style_conversations(self):
        """Test task description contains each style's conversation."""
        conversations = {"变革型领导": "变革对话内容", "权威型领导": "权威对话内容"}
        crew = build_comparison_crew("测试", conversations)
        desc = crew.tasks[0].description
        assert "变革型领导" in desc
        assert "权威型领导" in desc
        assert "变革对话内容" in desc
        assert "权威对话内容" in desc

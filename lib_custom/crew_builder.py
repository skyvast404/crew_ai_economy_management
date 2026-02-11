"""Crew builder for creating agents and tasks from role configurations."""

from crewai import Agent, Crew, Process, Task

from lib_custom.leadership_styles import DEFAULT_COMPARISON_ANALYST_PROMPT
from lib_custom.role_models import RoleConfig, RolesDatabase


class CrewBuilder:
    """Builds CrewAI agents and tasks from role configurations."""

    def __init__(self, roles_db: RolesDatabase):
        """Initialize with roles database."""
        self.roles_db = roles_db

    def build_agent(self, role_config: RoleConfig) -> Agent:
        """Create an Agent from RoleConfig."""
        return Agent(
            role=role_config.role_name,
            goal=role_config.goal,
            backstory=role_config.backstory,
            verbose=False,
            allow_delegation=False,
        )

    def create_conversation_task(
        self,
        agent: Agent,
        role_config: RoleConfig,
        topic: str,
        round_num: int,
        context_tasks: list[Task],
    ) -> Task:
        """Create a conversation task for a specific round."""
        if round_num == 1:
            prompt = role_config.round_1_prompt or self._default_round_1_prompt()
        else:
            prompt = role_config.followup_prompt or self._default_followup_prompt()

        description = prompt.format(
            role_name=role_config.role_name,
            goal=role_config.goal,
            backstory=role_config.backstory,
            personality=role_config.personality or "",
            communication_style=role_config.communication_style or "",
            emotional_tendency=role_config.emotional_tendency or "",
            values=role_config.values or "",
            topic=topic,
            round=round_num,
            context="",
        )

        return Task(
            description=description,
            expected_output=f"第{round_num}轮 - {role_config.role_name}的发言",
            agent=agent,
            context=context_tasks,
        )

    def create_analyst_task(
        self,
        analyst_agent: Agent,
        analyst_config: RoleConfig,
        topic: str,
        num_rounds: int,
        conversation_tasks: list[Task],
    ) -> Task:
        """Create the final analyst task."""
        prompt = analyst_config.analyst_prompt or self._default_analyst_prompt()

        description = prompt.format(
            topic=topic,
            num_rounds=num_rounds,
            full_conversation="",
        )

        return Task(
            description=description,
            expected_output="结构化的多智能体群体互动分析报告",
            agent=analyst_agent,
            context=conversation_tasks,
        )

    def build_crew(self, topic: str, num_rounds: int = 3) -> Crew:
        """Build a complete crew with conversation and analyst agents."""
        # Get conversation roles
        conv_roles = self.roles_db.get_conversation_roles()
        analyst_role = self.roles_db.get_analyst_role()

        if not analyst_role:
            raise ValueError("No analyst role found in database")

        # Create agents
        conv_agents = [self.build_agent(role) for role in conv_roles]
        analyst_agent = self.build_agent(analyst_role)

        # Create tasks
        tasks: list[Task] = []

        for round_idx in range(num_rounds):
            round_num = round_idx + 1
            for i, agent in enumerate(conv_agents):
                role_config = conv_roles[i]
                # Context: last 2 rounds (8 tasks max)
                ctx = tasks[-8:] if tasks else []
                task = self.create_conversation_task(
                    agent, role_config, topic, round_num, ctx
                )
                tasks.append(task)

        # Add analyst task
        analyst_task = self.create_analyst_task(
            analyst_agent, analyst_role, topic, num_rounds, tasks
        )
        tasks.append(analyst_task)

        return Crew(
            agents=[*conv_agents, analyst_agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )

    @staticmethod
    def _default_round_1_prompt() -> str:
        """Default prompt for round 1."""
        return """你是{role_name}。

你的目标: {goal}

你的背景: {backstory}

你的性格: {personality}
你的沟通风格: {communication_style}
你的情绪倾向: {emotional_tendency}
你的价值观: {values}

当前会议主题: {topic}

请根据你的角色定位，发表你对这个主题的初步看法。注意保持你的角色特征和说话风格。"""

    @staticmethod
    def _default_followup_prompt() -> str:
        """Default prompt for followup rounds."""
        return """你是{role_name}。

你的性格: {personality}
你的沟通风格: {communication_style}
你的情绪倾向: {emotional_tendency}
你的价值观: {values}

会议主题: {topic}

之前的讨论:
{context}

请根据之前的讨论，继续发表你的看法。注意:
1. 回应其他人的观点
2. 保持你的角色特征和沟通风格
3. 推进讨论或维护你的立场"""

    @staticmethod
    def _default_analyst_prompt() -> str:
        """Default prompt for analyst."""
        return """你是组织行为学研究者。请分析以下会议讨论:

会议主题: {topic}

完整对话记录:
{full_conversation}

请从以下4个模块进行学术分析:

模块1: 硬指标 (绩效影响)
- 任务绩效影响
- 组织绩效影响

模块2: 软指标 (行为变量)
- OCB (组织公民行为)
- 创新行为
- UPB (非伦理亲组织行为)
- 离职倾向信号

模块3: 变量关系建模
- 自变量 (IV)
- 因变量 (DV)
- 中介变量
- 调节变量
- 研究假设

模块4: 测量与方法论
- 推荐量表
- 适用理论
- 研究模型路径

请提供结构化的学术分析报告。"""


def build_comparison_crew(
    topic: str, style_conversations: dict[str, str]
) -> Crew:
    """Build a crew for cross-style comparison analysis.

    Args:
        topic: The discussion topic.
        style_conversations: Mapping of style_name -> conversation text.

    Returns:
        A Crew with a single comparison analyst agent and task.
    """
    # Format all style conversations into a single block
    sections = []
    for style_name, conversation in style_conversations.items():
        sections.append(f"--- {style_name} ---\n{conversation}\n")
    combined = "\n".join(sections)

    description = DEFAULT_COMPARISON_ANALYST_PROMPT.format(
        topic=topic,
        style_conversations=combined,
    )

    agent = Agent(
        role="跨风格对比分析师",
        goal="对比分析不同领导风格下的团队互动差异",
        backstory="你是资深组织行为学研究者，擅长跨条件对比分析和领导力研究。",
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=description,
        expected_output="结构化的跨领导风格对比分析报告",
        agent=agent,
    )

    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

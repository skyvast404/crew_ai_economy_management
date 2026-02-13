"""Thesis crew builder — bridges personality types to CrewAI roles.

Converts PersonalityType + TeamMember into RoleConfig/Agent,
builds the evaluator agent, and constructs the complete thesis experiment crew.

Supports two boss construction paths:
    1. Original: boss_type_id → leadership style (backward compatible)
    2. TTL path: TemporalLeadershipConfig → neutral boss + TTL behavior overlay
"""

from __future__ import annotations

from crewai import LLM, Agent, Crew, Process, Task

from lib_custom.leadership_styles import (
    LeadershipStyle,
    get_leadership_styles_for_boss,
)
from lib_custom.okr_models import EVALUATION_DIMENSIONS, OKRSet, format_okrs_for_prompt
from lib_custom.personality_types import (
    BOSS_TYPES,
    PERSONALITY_TYPES,
    BossType,
    PersonalityType,
    TeamConfig,
    TeamMember,
)
from lib_custom.role_models import RoleConfig
from lib_custom.temporal_leadership import TemporalLeadershipConfig, build_ttl_boss_role


# ---------------------------------------------------------------------------
# Personality → RoleConfig mapping
# ---------------------------------------------------------------------------
def _urgency_label(urgency: str) -> str:
    return "紧迫感强" if urgency == "high" else "从容不迫"


def _action_label(action: str) -> str:
    mapping = {
        "early": "提前行动型",
        "steady": "稳步推进型",
        "deadline": "截止日驱动型",
    }
    return mapping.get(action, action)


def _time_label(orientation: str) -> str:
    return "着眼未来" if orientation == "future" else "专注当下"


def personality_to_role(member: TeamMember, ptype: PersonalityType) -> RoleConfig:
    """Convert a personality type + team member into a CrewAI RoleConfig."""
    dims = ptype.dimensions

    goal = (
        f"以{_urgency_label(dims.urgency)}的节奏，"
        f"采用{_action_label(dims.action_pattern)}的方式推进工作，"
        f"{_time_label(dims.time_orientation)}"
    )

    strengths_str = "、".join(ptype.strengths)
    weaknesses_str = "、".join(ptype.weaknesses)
    backstory = (
        f"你是{member.name}，性格类型为「{ptype.name_zh}」。"
        f"{ptype.description} "
        f"你的优势是{strengths_str}；"
        f"你的劣势是{weaknesses_str}。"
    )

    personality = (
        f"{_urgency_label(dims.urgency)}、"
        f"{_action_label(dims.action_pattern)}、"
        f"{_time_label(dims.time_orientation)}"
    )
    communication_style = _derive_communication_style(dims)
    emotional_tendency = _derive_emotional_tendency(dims)
    values = _derive_values(dims)

    return RoleConfig(
        role_id=f"member_{member.id}",
        role_name=f"{ptype.icon} {member.name} ({ptype.name_zh})",
        goal=goal,
        backstory=backstory,
        avatar=ptype.icon,
        role_type="conversation",
        is_default=False,
        order=member.order,
        personality=personality,
        communication_style=communication_style,
        emotional_tendency=emotional_tendency,
        values=values,
        personality_type_id=ptype.id,
    )


def _derive_communication_style(dims) -> str:
    """Derive communication style from personality dimensions."""
    parts: list[str] = []
    if dims.urgency == "high":
        parts.append("直接高效")
    else:
        parts.append("温和耐心")
    if dims.action_pattern == "early":
        parts.append("主动推动讨论")
    elif dims.action_pattern == "steady":
        parts.append("条理清晰")
    else:
        parts.append("倾向后期发力")
    if dims.time_orientation == "future":
        parts.append("善于展望全局")
    else:
        parts.append("聚焦具体细节")
    return "、".join(parts)


def _derive_emotional_tendency(dims) -> str:
    """Derive emotional tendency from personality dimensions."""
    if dims.urgency == "high" and dims.action_pattern == "deadline":
        return "压力下爆发力强，但过程中可能焦虑"
    if dims.urgency == "high":
        return "充满干劲，容易因进度问题紧张"
    if dims.action_pattern == "deadline":
        return "心态平和，临近截止时才调动情绪"
    return "情绪稳定，不易受外界压力影响"


def _derive_values(dims) -> str:
    """Derive values from personality dimensions."""
    parts: list[str] = []
    if dims.urgency == "high":
        parts.append("效率")
    else:
        parts.append("质量")
    if dims.action_pattern == "early":
        parts.append("主动性")
    elif dims.action_pattern == "steady":
        parts.append("稳定性")
    else:
        parts.append("结果导向")
    if dims.time_orientation == "future":
        parts.append("长远规划")
    else:
        parts.append("务实落地")
    return "、".join(parts)


# ---------------------------------------------------------------------------
# Boss → RoleConfig
# ---------------------------------------------------------------------------
def build_boss_role(
    boss_type: BossType,
    style: LeadershipStyle,
) -> RoleConfig:
    """Convert a boss type + leadership style into a boss RoleConfig."""
    return RoleConfig(
        role_id="boss",
        role_name=f"👔 老板 ({boss_type.name_zh})",
        goal=style.boss_goal,
        backstory=(
            f"你是团队的领导者，管理风格为「{boss_type.name_zh}」。"
            f"{boss_type.description} "
            f"你的领导风格: {style.style_name} — {style.description}"
        ),
        avatar="👔",
        role_type="conversation",
        is_default=False,
        order=0,
        personality=style.boss_personality,
        communication_style=style.boss_communication_style,
        emotional_tendency=style.boss_emotional_tendency,
        values=style.boss_values,
    )


# ---------------------------------------------------------------------------
# Evaluator Agent
# ---------------------------------------------------------------------------
def build_evaluator_role() -> RoleConfig:
    """Create an independent, objective performance evaluator role."""
    return RoleConfig(
        role_id="evaluator",
        role_name="📊 绩效评估专家",
        goal="客观公正地评估团队绩效，基于OKR目标和讨论记录进行量化评分",
        backstory=(
            "你是一位独立客观的组织行为学研究者，拥有管理学博士学位，"
            "专注于领导力与团队绩效研究超过15年。"
            "你擅长从团队互动中识别绩效信号，并能给出严格、有据可循的评分。"
        ),
        avatar="📊",
        role_type="analyst",
        is_default=False,
        order=999,
    )


def build_evaluator_prompt(okrs: OKRSet, full_conversation: str) -> str:
    """Build the evaluator's task prompt with OKR context and conversation."""
    okrs_formatted = format_okrs_for_prompt(okrs)

    dim_lines: list[str] = []
    for i, dim in enumerate(EVALUATION_DIMENSIONS.values(), 1):
        weight_pct = int(dim.weight * 100)
        dim_lines.append(f"{i}. {dim.name_zh}({weight_pct}%): {dim.description}")
    dimensions_text = "\n".join(dim_lines)

    return f"""你是一位独立客观的组织行为学研究者，拥有管理学博士学位。
你需要基于以下OKR目标和团队讨论记录，对团队绩效进行严格评估。

## 团队OKR
{okrs_formatted}

## 讨论记录
{full_conversation}

## 评估要求
请从以下8个维度打分(0-100)，并给出评分依据:

{dimensions_text}

## 输出格式 (严格JSON，不要包含任何其他文字)
{{
  "dimensions": {{
    "task_completion": {{"score": 0, "evidence": "..."}},
    "collaboration": {{"score": 0, "evidence": "..."}},
    "decision_quality": {{"score": 0, "evidence": "..."}},
    "innovation": {{"score": 0, "evidence": "..."}},
    "morale": {{"score": 0, "evidence": "..."}},
    "communication": {{"score": 0, "evidence": "..."}},
    "risk_management": {{"score": 0, "evidence": "..."}},
    "goal_alignment": {{"score": 0, "evidence": "..."}}
  }},
  "overall_score": 0,
  "key_findings": ["发现1", "发现2"],
  "boss_impact_analysis": "对老板领导风格影响的分析",
  "recommendations": ["建议1", "建议2"]
}}"""


# ---------------------------------------------------------------------------
# Crew construction
# ---------------------------------------------------------------------------
_ROUND_1_PROMPT = """你是{role_name}。

你的目标: {goal}

你的背景: {backstory}

你的性格: {personality}
你的沟通风格: {communication_style}
你的情绪倾向: {emotional_tendency}
你的价值观: {values}

当前会议主题: {topic}

请根据你的角色定位，发表你对这个主题的初步看法。注意保持你的角色特征和说话风格。
发言控制在150字以内。"""

_FOLLOWUP_PROMPT = """你是{role_name}。

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
3. 推进讨论或维护你的立场
发言控制在150字以内。"""


def build_thesis_crew(
    team: TeamConfig,
    boss_type_id: str,
    topic: str,
    okrs: OKRSet,
    num_rounds: int = 3,
    llm: LLM | None = None,
    config: dict | None = None,
    ttl_config: TemporalLeadershipConfig | None = None,
) -> Crew:
    """Build a complete crew for thesis experiments.

    Creates:
    - N employee agents (conversation roles mapped from personality types)
    - 1 boss agent (leadership-style path OR TTL path)
    - 1 evaluator agent (receives OKR + full transcript, outputs JSON scores)

    Two boss construction paths:
        - ttl_config provided → neutral boss + TTL behavior overlay (decoupled)
        - ttl_config is None → original boss_type_id → leadership style path

    Args:
        team: Team configuration (variable size, not restricted to 12).
        boss_type_id: "time_master" or "time_chaos" (used in original path).
        topic: Discussion topic.
        okrs: OKR set for evaluation context.
        num_rounds: Number of conversation rounds.
        llm: Optional LLM override.
        config: Optional config dict (agent_timeout, max_iterations, context_window).
        ttl_config: Optional TTL configuration. When provided, boss is built
            with neutral persona + TTL overlay (bypasses leadership styles).

    Returns:
        A fully constructed Crew ready to kickoff.
    """
    cfg = config or {
        "agent_timeout": 120,
        "max_iterations": 5,
        "context_window": 4,
    }

    # --- Build member roles (variable team size) ---
    member_roles: list[RoleConfig] = []
    for member in team.members:
        ptype = PERSONALITY_TYPES.get(member.personality_type_id)
        if ptype is None:
            continue
        member_roles.append(personality_to_role(member, ptype))

    # --- Build boss role (TTL path vs original path) ---
    if ttl_config is not None:
        boss_role = build_ttl_boss_role(ttl_config)
    else:
        boss_type = BOSS_TYPES.get(boss_type_id)
        if boss_type is None:
            raise ValueError(f"Unknown boss type: {boss_type_id}")
        styles = get_leadership_styles_for_boss(boss_type_id)
        if not styles:
            raise ValueError(f"No leadership styles found for boss: {boss_type_id}")
        boss_role = build_boss_role(boss_type, styles[0])

    # --- Build evaluator role ---
    evaluator_role = build_evaluator_role()

    # --- Create agents ---
    def _make_agent(role: RoleConfig) -> Agent:
        kwargs: dict = {
            "role": role.role_name,
            "goal": role.goal,
            "backstory": role.backstory,
            "verbose": False,
            "allow_delegation": False,
            "max_iter": cfg["max_iterations"],
            "max_execution_time": cfg["agent_timeout"],
        }
        if llm is not None:
            kwargs["llm"] = llm
        return Agent(**kwargs)

    boss_agent = _make_agent(boss_role)
    member_agents = [_make_agent(r) for r in member_roles]
    evaluator_agent = _make_agent(evaluator_role)

    # All conversation roles: boss first, then members
    all_conv_roles = [boss_role, *member_roles]
    all_conv_agents = [boss_agent, *member_agents]

    # --- Create conversation tasks ---
    tasks: list[Task] = []
    context_size = cfg["context_window"]

    for round_idx in range(num_rounds):
        round_num = round_idx + 1
        for i, agent in enumerate(all_conv_agents):
            role = all_conv_roles[i]
            prompt = _ROUND_1_PROMPT if round_num == 1 else _FOLLOWUP_PROMPT
            description = prompt.format(
                role_name=role.role_name,
                goal=role.goal,
                backstory=role.backstory,
                personality=role.personality or "",
                communication_style=role.communication_style or "",
                emotional_tendency=role.emotional_tendency or "",
                values=role.values or "",
                topic=topic,
                round=round_num,
                context="",
            )
            ctx = tasks[-context_size:] if tasks else []
            task = Task(
                description=description,
                expected_output=f"第{round_num}轮 - {role.role_name}的发言",
                agent=agent,
                context=ctx,
            )
            tasks.append(task)

    # --- Create evaluator task ---
    # The evaluator sees all conversation tasks as context.
    # The prompt will be filled with the full conversation during execution,
    # but we pass a placeholder now and rely on context passing.
    evaluator_prompt = build_evaluator_prompt(okrs, "(完整对话记录将通过上下文自动提供)")

    evaluator_task = Task(
        description=evaluator_prompt,
        expected_output="严格JSON格式的8维度团队绩效评估报告",
        agent=evaluator_agent,
        context=tasks,  # evaluator sees ALL conversation tasks
    )
    tasks.append(evaluator_task)

    return Crew(
        agents=[*all_conv_agents, evaluator_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
        max_rpm=10,
        stream=bool(cfg.get("stream", True)),
    )

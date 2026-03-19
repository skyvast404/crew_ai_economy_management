"""Personality type definitions for team management dashboard.

Defines 12 employee personality types (3 dimensions: urgency × action_pattern × time_orientation)
and 3 boss management types (time_master / time_chaos / time_neutral).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dimension enums
# ---------------------------------------------------------------------------
UrgencyLevel = Literal["high", "low"]
ActionPattern = Literal["early", "steady", "deadline"]
TimeOrientation = Literal["future", "present"]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class PersonalityDimensions(BaseModel):
    """Three-axis personality dimensions."""

    urgency: UrgencyLevel
    action_pattern: ActionPattern
    time_orientation: TimeOrientation


class PersonalityType(BaseModel):
    """A single employee personality type."""

    id: str = Field(..., min_length=1, max_length=60)
    name_zh: str = Field(..., min_length=1, max_length=30)
    dimensions: PersonalityDimensions
    description: str = Field(..., min_length=5)
    strengths: list[str] = Field(default_factory=list, min_length=1)
    weaknesses: list[str] = Field(default_factory=list, min_length=1)
    color: str = Field(default="#4A90D9")
    icon: str = Field(default="👤", min_length=1, max_length=2)


class BossType(BaseModel):
    """A boss management style type."""

    id: str = Field(..., min_length=1, max_length=60)
    name_zh: str = Field(..., min_length=1, max_length=30)
    description: str = Field(..., min_length=5)
    traits: list[str] = Field(default_factory=list, min_length=1)
    management_style: str = Field(..., min_length=5)


class TeamMember(BaseModel):
    """A single team member with assigned personality type."""

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=50)
    personality_type_id: str = Field(..., min_length=1)
    order: int = Field(default=0, ge=0)


class TeamConfig(BaseModel):
    """Full team configuration: boss type + member list."""

    boss_type_id: str = Field(..., min_length=1)
    members: list[TeamMember] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pre-defined 12 employee personality types
# ---------------------------------------------------------------------------
PERSONALITY_TYPES: dict[str, PersonalityType] = {
    "strategic_charger": PersonalityType(
        id="strategic_charger",
        name_zh="战略冲锋手",
        dimensions=PersonalityDimensions(urgency="high", action_pattern="early", time_orientation="future"),
        description="紧迫感强，行动迅速，且着眼未来。擅长快速启动战略性项目，是团队的先锋力量。",
        strengths=["快速启动项目", "战略眼光", "强执行力"],
        weaknesses=["容易忽略细节", "可能给团队施加过大压力"],
        color="#E74C3C",
        icon="⚡",
    ),
    "blitz_executor": PersonalityType(
        id="blitz_executor",
        name_zh="快攻执行者",
        dimensions=PersonalityDimensions(urgency="high", action_pattern="early", time_orientation="present"),
        description="紧迫感强，行动迅速，专注当下。以极高效率完成当前任务，是最快出活的类型。",
        strengths=["极高的执行速度", "专注当下任务", "结果导向"],
        weaknesses=["缺乏长远规划", "容易疲劳和倦怠"],
        color="#E67E22",
        icon="🏃",
    ),
    "planned_pusher": PersonalityType(
        id="planned_pusher",
        name_zh="规划型推进器",
        dimensions=PersonalityDimensions(urgency="high", action_pattern="steady", time_orientation="future"),
        description="紧迫感强但节奏稳定，着眼未来。善于将长期目标分解为稳步推进的计划。",
        strengths=["计划性强", "持续高产", "战略思维"],
        weaknesses=["对突发变化适应慢", "可能过于刻板"],
        color="#9B59B6",
        icon="📋",
    ),
    "high_pressure_steady": PersonalityType(
        id="high_pressure_steady",
        name_zh="高压稳定输出者",
        dimensions=PersonalityDimensions(urgency="high", action_pattern="steady", time_orientation="present"),
        description="紧迫感强但能稳定输出，专注当下。在压力下依然保持稳定产出的可靠力量。",
        strengths=["抗压能力强", "稳定输出", "可靠性高"],
        weaknesses=["缺乏创新性", "容易陷入机械工作"],
        color="#2980B9",
        icon="🔧",
    ),
    "goal_driven_deadline": PersonalityType(
        id="goal_driven_deadline",
        name_zh="目标驱动压线高手",
        dimensions=PersonalityDimensions(urgency="high", action_pattern="deadline", time_orientation="future"),
        description="紧迫感强，deadline驱动，着眼未来。善于在截止日期前爆发，同时保持战略视野。",
        strengths=["目标感强", "爆发力惊人", "战略性压线"],
        weaknesses=["过程管理弱", "给协作者带来焦虑"],
        color="#E91E63",
        icon="🎯",
    ),
    "firefighter_closer": PersonalityType(
        id="firefighter_closer",
        name_zh="救火型终结者",
        dimensions=PersonalityDimensions(urgency="high", action_pattern="deadline", time_orientation="present"),
        description="紧迫感强，deadline驱动，专注当下。擅长在最后时刻完成任务的救火队员。",
        strengths=["危机处理能力强", "临场应变快", "抗压极强"],
        weaknesses=["拖延倾向", "工作质量不稳定", "影响团队节奏"],
        color="#FF5722",
        icon="🚒",
    ),
    "calm_preemptive": PersonalityType(
        id="calm_preemptive",
        name_zh="从容先手布局者",
        dimensions=PersonalityDimensions(urgency="low", action_pattern="early", time_orientation="future"),
        description="不急不躁，提前行动，着眼未来。从容地做好前瞻性布局，是团队的战略定海神针。",
        strengths=["前瞻性布局", "从容不迫", "深思熟虑"],
        weaknesses=["可能过于保守", "在紧急项目中显得慢"],
        color="#27AE60",
        icon="🏗️",
    ),
    "light_explorer": PersonalityType(
        id="light_explorer",
        name_zh="轻快探索先锋",
        dimensions=PersonalityDimensions(urgency="low", action_pattern="early", time_orientation="present"),
        description="轻松心态，提前行动，专注当下。以好奇心驱动，快速探索和尝试的创新者。",
        strengths=["创造力强", "乐于尝试", "团队氛围调节者"],
        weaknesses=["缺乏持久力", "容易分心", "深度不足"],
        color="#1ABC9C",
        icon="🔍",
    ),
    "long_term_builder": PersonalityType(
        id="long_term_builder",
        name_zh="长期主义耐力型建设者",
        dimensions=PersonalityDimensions(urgency="low", action_pattern="steady", time_orientation="future"),
        description="不急躁，节奏稳定，着眼未来。像马拉松选手一样持续建设长期项目。",
        strengths=["耐力持久", "长期价值创造", "基础设施建设"],
        weaknesses=["短期成果少", "容易被忽视", "节奏可能过慢"],
        color="#3498DB",
        icon="🧱",
    ),
    "stable_implementer": PersonalityType(
        id="stable_implementer",
        name_zh="稳态落地者",
        dimensions=PersonalityDimensions(urgency="low", action_pattern="steady", time_orientation="present"),
        description="不急躁，节奏稳定，专注当下。是团队中最稳定可靠的执行者。",
        strengths=["极高可靠性", "质量稳定", "团队定心丸"],
        weaknesses=["缺乏冲劲", "不擅长应对变化", "创新不足"],
        color="#34495E",
        icon="⚙️",
    ),
    "gentle_deadline": PersonalityType(
        id="gentle_deadline",
        name_zh="温和压线规划者",
        dimensions=PersonalityDimensions(urgency="low", action_pattern="deadline", time_orientation="future"),
        description="不急躁，deadline驱动，着眼未来。温和地规划未来，但习惯性压线完成。",
        strengths=["规划能力好", "临近deadline效率提升", "心态平和"],
        weaknesses=["拖延倾向", "早期产出低", "可能误判deadline"],
        color="#8E44AD",
        icon="🕰️",
    ),
    "zen_deadline": PersonalityType(
        id="zen_deadline",
        name_zh="佛系压线选手",
        dimensions=PersonalityDimensions(urgency="low", action_pattern="deadline", time_orientation="present"),
        description="不急躁，deadline驱动，专注当下。以佛系心态面对工作，但总能在最后完成。",
        strengths=["心态极好", "不给团队传递焦虑", "最终能完成任务"],
        weaknesses=["严重拖延", "不可预测性高", "难以协作"],
        color="#95A5A6",
        icon="🧘",
    ),
}


# ---------------------------------------------------------------------------
# Pre-defined 2 boss types
# ---------------------------------------------------------------------------
BOSS_TYPES: dict[str, BossType] = {
    "time_master": BossType(
        id="time_master",
        name_zh="高效时间管理者",
        description="有条理、清晰 deadline、结构化管理。为团队提供明确的时间框架和优先级排序。",
        traits=["清晰的deadline设定", "结构化工作安排", "优先级管理明确", "节奏感强"],
        management_style="通过清晰的计划和时间框架引导团队，强调按部就班和可预测性。善于分解任务、设定里程碑。",
    ),
    "time_chaos": BossType(
        id="time_chaos",
        name_zh="混乱时间管理者",
        description="优先级多变、临时改需求、缺乏规划。给团队带来不确定性和频繁的方向调整。",
        traits=["优先级频繁变动", "临时改需求", "缺乏长期规划", "紧急任务多"],
        management_style="管理风格随性，经常改变方向和优先级。团队需要很强的适应能力和自我管理能力。",
    ),
    "time_neutral": BossType(
        id="time_neutral",
        name_zh="中性管理者",
        description="管理风格均衡，不特别强调时间管理，也不刻意制造混乱。按需推进工作。",
        traits=["任务导向", "情境适应", "不偏不倚", "自然节奏"],
        management_style="根据实际情况灵活管理，不刻意强调时间框架，也不刻意打乱节奏。",
    ),
}


def get_personality_type(type_id: str) -> PersonalityType | None:
    """Look up a personality type by ID."""
    return PERSONALITY_TYPES.get(type_id)


def get_boss_type(type_id: str) -> BossType | None:
    """Look up a boss type by ID."""
    return BOSS_TYPES.get(type_id)


def get_member_dimensions(member: TeamMember) -> PersonalityDimensions | None:
    """Resolve a member's personality dimensions from registry."""
    ptype = PERSONALITY_TYPES.get(member.personality_type_id)
    return ptype.dimensions if ptype else None

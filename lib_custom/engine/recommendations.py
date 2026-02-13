"""Management recommendation generator.

Produces actionable suggestions based on team config + project type.
All functions are *pure*.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lib_custom.engine.compatibility import boss_compatibility_for_team
from lib_custom.engine.conflicts import detect_conflicts
from lib_custom.personality_types import (
    BOSS_TYPES,
    PERSONALITY_TYPES,
    TeamConfig,
)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------
class Recommendation(BaseModel):
    """A single management recommendation."""

    category: str  # e.g. "任务分配", "沟通策略", "风险预警"
    title: str
    description: str
    target_members: list[str] = Field(default_factory=list)
    priority: int = Field(default=1, ge=1, le=3)  # 1=high 3=low


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_recommendations(
    config: TeamConfig,
    project_type_id: str = "urgent_launch",
) -> list[Recommendation]:
    """Return a sorted list of recommendations (priority asc)."""
    recs: list[Recommendation] = []

    _task_assignment_recs(config, project_type_id, recs)
    _communication_recs(config, recs)
    _risk_recs(config, recs)

    return sorted(recs, key=lambda r: r.priority)


# ---------------------------------------------------------------------------
# Recommendation generators
# ---------------------------------------------------------------------------
def _task_assignment_recs(
    config: TeamConfig,
    project_type_id: str,
    recs: list[Recommendation],
) -> None:
    """Task assignment suggestions based on personality types."""
    for m in config.members:
        pt = PERSONALITY_TYPES.get(m.personality_type_id)
        if pt is None:
            continue
        dim = pt.dimensions

        if dim.action_pattern == "early" and dim.time_orientation == "future":
            recs.append(Recommendation(
                category="任务分配",
                title=f"让{m.name}负责项目前期规划",
                description=f"{m.name}（{pt.name_zh}）擅长提前布局和战略规划，适合负责项目初始阶段和方向设定。",
                target_members=[m.id],
                priority=2,
            ))
        elif dim.action_pattern == "steady":
            recs.append(Recommendation(
                category="任务分配",
                title=f"安排{m.name}负责持续性任务",
                description=f"{m.name}（{pt.name_zh}）擅长稳定输出，适合需要持续推进的核心模块。",
                target_members=[m.id],
                priority=2,
            ))
        elif dim.action_pattern == "deadline" and dim.urgency == "high":
            recs.append(Recommendation(
                category="任务分配",
                title=f"给{m.name}设定明确的checkpoint",
                description=f"{m.name}（{pt.name_zh}）是deadline驱动型且紧迫感强，建议设置多个中间检查点避免最后压线。",
                target_members=[m.id],
                priority=1,
            ))
        elif dim.action_pattern == "deadline" and dim.urgency == "low":
            recs.append(Recommendation(
                category="任务分配",
                title=f"为{m.name}提前设置截止日期",
                description=f"{m.name}（{pt.name_zh}）有拖延倾向，建议将实际deadline提前，并安排定期跟进。",
                target_members=[m.id],
                priority=1,
            ))


def _communication_recs(config: TeamConfig, recs: list[Recommendation]) -> None:
    """Communication strategy based on boss type."""
    boss = BOSS_TYPES.get(config.boss_type_id)
    if boss is None:
        return

    compat_scores = boss_compatibility_for_team(config)
    low_compat = [s for s in compat_scores if s.score < 50]

    if config.boss_type_id == "time_master":
        recs.append(Recommendation(
            category="沟通策略",
            title="利用结构化管理优势",
            description="高效时间管理型老板擅长清晰沟通，建议充分利用其结构化能力，建立规范的进度汇报机制。",
            priority=2,
        ))
        if low_compat:
            names = ", ".join(s.member_name for s in low_compat)
            recs.append(Recommendation(
                category="沟通策略",
                title="关注低兼容性成员",
                description=f"以下成员与时间管理型老板兼容性较低：{names}。建议给予更多灵活空间或调整沟通频率。",
                target_members=[s.member_id for s in low_compat],
                priority=1,
            ))
    elif config.boss_type_id == "time_chaos":
        recs.append(Recommendation(
            category="沟通策略",
            title="建立变化缓冲机制",
            description="混乱管理型老板需求变化快，建议团队建立变化缓冲层，避免频繁方向调整直接冲击一线开发。",
            priority=1,
        ))
        if low_compat:
            names = ", ".join(s.member_name for s in low_compat)
            recs.append(Recommendation(
                category="沟通策略",
                title="保护低适应性成员",
                description=f"以下成员不适应频繁变化：{names}。建议安排稳定的工作内容，减少受需求变更影响。",
                target_members=[s.member_id for s in low_compat],
                priority=1,
            ))


def _risk_recs(config: TeamConfig, recs: list[Recommendation]) -> None:
    """Risk warnings from conflict detection."""
    conflicts = detect_conflicts(config)
    recs.extend(
        Recommendation(
            category="风险预警",
            title=alert.title,
            description=alert.description,
            target_members=alert.affected_members,
            priority=1 if alert.severity == "high" else 2,
        )
        for alert in conflicts
    )

"""Temporal leadership (TTL) module — decoupled from leadership styles.

TTL captures how much a leader emphasizes temporal structuring behaviors
(deadline setting, progress syncing, milestone management, pacing coordination).
Three levels: low (0.0), medium (0.5), high (1.0).

References:
    Mohammed & Nadkarni (2011). Temporal Diversity and Team Performance.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lib_custom.role_models import RoleConfig


# ---------------------------------------------------------------------------
# TTL behaviour descriptions (prompt-level operationalization)
# ---------------------------------------------------------------------------
_TTL_BEHAVIORS: dict[str, dict[str, str]] = {
    "high": {
        "label": "高 TTL",
        "code": 1.0,
        "time_framing": (
            "你会为每个议题明确分配讨论时间，频繁提及时间节点和 deadline，"
            "主动追问各成员的进度完成情况。"
        ),
        "milestone_mgmt": (
            "你善于设定阶段性里程碑并严格追踪，"
            "确保团队在关键时间节点完成对应产出。"
        ),
        "pacing_sync": (
            "你会识别成员之间的节奏差异并主动协调，"
            "对节奏过慢的成员提出具体的时间建议，"
            "对节奏过快的成员提醒注意质量。"
        ),
        "overall_instruction": (
            "在讨论中，你始终关注时间维度。每次发言都应涉及时间节点、进度或 deadline。"
            "为讨论设定明确的时间框架，推动团队按节奏前进。"
        ),
    },
    "medium": {
        "label": "中 TTL",
        "code": 0.5,
        "time_framing": (
            "你偶尔会提及 deadline 和关键时间节点，"
            "在重要阶段转换时检查进度。"
        ),
        "milestone_mgmt": (
            "你在被问时给出时间建议，"
            "在关键节点主动检查里程碑完成情况。"
        ),
        "pacing_sync": (
            "你注意到成员之间明显的节奏不一致时会提醒，"
            "但不会过度干预个人工作节奏。"
        ),
        "overall_instruction": (
            "在讨论中，你会在合适的时候提及时间和进度，但不会让它成为每次发言的核心。"
            "保持内容质量与时间管理之间的平衡。"
        ),
    },
    "low": {
        "label": "低 TTL",
        "code": 0.0,
        "time_framing": (
            "你几乎不主动提及时间和 deadline，"
            "更关注讨论的内容质量而非进度。"
        ),
        "milestone_mgmt": (
            "你不设定明确的时间框架或里程碑，"
            "让事情按自然节奏发展。"
        ),
        "pacing_sync": (
            "你不特别关注成员之间的节奏差异，"
            "不主动协调团队的工作步调。"
        ),
        "overall_instruction": (
            "在讨论中，你专注于内容本身的质量和深度。不主动设定时间框架，"
            "不追问进度，让讨论自然展开。"
        ),
    },
}

TTL_LEVELS: list[str] = ["low", "medium", "high"]
TTL_CODE_MAP: dict[str, float] = {"low": 0.0, "medium": 0.5, "high": 1.0}


# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------
class TemporalLeadershipConfig(BaseModel):
    """Configuration for a specific TTL level."""

    level: str = Field(..., pattern=r"^(low|medium|high)$")
    code: float = Field(..., ge=0.0, le=1.0)
    label: str
    time_framing: str
    milestone_mgmt: str
    pacing_sync: str
    overall_instruction: str


def get_ttl_config(level: str) -> TemporalLeadershipConfig:
    """Return a TemporalLeadershipConfig for the given level.

    Args:
        level: One of "low", "medium", "high".

    Raises:
        ValueError: If level is not recognized.
    """
    behaviors = _TTL_BEHAVIORS.get(level)
    if behaviors is None:
        raise ValueError(f"Unknown TTL level: {level!r}. Must be one of {TTL_LEVELS}")
    return TemporalLeadershipConfig(
        level=level,
        code=TTL_CODE_MAP[level],
        label=behaviors["label"],
        time_framing=behaviors["time_framing"],
        milestone_mgmt=behaviors["milestone_mgmt"],
        pacing_sync=behaviors["pacing_sync"],
        overall_instruction=behaviors["overall_instruction"],
    )


# ---------------------------------------------------------------------------
# Boss role builder (TTL-aware, neutral base persona)
# ---------------------------------------------------------------------------
_NEUTRAL_BOSS_BACKSTORY = (
    "你是团队的项目负责人，负责协调团队完成当前项目目标。"
    "你的管理经验丰富，善于引导讨论并推动共识形成。"
    "你不偏向任何特定的领导风格，而是根据情境需要灵活调整。"
)

_NEUTRAL_BOSS_PERSONALITY = "沉稳、务实、善于协调"
_NEUTRAL_BOSS_COMMUNICATION = "清晰、简洁、注重倾听"
_NEUTRAL_BOSS_EMOTIONAL = "情绪稳定、客观理性"
_NEUTRAL_BOSS_VALUES = "团队目标达成、成员发展、高效协作"


def build_ttl_boss_role(
    ttl_config: TemporalLeadershipConfig,
    base_name: str = "项目负责人",
) -> RoleConfig:
    """Build a boss RoleConfig with neutral persona + TTL behavior overlay.

    The boss persona is deliberately neutral (no leadership style binding).
    TTL behaviors are layered on top via the backstory and goal.

    Args:
        ttl_config: TTL level configuration.
        base_name: Display name for the boss.

    Returns:
        A RoleConfig with TTL-specific behavioral instructions.
    """
    ttl_section = (
        f"\n\n## 你的时间领导力行为 ({ttl_config.label})\n"
        f"- 时间框架设定: {ttl_config.time_framing}\n"
        f"- 里程碑管理: {ttl_config.milestone_mgmt}\n"
        f"- 节奏同步: {ttl_config.pacing_sync}\n"
        f"- 总体要求: {ttl_config.overall_instruction}"
    )

    goal = (
        f"作为项目负责人，引导团队高效讨论并形成可执行方案。"
        f"{ttl_config.overall_instruction}"
    )

    backstory = _NEUTRAL_BOSS_BACKSTORY + ttl_section

    return RoleConfig(
        role_id="boss",
        role_name=f"👔 {base_name} ({ttl_config.label})",
        goal=goal,
        backstory=backstory,
        avatar="👔",
        role_type="conversation",
        is_default=False,
        order=0,
        personality=_NEUTRAL_BOSS_PERSONALITY,
        communication_style=_NEUTRAL_BOSS_COMMUNICATION,
        emotional_tendency=_NEUTRAL_BOSS_EMOTIONAL,
        values=_NEUTRAL_BOSS_VALUES,
    )

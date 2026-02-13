"""Default 12-member team for thesis experiments.

Each of the 12 personality types is represented by one member with a
Chinese name that reflects their personality trait.
"""

from __future__ import annotations

from lib_custom.personality_types import PERSONALITY_TYPES, TeamConfig, TeamMember


# ---------------------------------------------------------------------------
# 12 default members — one per personality type, Chinese names
# ---------------------------------------------------------------------------
_DEFAULT_MEMBERS: list[dict[str, str | int]] = [
    {"id": "m01", "name": "赵冲", "personality_type_id": "strategic_charger", "order": 1},
    {"id": "m02", "name": "钱迅", "personality_type_id": "blitz_executor", "order": 2},
    {"id": "m03", "name": "孙策", "personality_type_id": "planned_pusher", "order": 3},
    {"id": "m04", "name": "李稳", "personality_type_id": "high_pressure_steady", "order": 4},
    {"id": "m05", "name": "周驱", "personality_type_id": "goal_driven_deadline", "order": 5},
    {"id": "m06", "name": "吴救", "personality_type_id": "firefighter_closer", "order": 6},
    {"id": "m07", "name": "郑谋", "personality_type_id": "calm_preemptive", "order": 7},
    {"id": "m08", "name": "王探", "personality_type_id": "light_explorer", "order": 8},
    {"id": "m09", "name": "冯筑", "personality_type_id": "long_term_builder", "order": 9},
    {"id": "m10", "name": "陈实", "personality_type_id": "stable_implementer", "order": 10},
    {"id": "m11", "name": "卫缓", "personality_type_id": "gentle_deadline", "order": 11},
    {"id": "m12", "name": "蒋安", "personality_type_id": "zen_deadline", "order": 12},
]

DEFAULT_TEAM_MEMBERS: list[TeamMember] = [
    TeamMember(**m) for m in _DEFAULT_MEMBERS  # type: ignore[arg-type]
]


def create_default_team(boss_type_id: str) -> TeamConfig:
    """Create a full 12-member team config with the given boss type.

    Args:
        boss_type_id: One of "time_master" or "time_chaos".

    Returns:
        A TeamConfig with 12 members (one per personality type).
    """
    return TeamConfig(
        boss_type_id=boss_type_id,
        members=list(DEFAULT_TEAM_MEMBERS),
    )


def get_member_personality_name(member: TeamMember) -> str:
    """Return the Chinese personality type name for a team member."""
    ptype = PERSONALITY_TYPES.get(member.personality_type_id)
    return ptype.name_zh if ptype else member.personality_type_id


def get_all_personality_type_ids() -> set[str]:
    """Return all personality type IDs covered by the default team."""
    return {m.personality_type_id for m in DEFAULT_TEAM_MEMBERS}

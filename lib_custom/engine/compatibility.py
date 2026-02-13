"""Compatibility scoring between boss ↔ employee and peer ↔ peer.

All functions are *pure* — no side-effects, no I/O.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lib_custom.personality_types import (
    BOSS_TYPES,
    PERSONALITY_TYPES,
    PersonalityDimensions,
    TeamConfig,
    TeamMember,
)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------
class CompatibilityScore(BaseModel):
    """Score for a single boss ↔ employee pair."""

    member_id: str
    member_name: str
    boss_type_id: str
    score: int = Field(ge=0, le=100)
    detail: str = ""


class PeerCompatibilityScore(BaseModel):
    """Score for a peer ↔ peer pair."""

    member_a_id: str
    member_b_id: str
    score: int = Field(ge=0, le=100)
    detail: str = ""


# ---------------------------------------------------------------------------
# Scoring matrices
# ---------------------------------------------------------------------------
_BOSS_SCORE_MATRIX: dict[str, dict[str, int]] = {
    "time_master": {
        "urgency:high": 15,
        "urgency:low": 5,
        "action:early": 20,
        "action:steady": 15,
        "action:deadline": -10,
        "time:future": 10,
        "time:present": 5,
    },
    "time_chaos": {
        "urgency:high": -5,
        "urgency:low": 10,
        "action:early": -10,
        "action:steady": 5,
        "action:deadline": 15,
        "time:future": -5,
        "time:present": 15,
    },
}


def _dim_keys(dim: PersonalityDimensions) -> list[str]:
    return [
        f"urgency:{dim.urgency}",
        f"action:{dim.action_pattern}",
        f"time:{dim.time_orientation}",
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def calculate_boss_compatibility(
    boss_type_id: str,
    dimensions: PersonalityDimensions,
) -> int:
    """Return compatibility score (0-100) for a boss type + personality dims."""
    matrix = _BOSS_SCORE_MATRIX.get(boss_type_id, {})
    score = 50
    for key in _dim_keys(dimensions):
        score += matrix.get(key, 0)
    return max(0, min(100, score))


def calculate_peer_compatibility(
    a: PersonalityDimensions,
    b: PersonalityDimensions,
) -> int:
    """Return compatibility score (0-100) for two peers."""
    score = 50

    # Same urgency → +10
    if a.urgency == b.urgency:
        score += 10

    # Same action_pattern → +15; early↔deadline → −15
    if a.action_pattern == b.action_pattern:
        score += 15
    elif {a.action_pattern, b.action_pattern} == {"early", "deadline"}:
        score -= 15

    # Same time_orientation → +10
    if a.time_orientation == b.time_orientation:
        score += 10

    return max(0, min(100, score))


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------
def boss_compatibility_for_team(config: TeamConfig) -> list[CompatibilityScore]:
    """Compute boss compatibility for every member in the team."""
    results: list[CompatibilityScore] = []
    boss = BOSS_TYPES.get(config.boss_type_id)
    if boss is None:
        return results

    for member in config.members:
        ptype = PERSONALITY_TYPES.get(member.personality_type_id)
        if ptype is None:
            continue
        sc = calculate_boss_compatibility(config.boss_type_id, ptype.dimensions)
        results.append(CompatibilityScore(
            member_id=member.id,
            member_name=member.name,
            boss_type_id=config.boss_type_id,
            score=sc,
            detail=_boss_detail(config.boss_type_id, ptype.dimensions, sc),
        ))
    return results


def peer_compatibility_matrix(members: list[TeamMember]) -> list[PeerCompatibilityScore]:
    """Compute NxN peer compatibility (upper-triangle only)."""
    results: list[PeerCompatibilityScore] = []
    resolved = [
        (m, PERSONALITY_TYPES[m.personality_type_id].dimensions)
        for m in members
        if m.personality_type_id in PERSONALITY_TYPES
    ]
    for i, (ma, da) in enumerate(resolved):
        for mb, db in resolved[i + 1:]:
            sc = calculate_peer_compatibility(da, db)
            results.append(PeerCompatibilityScore(
                member_a_id=ma.id,
                member_b_id=mb.id,
                score=sc,
                detail=_peer_detail(da, db, sc),
            ))
    return results


# ---------------------------------------------------------------------------
# Detail text helpers
# ---------------------------------------------------------------------------
def _boss_detail(boss_id: str, dim: PersonalityDimensions, score: int) -> str:
    boss_name = BOSS_TYPES[boss_id].name_zh if boss_id in BOSS_TYPES else boss_id
    if score >= 80:
        return f"与{boss_name}高度契合"
    if score >= 60:
        return f"与{boss_name}较为适配"
    if score >= 40:
        return f"与{boss_name}存在一定摩擦"
    return f"与{boss_name}可能产生较大冲突"


def _peer_detail(a: PersonalityDimensions, b: PersonalityDimensions, score: int) -> str:
    if score >= 75:
        return "高度互补或相似，协作顺畅"
    if score >= 55:
        return "较为兼容，偶有分歧"
    if score >= 40:
        return "存在一定摩擦，需注意沟通"
    return "差异较大，协作需额外关注"

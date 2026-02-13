"""Performance prediction engine.

3 project types × per-member fitness + team synergy = prediction.
All functions are *pure*.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from lib_custom.engine.compatibility import (
    calculate_boss_compatibility,
    calculate_peer_compatibility,
)
from lib_custom.personality_types import (
    BOSS_TYPES,
    PERSONALITY_TYPES,
    PersonalityDimensions,
    TeamConfig,
)


# ---------------------------------------------------------------------------
# Project types
# ---------------------------------------------------------------------------
ProjectTypeId = Literal["urgent_launch", "long_term_platform", "exploratory_prototype"]

PROJECT_TYPES: dict[str, dict[str, str]] = {
    "urgent_launch": {
        "name_zh": "紧急上线",
        "icon": "🚀",
        "description": "时间紧迫、需要快速交付的项目",
    },
    "long_term_platform": {
        "name_zh": "长期平台建设",
        "icon": "🏛️",
        "description": "持续数月甚至数年的基础设施或平台项目",
    },
    "exploratory_prototype": {
        "name_zh": "探索性原型",
        "icon": "🧪",
        "description": "需要创新和快速试错的原型探索项目",
    },
}

# Fitness weights per project type (dimension_key → bonus/penalty)
_FITNESS: dict[str, dict[str, int]] = {
    "urgent_launch": {
        "urgency:high": 20,
        "urgency:low": -10,
        "action:early": 15,
        "action:steady": 5,
        "action:deadline": -5,
        "time:present": 10,
        "time:future": 0,
    },
    "long_term_platform": {
        "urgency:high": -5,
        "urgency:low": 15,
        "action:early": 5,
        "action:steady": 20,
        "action:deadline": -10,
        "time:future": 20,
        "time:present": -5,
    },
    "exploratory_prototype": {
        "urgency:high": 5,
        "urgency:low": 10,
        "action:early": 20,
        "action:steady": 0,
        "action:deadline": -5,
        "time:future": 15,
        "time:present": 5,
    },
}


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------
class MemberFitness(BaseModel):
    """Individual member fitness for a project type."""

    member_id: str
    member_name: str
    score: int = Field(ge=0, le=100)
    detail: str = ""


class PerformancePrediction(BaseModel):
    """Aggregated prediction for a team + project type."""

    project_type_id: str
    project_name: str
    individual_scores: list[MemberFitness]
    team_synergy_score: int = Field(ge=0, le=100)
    overall_score: int = Field(ge=0, le=100)
    summary: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _dim_keys(dim: PersonalityDimensions) -> list[str]:
    return [
        f"urgency:{dim.urgency}",
        f"action:{dim.action_pattern}",
        f"time:{dim.time_orientation}",
    ]


def _individual_fitness(dim: PersonalityDimensions, project_type_id: str) -> int:
    weights = _FITNESS.get(project_type_id, {})
    score = 50
    for key in _dim_keys(dim):
        score += weights.get(key, 0)
    return max(0, min(100, score))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_performance(
    config: TeamConfig,
    project_type_id: str,
) -> PerformancePrediction:
    """Predict team performance for a given project type."""
    pinfo = PROJECT_TYPES.get(project_type_id, {"name_zh": project_type_id, "icon": "❓", "description": ""})

    # Individual fitness
    individual: list[MemberFitness] = []
    dims_list: list[PersonalityDimensions] = []
    for m in config.members:
        pt = PERSONALITY_TYPES.get(m.personality_type_id)
        if pt is None:
            continue
        sc = _individual_fitness(pt.dimensions, project_type_id)
        dims_list.append(pt.dimensions)
        individual.append(MemberFitness(
            member_id=m.id,
            member_name=m.name,
            score=sc,
            detail=_fitness_label(sc),
        ))

    # Team synergy = avg peer compat + boss compat bonus
    synergy = _calc_synergy(config, dims_list)

    # Overall = 60% individual avg + 40% synergy
    avg_ind = sum(mf.score for mf in individual) / len(individual) if individual else 50
    overall = int(avg_ind * 0.6 + synergy * 0.4)
    overall = max(0, min(100, overall))

    return PerformancePrediction(
        project_type_id=project_type_id,
        project_name=pinfo["name_zh"],
        individual_scores=individual,
        team_synergy_score=synergy,
        overall_score=overall,
        summary=_overall_label(overall),
    )


def _calc_synergy(config: TeamConfig, dims_list: list[PersonalityDimensions]) -> int:
    if not dims_list:
        return 50

    # Peer compatibility average
    peer_scores: list[int] = [
        calculate_peer_compatibility(da, db)
        for i, da in enumerate(dims_list)
        for db in dims_list[i + 1:]
    ]
    avg_peer = sum(peer_scores) / len(peer_scores) if peer_scores else 50

    # Boss compatibility average
    boss_scores: list[int] = (
        [calculate_boss_compatibility(config.boss_type_id, dim) for dim in dims_list]
        if config.boss_type_id in BOSS_TYPES
        else []
    )
    avg_boss = sum(boss_scores) / len(boss_scores) if boss_scores else 50

    synergy = int(avg_peer * 0.5 + avg_boss * 0.5)
    return max(0, min(100, synergy))


def _fitness_label(score: int) -> str:
    if score >= 80:
        return "非常适合"
    if score >= 60:
        return "较为适合"
    if score >= 40:
        return "一般"
    return "不太适合"


def _overall_label(score: int) -> str:
    if score >= 80:
        return "团队高度适配该项目类型，预期表现优秀"
    if score >= 60:
        return "团队整体适配度较好，可正常推进"
    if score >= 40:
        return "团队适配度一般，建议调整人员或管理方式"
    return "团队与项目类型匹配度低，建议重新评估"

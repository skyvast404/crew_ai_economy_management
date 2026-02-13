"""Team balance analysis — diversity scoring and dimension gap detection.

All functions are *pure*.
"""

from __future__ import annotations

from collections import Counter
import math

from pydantic import BaseModel, Field

from lib_custom.personality_types import PERSONALITY_TYPES, TeamMember


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------
class DimensionDistribution(BaseModel):
    """Count distribution for a single dimension."""

    dimension_name: str
    counts: dict[str, int]
    entropy: float = 0.0


class BlauDiversityMetrics(BaseModel):
    """Blau's index of heterogeneity for team composition dimensions."""

    urgency_blau: float = Field(ge=0.0, le=1.0, default=0.0)
    action_pattern_blau: float = Field(ge=0.0, le=1.0, default=0.0)
    time_orientation_blau: float = Field(ge=0.0, le=1.0, default=0.0)
    composite: float = Field(ge=0.0, le=1.0, default=0.0)


class TeamBalance(BaseModel):
    """Overall team balance report."""

    diversity_score: float = Field(ge=0.0, le=100.0)
    distributions: list[DimensionDistribution]
    missing_values: dict[str, list[str]]  # dim_name → missing values
    warnings: list[str]


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------
def _shannon_entropy(counts: dict[str, int]) -> float:
    """Normalised Shannon entropy ∈ [0, 1]."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    k = len(counts)
    if k <= 1:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    raw = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(k)
    return raw / max_entropy if max_entropy > 0 else 0.0


# ---------------------------------------------------------------------------
# Blau's index of heterogeneity
# ---------------------------------------------------------------------------
def calculate_blau_index(values: list[str]) -> float:
    """Calculate Blau's index for a single categorical dimension.

    Blau = 1 - Σ(pᵢ²), where pᵢ is proportion of category i.
    Range: [0, 1]. Higher = more diverse.

    Args:
        values: List of categorical values (e.g., ["high", "low", "high"]).

    Returns:
        Blau index ∈ [0, 1].
    """
    if not values:
        return 0.0
    n = len(values)
    counts = Counter(values)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def calculate_blau_diversity(members: list[TeamMember]) -> BlauDiversityMetrics:
    """Calculate 3-dimension Blau indices + composite for team members.

    Args:
        members: List of team members.

    Returns:
        BlauDiversityMetrics with per-dimension Blau and composite average.
    """
    urgency_vals: list[str] = []
    action_vals: list[str] = []
    time_vals: list[str] = []

    for m in members:
        pt = PERSONALITY_TYPES.get(m.personality_type_id)
        if pt is None:
            continue
        urgency_vals.append(pt.dimensions.urgency)
        action_vals.append(pt.dimensions.action_pattern)
        time_vals.append(pt.dimensions.time_orientation)

    u_blau = calculate_blau_index(urgency_vals)
    a_blau = calculate_blau_index(action_vals)
    t_blau = calculate_blau_index(time_vals)
    composite = (u_blau + a_blau + t_blau) / 3.0

    return BlauDiversityMetrics(
        urgency_blau=round(u_blau, 4),
        action_pattern_blau=round(a_blau, 4),
        time_orientation_blau=round(t_blau, 4),
        composite=round(composite, 4),
    )


# ---------------------------------------------------------------------------
# Possible values per dimension
# ---------------------------------------------------------------------------
_POSSIBLE: dict[str, list[str]] = {
    "urgency": ["high", "low"],
    "action_pattern": ["early", "steady", "deadline"],
    "time_orientation": ["future", "present"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def calculate_team_balance(members: list[TeamMember]) -> TeamBalance:
    """Analyse dimension distribution and diversity for *members*."""
    dims_data: dict[str, Counter[str]] = {
        "urgency": Counter(),
        "action_pattern": Counter(),
        "time_orientation": Counter(),
    }

    for m in members:
        pt = PERSONALITY_TYPES.get(m.personality_type_id)
        if pt is None:
            continue
        dims_data["urgency"][pt.dimensions.urgency] += 1
        dims_data["action_pattern"][pt.dimensions.action_pattern] += 1
        dims_data["time_orientation"][pt.dimensions.time_orientation] += 1

    distributions: list[DimensionDistribution] = []
    entropies: list[float] = []
    missing: dict[str, list[str]] = {}
    warnings: list[str] = []

    for dim_name, counter in dims_data.items():
        # fill missing values with 0 for entropy calc
        full_counts: dict[str, int] = {v: counter.get(v, 0) for v in _POSSIBLE[dim_name]}
        ent = _shannon_entropy(full_counts)
        entropies.append(ent)
        distributions.append(DimensionDistribution(
            dimension_name=dim_name,
            counts=full_counts,
            entropy=round(ent, 3),
        ))

        # detect missing values
        absent = [v for v in _POSSIBLE[dim_name] if counter.get(v, 0) == 0]
        if absent:
            missing[dim_name] = absent

    diversity = round(sum(entropies) / len(entropies) * 100, 1) if entropies else 0.0

    # generate warnings
    if "future" in missing.get("time_orientation", []):
        warnings.append("团队缺少面向未来的成员，可能存在战略盲区")
    if "present" in missing.get("time_orientation", []):
        warnings.append("团队缺少关注当下的成员，执行落地能力可能不足")
    if "early" in missing.get("action_pattern", []):
        warnings.append("团队无提前行动者，项目启动可能较慢")
    if "steady" in missing.get("action_pattern", []):
        warnings.append("团队缺少稳定输出者，产出节奏可能不均匀")

    total = sum(sum(c.values()) for c in dims_data.values()) // 3
    if total > 0:
        type_counter = Counter(m.personality_type_id for m in members)
        most_common_count = type_counter.most_common(1)[0][1]
        if most_common_count / total >= 0.8:
            warnings.append("80%+ 成员类型相同，存在群体思维风险")

    return TeamBalance(
        diversity_score=diversity,
        distributions=distributions,
        missing_values=missing,
        warnings=warnings,
    )

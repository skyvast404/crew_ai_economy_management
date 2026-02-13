"""Team composition generator with Blau diversity index.

Generates variable 5-person team compositions from the 12-personality-type pool,
stratified by temporal diversity (Blau index).

References:
    Blau, P. M. (1977). Inequality and Heterogeneity.
    Mohammed & Nadkarni (2011). Temporal Diversity and Team Performance.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter

from pydantic import BaseModel, Field

from lib_custom.personality_types import PERSONALITY_TYPES, TeamMember


# ---------------------------------------------------------------------------
# Blau index calculation
# ---------------------------------------------------------------------------
def calculate_blau_index(values: list[str]) -> float:
    """Calculate Blau's index of heterogeneity for a single categorical dimension.

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


# ---------------------------------------------------------------------------
# Diversity metrics model
# ---------------------------------------------------------------------------
class DiversityMetrics(BaseModel):
    """Multi-dimension diversity scores for a team composition."""

    urgency_blau: float = Field(ge=0.0, le=1.0)
    action_pattern_blau: float = Field(ge=0.0, le=1.0)
    time_orientation_blau: float = Field(ge=0.0, le=1.0)
    composite: float = Field(ge=0.0, le=1.0)


def calculate_diversity_metrics(type_ids: list[str]) -> DiversityMetrics:
    """Calculate 3-dimension Blau indices + composite for a set of personality types.

    Args:
        type_ids: List of personality type IDs.

    Returns:
        DiversityMetrics with per-dimension Blau and composite average.
    """
    urgency_vals: list[str] = []
    action_vals: list[str] = []
    time_vals: list[str] = []

    for tid in type_ids:
        ptype = PERSONALITY_TYPES.get(tid)
        if ptype is None:
            continue
        urgency_vals.append(ptype.dimensions.urgency)
        action_vals.append(ptype.dimensions.action_pattern)
        time_vals.append(ptype.dimensions.time_orientation)

    urgency_blau = calculate_blau_index(urgency_vals)
    action_blau = calculate_blau_index(action_vals)
    time_blau = calculate_blau_index(time_vals)
    composite = (urgency_blau + action_blau + time_blau) / 3.0

    return DiversityMetrics(
        urgency_blau=round(urgency_blau, 4),
        action_pattern_blau=round(action_blau, 4),
        time_orientation_blau=round(time_blau, 4),
        composite=round(composite, 4),
    )


# ---------------------------------------------------------------------------
# Diversity stratum classification
# ---------------------------------------------------------------------------
# Thresholds calibrated to C(12,5)=792 actual composite range [0.37, 0.53].
# Uses ~33rd/67th percentile breakpoints for balanced strata.
_LOW_THRESHOLD = 0.48
_HIGH_THRESHOLD = 0.51


def classify_diversity_stratum(composite: float) -> str:
    """Classify a composite Blau score into low/medium/high stratum.

    Thresholds calibrated to the actual composite Blau distribution
    of C(12,5) = 792 possible 5-person combinations from the 12-type pool.
    Actual range: [0.3733, 0.5333].

    Args:
        composite: Composite Blau index ∈ [0, 1].

    Returns:
        "low", "medium", or "high".
    """
    if composite < _LOW_THRESHOLD:
        return "low"
    if composite < _HIGH_THRESHOLD:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Team composition model
# ---------------------------------------------------------------------------
class TeamComposition(BaseModel):
    """A specific 5-person team composition with diversity info."""

    composition_id: str
    type_ids: list[str] = Field(min_length=1)
    diversity: DiversityMetrics
    stratum: str = Field(pattern=r"^(low|medium|high)$")


# ---------------------------------------------------------------------------
# Chinese name pool for generated members
# ---------------------------------------------------------------------------
_NAME_POOL: list[str] = [
    "赵冲", "钱迅", "孙策", "李稳", "周驱",
    "吴救", "郑谋", "王探", "冯筑", "陈实",
    "卫缓", "蒋安", "林达", "黄毅", "韩博",
]


def _build_members_from_type_ids(
    type_ids: list[str],
    composition_id: str,
) -> list[TeamMember]:
    """Build TeamMember list from personality type IDs.

    Args:
        type_ids: Personality type IDs for the team.
        composition_id: Used as prefix for member IDs.

    Returns:
        List of TeamMember with assigned names and orders.
    """
    members: list[TeamMember] = []
    for i, tid in enumerate(type_ids):
        name = _NAME_POOL[i] if i < len(_NAME_POOL) else f"成员{i + 1}"
        members.append(
            TeamMember(
                id=f"{composition_id}_m{i + 1:02d}",
                name=name,
                personality_type_id=tid,
                order=i + 1,
            )
        )
    return members


# ---------------------------------------------------------------------------
# Team composition generation
# ---------------------------------------------------------------------------
def _all_5_person_combinations() -> list[tuple[str, ...]]:
    """Generate all C(12,5) = 792 combinations of personality type IDs."""
    all_type_ids = sorted(PERSONALITY_TYPES.keys())
    return list(itertools.combinations(all_type_ids, 5))


def generate_team_compositions(
    n_per_stratum: int = 8,
    team_size: int = 5,
    seed: int | None = 42,
) -> dict[str, list[TeamComposition]]:
    """Generate team compositions stratified by diversity level.

    1. Enumerate all C(12, team_size) combinations.
    2. Compute composite Blau for each.
    3. Classify into low/medium/high strata.
    4. Sample n_per_stratum from each stratum (without replacement).

    Args:
        n_per_stratum: Number of compositions to sample per stratum.
        team_size: Team size (default 5, matching paper).
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping stratum → list of TeamComposition.
    """
    rng = random.Random(seed)

    all_type_ids = sorted(PERSONALITY_TYPES.keys())
    combos = list(itertools.combinations(all_type_ids, team_size))

    # Classify each combination
    strata: dict[str, list[tuple[str, ...]]] = {
        "low": [],
        "medium": [],
        "high": [],
    }
    for combo in combos:
        metrics = calculate_diversity_metrics(list(combo))
        stratum = classify_diversity_stratum(metrics.composite)
        strata[stratum].append(combo)

    # Sample from each stratum
    result: dict[str, list[TeamComposition]] = {}
    for stratum_name, pool in strata.items():
        sample_size = min(n_per_stratum, len(pool))
        sampled = rng.sample(pool, sample_size)
        compositions: list[TeamComposition] = []
        for idx, combo in enumerate(sampled):
            comp_id = f"{stratum_name}_{idx + 1:02d}"
            type_list = list(combo)
            metrics = calculate_diversity_metrics(type_list)
            compositions.append(
                TeamComposition(
                    composition_id=comp_id,
                    type_ids=type_list,
                    diversity=metrics,
                    stratum=stratum_name,
                )
            )
        result[stratum_name] = compositions

    return result


def composition_to_members(comp: TeamComposition) -> list[TeamMember]:
    """Convert a TeamComposition to a list of TeamMember objects."""
    return _build_members_from_type_ids(comp.type_ids, comp.composition_id)

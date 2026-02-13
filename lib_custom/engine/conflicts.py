"""Conflict detection rules for team configurations.

All functions are *pure*.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from lib_custom.personality_types import PERSONALITY_TYPES, TeamConfig


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------
Severity = Literal["high", "medium", "low"]


class ConflictAlert(BaseModel):
    """A single risk / conflict alert."""

    severity: Severity
    title: str
    description: str
    affected_members: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_conflicts(config: TeamConfig) -> list[ConflictAlert]:
    """Return a list of conflict alerts for *config*, sorted severity desc."""
    alerts: list[ConflictAlert] = []

    dims = _resolve_dims(config)
    if not dims:
        return alerts

    _rule_deadline_chaos(config, dims, alerts)
    _rule_no_future(dims, alerts)
    _rule_no_present(dims, alerts)
    _rule_groupthink(config, alerts)
    _rule_all_high_urgency(dims, alerts)
    _rule_early_vs_deadline(dims, alerts)

    severity_order: dict[Severity, int] = {"high": 0, "medium": 1, "low": 2}
    return sorted(alerts, key=lambda a: severity_order[a.severity])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_DimInfo = dict[str, dict[str, str]]  # member_id → {"urgency": ..., ...}


def _resolve_dims(config: TeamConfig) -> _DimInfo:
    result: _DimInfo = {}
    for m in config.members:
        pt = PERSONALITY_TYPES.get(m.personality_type_id)
        if pt is None:
            continue
        result[m.id] = {
            "urgency": pt.dimensions.urgency,
            "action_pattern": pt.dimensions.action_pattern,
            "time_orientation": pt.dimensions.time_orientation,
            "name": m.name,
        }
    return result


def _rule_deadline_chaos(
    config: TeamConfig,
    dims: _DimInfo,
    alerts: list[ConflictAlert],
) -> None:
    """3+ deadline members + chaos boss → high risk."""
    deadline_ids = [mid for mid, d in dims.items() if d["action_pattern"] == "deadline"]
    if len(deadline_ids) >= 3 and config.boss_type_id == "time_chaos":
        alerts.append(ConflictAlert(
            severity="high",
            title="大量压线成员 + 混乱管理",
            description=(
                f"{len(deadline_ids)} 位deadline驱动成员搭配混乱管理者，"
                "极易出现多任务同时压线、互相阻塞的高风险场景。"
            ),
            affected_members=deadline_ids,
        ))


def _rule_no_future(dims: _DimInfo, alerts: list[ConflictAlert]) -> None:
    """No future-oriented member → strategic blind spot."""
    has_future = any(d["time_orientation"] == "future" for d in dims.values())
    if not has_future:
        alerts.append(ConflictAlert(
            severity="high",
            title="战略盲区",
            description="团队中没有面向未来的成员，可能缺乏长远规划和前瞻性思考。",
            affected_members=[],
        ))


def _rule_no_present(dims: _DimInfo, alerts: list[ConflictAlert]) -> None:
    """No present-oriented member → execution gap."""
    has_present = any(d["time_orientation"] == "present" for d in dims.values())
    if not has_present:
        alerts.append(ConflictAlert(
            severity="medium",
            title="执行力缺口",
            description="团队中没有关注当下的成员，短期任务的执行力可能不足。",
            affected_members=[],
        ))


def _rule_groupthink(config: TeamConfig, alerts: list[ConflictAlert]) -> None:
    """80%+ same personality type → groupthink risk."""
    if len(config.members) < 2:
        return
    counter = Counter(m.personality_type_id for m in config.members)
    most_common_id, most_common_count = counter.most_common(1)[0]
    ratio = most_common_count / len(config.members)
    if ratio >= 0.8:
        pt = PERSONALITY_TYPES.get(most_common_id)
        name = pt.name_zh if pt else most_common_id
        affected = [m.id for m in config.members if m.personality_type_id == most_common_id]
        alerts.append(ConflictAlert(
            severity="high",
            title="群体思维风险",
            description=f"{ratio:.0%} 的成员是「{name}」类型，团队观点可能过于单一。",
            affected_members=affected,
        ))


def _rule_all_high_urgency(dims: _DimInfo, alerts: list[ConflictAlert]) -> None:
    """All members high urgency → burnout risk."""
    if len(dims) < 2:
        return
    all_high = all(d["urgency"] == "high" for d in dims.values())
    if all_high:
        alerts.append(ConflictAlert(
            severity="medium",
            title="全员高紧迫感",
            description="所有成员都处于高紧迫状态，团队可能面临整体倦怠风险。",
            affected_members=list(dims.keys()),
        ))


def _rule_early_vs_deadline(dims: _DimInfo, alerts: list[ConflictAlert]) -> None:
    """Many early + many deadline → pacing conflict."""
    early = [mid for mid, d in dims.items() if d["action_pattern"] == "early"]
    deadline = [mid for mid, d in dims.items() if d["action_pattern"] == "deadline"]
    if len(early) >= 2 and len(deadline) >= 2:
        alerts.append(ConflictAlert(
            severity="medium",
            title="节奏冲突",
            description=(
                f"{len(early)} 位提前行动者与 {len(deadline)} 位压线者共存，"
                "工作节奏差异可能导致协作摩擦。"
            ),
            affected_members=[*early, *deadline],
        ))

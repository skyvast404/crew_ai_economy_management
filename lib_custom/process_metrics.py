"""Rule-based process metrics extraction from conversation transcripts.

Provides objective, programmatically extractable process variables to
complement LLM-based subjective evaluation scores. These metrics serve
as manipulation checks and convergent validity evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------
_TEMPORAL_MARKERS = re.compile(
    r"deadline|进度|里程碑|时间节点|排期|工期|倒计时|紧急|加快|提前|延期|周期"
)
_CROSS_REFERENCE = re.compile(
    r"同意.{0,4}(?:的|说|观点)|@|提到|正如.{0,4}说"
)
_CONFLICT_MARKERS = re.compile(
    r"不同意|反对|但是我认为|我觉得不|有不同看法"
)
_PRIORITY_CHANGE = re.compile(
    r"改为|调整优先|换方向|不做了|暂停|先放|临时加|插入"
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProcessMetrics:
    """Rule-based process metrics extracted from conversation transcript."""

    # Information sharing
    total_messages: int
    boss_message_count: int
    member_message_count: int
    avg_message_length: float

    # Temporal management behavior markers
    temporal_marker_count: int
    temporal_marker_density: float

    # Collaboration patterns
    cross_reference_count: int
    conflict_marker_count: int

    # Planning consistency
    priority_change_count: int

    # Phase completion
    phases_completed: int

    def to_dict(self) -> dict[str, float]:
        """Return all metrics as a flat dict of floats (for CSV export)."""
        return {
            "pm_total_messages": float(self.total_messages),
            "pm_boss_message_count": float(self.boss_message_count),
            "pm_member_message_count": float(self.member_message_count),
            "pm_avg_message_length": round(self.avg_message_length, 2),
            "pm_temporal_marker_count": float(self.temporal_marker_count),
            "pm_temporal_marker_density": round(self.temporal_marker_density, 4),
            "pm_cross_reference_count": float(self.cross_reference_count),
            "pm_conflict_marker_count": float(self.conflict_marker_count),
            "pm_priority_change_count": float(self.priority_change_count),
            "pm_phases_completed": float(self.phases_completed),
        }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _count_pattern(pattern: re.Pattern[str], text: str) -> int:
    """Count non-overlapping matches of *pattern* in *text*."""
    return len(pattern.findall(text))


def extract_process_metrics(
    messages: list[dict],
    boss_role_name: str,
    phases_completed: int = 0,
) -> ProcessMetrics:
    """Extract rule-based process metrics from a message list.

    Args:
        messages: List of dicts with at least ``role`` and ``content`` keys.
        boss_role_name: The role name string used for the boss agent so we
            can separate boss vs member messages.
        phases_completed: Number of phases the project completed.

    Returns:
        A frozen ``ProcessMetrics`` instance.
    """
    boss_texts: list[str] = []
    member_texts: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if boss_role_name and boss_role_name in role:
            boss_texts.append(content)
        else:
            member_texts.append(content)

    all_texts = boss_texts + member_texts
    total_messages = len(all_texts)
    total_length = sum(len(t) for t in all_texts)
    avg_length = total_length / total_messages if total_messages else 0.0

    boss_full = "\n".join(boss_texts)
    boss_length = len(boss_full)
    member_full = "\n".join(member_texts)

    temporal_count = _count_pattern(_TEMPORAL_MARKERS, boss_full)
    temporal_density = temporal_count / boss_length if boss_length else 0.0

    cross_ref_count = _count_pattern(_CROSS_REFERENCE, member_full)
    conflict_count = _count_pattern(_CONFLICT_MARKERS, member_full)
    priority_change = _count_pattern(_PRIORITY_CHANGE, boss_full)

    return ProcessMetrics(
        total_messages=total_messages,
        boss_message_count=len(boss_texts),
        member_message_count=len(member_texts),
        avg_message_length=avg_length,
        temporal_marker_count=temporal_count,
        temporal_marker_density=temporal_density,
        cross_reference_count=cross_ref_count,
        conflict_marker_count=conflict_count,
        priority_change_count=priority_change,
        phases_completed=phases_completed,
    )

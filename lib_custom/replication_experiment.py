"""Replication experiment runner — 72-run design matrix with dual evaluators.

Implements the 3×3 factorial design (Temporal Diversity × TTL) with
8 replications per cell, dual evaluator averaging, checkpoint resume,
and CSV/JSON export.

References:
    Mohammed & Nadkarni (2011). Temporal Diversity and Team Performance.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from lib_custom.okr_models import (
    EVALUATION_DIMENSIONS,
    OKRSet,
    format_okrs_for_prompt,
)
from lib_custom.team_generator import (
    TeamComposition,
    composition_to_members,
    generate_team_compositions,
)
from lib_custom.temporal_leadership import (
    TTL_LEVELS,
    TemporalLeadershipConfig,
    get_ttl_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------
class RunRecord(BaseModel):
    """Complete record for a single simulation run."""

    run_id: int
    cell_label: str  # e.g. "low_div × high_ttl"
    diversity_stratum: str  # "low" / "medium" / "high"
    ttl_level: str  # "low" / "medium" / "high"
    ttl_code: float  # 0.0 / 0.5 / 1.0
    composition_id: str
    type_ids: list[str]
    composite_blau: float

    # DV — 8 dimensions + overall (evaluator A, B, mean)
    scores_a: dict[str, float] = Field(default_factory=dict)
    scores_b: dict[str, float] = Field(default_factory=dict)
    scores_mean: dict[str, float] = Field(default_factory=dict)
    overall_a: float = 0.0
    overall_b: float = 0.0
    overall_mean: float = 0.0

    # Metadata
    elapsed_seconds: float = 0.0
    timestamp: str = ""
    status: str = "pending"  # pending / running / completed / failed
    error: str = ""


# ---------------------------------------------------------------------------
# Experiment design matrix
# ---------------------------------------------------------------------------
class ExperimentDesignMatrix(BaseModel):
    """72-run experimental design: 3 diversity strata × 3 TTL × 8 reps."""

    runs: list[RunRecord] = Field(default_factory=list)
    topic: str = "Q3产品发布计划讨论"
    project_type_id: str = "urgent_launch"
    team_size: int = 5
    num_rounds: int = 4
    seed: int = 42
    total_runs: int = 72
    completed_runs: int = 0


def build_design_matrix(
    topic: str = "Q3产品发布计划讨论",
    project_type_id: str = "urgent_launch",
    team_size: int = 5,
    num_rounds: int = 4,
    n_per_stratum: int = 8,
    seed: int = 42,
) -> ExperimentDesignMatrix:
    """Build the 72-run design matrix.

    3 diversity strata × 3 TTL levels × 8 replications = 72 runs.

    Args:
        topic: Discussion topic.
        project_type_id: Project type for OKR selection.
        team_size: Team size (default 5).
        num_rounds: Conversation rounds per run (default 4).
        n_per_stratum: Replications per cell (default 8).
        seed: Random seed for team composition sampling.

    Returns:
        ExperimentDesignMatrix with 72 RunRecords.
    """
    compositions = generate_team_compositions(
        n_per_stratum=n_per_stratum,
        team_size=team_size,
        seed=seed,
    )

    runs: list[RunRecord] = []
    run_id = 1

    diversity_levels = ["low", "medium", "high"]

    for div_stratum in diversity_levels:
        comps = compositions.get(div_stratum, [])
        for ttl_level in TTL_LEVELS:
            ttl_config = get_ttl_config(ttl_level)
            for comp in comps:
                cell_label = f"{div_stratum}_div × {ttl_level}_ttl"
                runs.append(
                    RunRecord(
                        run_id=run_id,
                        cell_label=cell_label,
                        diversity_stratum=div_stratum,
                        ttl_level=ttl_level,
                        ttl_code=ttl_config.code,
                        composition_id=comp.composition_id,
                        type_ids=comp.type_ids,
                        composite_blau=comp.diversity.composite,
                    )
                )
                run_id += 1

    total = len(runs)
    return ExperimentDesignMatrix(
        runs=runs,
        topic=topic,
        project_type_id=project_type_id,
        team_size=team_size,
        num_rounds=num_rounds,
        seed=seed,
        total_runs=total,
    )


# ---------------------------------------------------------------------------
# Dual evaluator prompt builders
# ---------------------------------------------------------------------------
_EVALUATOR_PERSPECTIVES: dict[str, dict[str, str]] = {
    "A": {
        "label": "评估者A",
        "focus": "团队绩效评估",
        "perspective": "绩效产出",
        "emphasis": "任务完成度、决策质量、目标对齐度等硬指标",
        "evidence_type": "绩效证据",
    },
    "B": {
        "label": "评估者B",
        "focus": "团队过程分析",
        "perspective": "团队行为过程",
        "emphasis": "协作过程、沟通模式、团队动态、成员互动质量",
        "evidence_type": "行为证据",
    },
}

_EVALUATOR_JSON_TEMPLATE = """\
{{
  "dimensions": {{
    "task_completion": {{"score": 0, "evidence": "..."}},
    "collaboration": {{"score": 0, "evidence": "..."}},
    "decision_quality": {{"score": 0, "evidence": "..."}},
    "innovation": {{"score": 0, "evidence": "..."}},
    "morale": {{"score": 0, "evidence": "..."}},
    "communication": {{"score": 0, "evidence": "..."}},
    "risk_management": {{"score": 0, "evidence": "..."}},
    "goal_alignment": {{"score": 0, "evidence": "..."}}
  }},
  "overall_score": 0,
  "key_findings": ["发现1", "发现2"],
  "boss_impact_analysis": "对领导者时间管理行为影响的分析",
  "recommendations": ["建议1", "建议2"]
}}"""


def _format_dimensions_text() -> str:
    """Format evaluation dimensions into numbered text list."""
    lines: list[str] = []
    for i, dim in enumerate(EVALUATION_DIMENSIONS.values(), 1):
        weight_pct = int(dim.weight * 100)
        lines.append(f"{i}. {dim.name_zh}({weight_pct}%): {dim.description}")
    return "\n".join(lines)


def build_evaluator_prompt(
    evaluator_id: str,
    okrs: OKRSet,
    full_conversation: str,
) -> str:
    """Build an evaluator prompt for either evaluator A or B.

    Args:
        evaluator_id: "A" (performance perspective) or "B" (behavioral perspective).
        okrs: OKR set for evaluation context.
        full_conversation: Full conversation transcript.

    Returns:
        Formatted evaluator prompt string.

    Raises:
        ValueError: If evaluator_id is not "A" or "B".
    """
    perspective = _EVALUATOR_PERSPECTIVES.get(evaluator_id)
    if perspective is None:
        raise ValueError(f"Unknown evaluator_id: {evaluator_id!r}. Must be 'A' or 'B'.")

    okrs_formatted = format_okrs_for_prompt(okrs)
    dimensions_text = _format_dimensions_text()

    return f"""你是一位独立客观的组织行为学研究者（{perspective['label']}），专注于{perspective['focus']}。
你需要从**{perspective['perspective']}**的角度，基于以下OKR目标和团队讨论记录，对团队绩效进行严格评估。

## 评估视角
你重点关注：{perspective['emphasis']}。
你的评分应严格基于讨论中可观察的{perspective['evidence_type']}。

## 团队OKR
{okrs_formatted}

## 讨论记录
{full_conversation}

## 评估要求
请从以下8个维度打分(0-100)，并给出评分依据:

{dimensions_text}

## 输出格式 (严格JSON，不要包含任何其他文字)
{_EVALUATOR_JSON_TEMPLATE}"""


# ---------------------------------------------------------------------------
# Score averaging
# ---------------------------------------------------------------------------
def average_evaluator_scores(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
) -> dict[str, float]:
    """Average scores from two evaluators, dimension by dimension.

    Args:
        scores_a: Dimension scores from evaluator A.
        scores_b: Dimension scores from evaluator B.

    Returns:
        Averaged scores per dimension.
    """
    all_dims = set(scores_a.keys()) | set(scores_b.keys())
    result: dict[str, float] = {}
    for dim in all_dims:
        a = scores_a.get(dim, 0.0)
        b = scores_b.get(dim, 0.0)
        result[dim] = round((a + b) / 2.0, 2)
    return result


# ---------------------------------------------------------------------------
# CSV / JSON export
# ---------------------------------------------------------------------------
_CSV_COLUMNS = [
    "run_id", "cell_label", "diversity_stratum", "ttl_level", "ttl_code",
    "composition_id", "composite_blau",
    "task_completion", "collaboration", "decision_quality", "innovation",
    "morale", "communication", "risk_management", "goal_alignment",
    "overall_mean", "overall_a", "overall_b",
    "elapsed_seconds", "timestamp", "status",
]


def export_to_csv(design: ExperimentDesignMatrix, filepath: str) -> str:
    """Export completed runs to CSV.

    Args:
        design: The experiment design matrix with completed runs.
        filepath: Output file path.

    Returns:
        The filepath written.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for run in design.runs:
            if run.status != "completed":
                continue
            row: dict[str, Any] = {
                "run_id": run.run_id,
                "cell_label": run.cell_label,
                "diversity_stratum": run.diversity_stratum,
                "ttl_level": run.ttl_level,
                "ttl_code": run.ttl_code,
                "composition_id": run.composition_id,
                "composite_blau": run.composite_blau,
                "overall_mean": run.overall_mean,
                "overall_a": run.overall_a,
                "overall_b": run.overall_b,
                "elapsed_seconds": run.elapsed_seconds,
                "timestamp": run.timestamp,
                "status": run.status,
            }
            # Add dimension scores (mean)
            for dim_id in EVALUATION_DIMENSIONS:
                row[dim_id] = run.scores_mean.get(dim_id, 0.0)
            writer.writerow(row)

    logger.info("CSV exported: %s", filepath)
    return filepath


def export_to_json(design: ExperimentDesignMatrix, filepath: str) -> str:
    """Export the full design matrix to JSON (for checkpoint / resume).

    Args:
        design: The experiment design matrix.
        filepath: Output file path.

    Returns:
        The filepath written.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(design.model_dump(), f, ensure_ascii=False, indent=2)
    logger.info("JSON checkpoint exported: %s", filepath)
    return filepath


def load_checkpoint(filepath: str) -> ExperimentDesignMatrix | None:
    """Load a previously saved experiment checkpoint.

    Args:
        filepath: Path to the JSON checkpoint file.

    Returns:
        ExperimentDesignMatrix if file exists and is valid, else None.
    """
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return ExperimentDesignMatrix.model_validate(data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to load checkpoint %s: %s", filepath, e)
        return None


# ---------------------------------------------------------------------------
# Experiment runner (orchestration)
# ---------------------------------------------------------------------------
def _update_run_at_index(
    runs: list[RunRecord],
    index: int,
    updated: RunRecord,
) -> list[RunRecord]:
    """Return a new list with run at `index` replaced by `updated`."""
    return [updated if i == index else r for i, r in enumerate(runs)]


def run_replication(
    design: ExperimentDesignMatrix,
    run_single_fn: Callable[[RunRecord, ExperimentDesignMatrix], RunRecord],
    checkpoint_path: str | None = None,
    on_run_complete: Callable[[RunRecord, int, int], None] | None = None,
) -> ExperimentDesignMatrix:
    """Execute the replication experiment with checkpoint resume.

    This function orchestrates 72 runs by delegating each run to
    `run_single_fn`. It handles:
    - Skipping already-completed runs (checkpoint resume)
    - Saving checkpoints after each completed run
    - Progress callbacks

    Returns a **new** ExperimentDesignMatrix (no mutation of the input).

    Args:
        design: The experiment design matrix.
        run_single_fn: Function that executes a single run and returns
            an updated RunRecord. Signature: (record, design) -> record.
        checkpoint_path: Path for JSON checkpoint file. None = no checkpoint.
        on_run_complete: Optional callback(record, completed, total).

    Returns:
        New ExperimentDesignMatrix with completed runs.
    """
    total = len(design.runs)
    runs = list(design.runs)
    completed = sum(1 for r in runs if r.status == "completed")

    for i, run in enumerate(runs):
        if run.status == "completed":
            continue

        logger.info(
            "Starting run %d/%d: %s (comp=%s)",
            run.run_id, total, run.cell_label, run.composition_id,
        )

        try:
            updated_run = run_single_fn(run, design)
            runs = _update_run_at_index(runs, i, updated_run)
            completed += 1
        except Exception as e:
            logger.error("Run %d failed: %s", run.run_id, e)
            failed_run = run.model_copy(
                update={
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            runs = _update_run_at_index(runs, i, failed_run)

        # Build updated design for checkpoint + callback
        design = design.model_copy(
            update={"runs": runs, "completed_runs": completed}
        )

        if checkpoint_path:
            export_to_json(design, checkpoint_path)

        if on_run_complete:
            on_run_complete(runs[i], completed, total)

    return design

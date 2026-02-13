"""Experiment runner for thesis experiments.

Orchestrates running simulations under different boss types,
collects transcripts and evaluations, and produces comparison analysis.

Extended to support TTL-aware experiments with variable team size.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, Field

from lib_custom.chat_store import ChatMessageStore
from lib_custom.default_team import create_default_team
from lib_custom.okr_models import EVALUATION_DIMENSIONS, OKRSet
from lib_custom.personality_types import TeamConfig


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / result models
# ---------------------------------------------------------------------------
class ExperimentConfig(BaseModel):
    """Configuration for a thesis experiment run."""

    topic: str = Field(..., min_length=3)
    okrs: OKRSet
    team: TeamConfig
    boss_types: list[str] = Field(default_factory=lambda: ["time_master", "time_chaos"])
    num_rounds: int = Field(default=3, ge=1, le=10)
    team_size: int = Field(default=12, ge=2, le=20)
    ttl_level: str | None = Field(default=None, pattern=r"^(low|medium|high)$")
    config: dict = Field(
        default_factory=lambda: {
            "agent_timeout": 120,
            "max_iterations": 5,
            "context_window": 4,
            "stream": True,
        }
    )


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    score: int = Field(ge=0, le=100)
    evidence: str = ""


class EvaluationResult(BaseModel):
    """Parsed evaluation output from the evaluator agent."""

    dimensions: dict[str, DimensionScore] = Field(default_factory=dict)
    overall_score: float = 0.0
    key_findings: list[str] = Field(default_factory=list)
    boss_impact_analysis: str = ""
    recommendations: list[str] = Field(default_factory=list)


class SingleRunResult(BaseModel):
    """Result of a single experiment run (one boss type)."""

    boss_type_id: str
    transcript: str = ""
    evaluation_raw: str = ""
    evaluation: EvaluationResult = Field(default_factory=EvaluationResult)
    messages: list[dict] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


class ThesisExperimentResult(BaseModel):
    """Complete result of a thesis experiment (both boss types)."""

    config: ExperimentConfig
    runs: dict[str, SingleRunResult] = Field(default_factory=dict)
    comparison_summary: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Flat run record for CSV export (replication experiment compatible)
# ---------------------------------------------------------------------------
class FlatRunRecord(BaseModel):
    """Flat record for a single experiment run, suitable for CSV export."""

    run_id: int = 0
    boss_type_id: str = ""
    ttl_level: str = ""
    team_size: int = 12
    diversity_composite: float = 0.0
    overall_score: float = 0.0
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    elapsed_seconds: float = 0.0
    timestamp: str = ""


def export_results_to_csv(
    records: list[FlatRunRecord],
    filepath: str,
) -> str:
    """Export flat run records to CSV.

    Args:
        records: List of run records.
        filepath: Output CSV path.

    Returns:
        The filepath written.
    """
    if not records:
        return filepath

    dim_ids = sorted(EVALUATION_DIMENSIONS.keys())
    columns = [
        "run_id", "boss_type_id", "ttl_level", "team_size",
        "diversity_composite", "overall_score",
        *dim_ids,
        "elapsed_seconds", "timestamp",
    ]

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for rec in records:
            row: dict[str, Any] = {
                "run_id": rec.run_id,
                "boss_type_id": rec.boss_type_id,
                "ttl_level": rec.ttl_level,
                "team_size": rec.team_size,
                "diversity_composite": rec.diversity_composite,
                "overall_score": rec.overall_score,
                "elapsed_seconds": rec.elapsed_seconds,
                "timestamp": rec.timestamp,
            }
            for dim_id in dim_ids:
                row[dim_id] = rec.dimension_scores.get(dim_id, 0.0)
            writer.writerow(row)

    logger.info("Results CSV exported: %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Transcript extraction
# ---------------------------------------------------------------------------
def extract_transcript(store: ChatMessageStore) -> str:
    """Extract completed conversation text from a message store."""
    parts = [
        f"[{msg.role}]: {msg.content}"
        for msg in store.get_all()
        if msg.msg_type == "completed" and msg.role != "system"
    ]
    return "\n\n".join(parts)


def extract_messages_as_dicts(store: ChatMessageStore) -> list[dict]:
    """Extract all messages as serializable dicts."""
    return [
        {
            "role": msg.role,
            "content": msg.content,
            "msg_type": msg.msg_type,
            "timestamp": msg.timestamp,
        }
        for msg in store.get_all()
    ]


# ---------------------------------------------------------------------------
# Evaluation parsing
# ---------------------------------------------------------------------------
def parse_evaluation(raw_text: str) -> EvaluationResult:
    """Parse the evaluator's JSON output into an EvaluationResult.

    Handles cases where JSON is embedded in markdown code blocks.
    Returns a default EvaluationResult if parsing fails.
    """
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, re.DOTALL)
    json_str = json_match.group(1) if json_match else raw_text.strip()

    # Also try bare JSON
    if not json_str.startswith("{"):
        brace_start = json_str.find("{")
        brace_end = json_str.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_str[brace_start : brace_end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning("Failed to parse evaluation JSON: %s...", raw_text[:200])
        return EvaluationResult()

    # Parse dimensions
    dimensions: dict[str, DimensionScore] = {}
    raw_dims = data.get("dimensions", {})
    if not isinstance(raw_dims, dict):
        raw_dims = {}
    for dim_id, dim_data in raw_dims.items():
        if isinstance(dim_data, dict):
            score = max(0, min(100, int(dim_data.get("score") or 0)))
            dimensions[dim_id] = DimensionScore(
                score=score,
                evidence=str(dim_data.get("evidence", "")),
            )

    return EvaluationResult(
        dimensions=dimensions,
        overall_score=float(data.get("overall_score", 0)),
        key_findings=list(data.get("key_findings", [])),
        boss_impact_analysis=str(data.get("boss_impact_analysis", "")),
        recommendations=list(data.get("recommendations", [])),
    )


def find_evaluator_output(store: ChatMessageStore) -> str:
    """Find the evaluator's output from the message store.

    The evaluator is typically the last "completed" non-system message
    containing JSON-like content.
    """
    messages = store.get_all()
    # Search from the end for evaluator output (contains JSON)
    for msg in reversed(messages):
        if msg.msg_type != "completed":
            continue
        if msg.role == "system":
            continue
        # Look for JSON markers
        if "{" in msg.content and "dimensions" in msg.content:
            return msg.content
    # Fallback: return last completed message
    for msg in reversed(messages):
        if msg.msg_type == "completed" and msg.role != "system":
            return msg.content
    return ""


# ---------------------------------------------------------------------------
# Default experiment factory
# ---------------------------------------------------------------------------
def create_default_experiment(
    topic: str = "Q3产品发布计划讨论",
    project_type_id: str = "urgent_launch",
    num_rounds: int = 3,
) -> ExperimentConfig:
    """Create a default experiment config with preset OKR and 12-member team."""
    from lib_custom.okr_models import DEFAULT_OKRS

    okrs = DEFAULT_OKRS.get(project_type_id)
    if okrs is None:
        raise ValueError(f"Unknown project type: {project_type_id}")

    team = create_default_team("time_master")  # boss_type_id will vary per run

    return ExperimentConfig(
        topic=topic,
        okrs=okrs,
        team=team,
        boss_types=["time_master", "time_chaos"],
        num_rounds=num_rounds,
    )


# ---------------------------------------------------------------------------
# Comparison prompt
# ---------------------------------------------------------------------------
COMPARISON_PROMPT = """你是组织行为学研究者。请对比分析同一团队在两种不同时间管理风格的老板领导下的绩效差异。

## 讨论主题
{topic}

## time_master（高效时间管理者）条件下的评估结果
{eval_master}

## time_chaos（混乱时间管理者）条件下的评估结果
{eval_chaos}

请从以下角度进行对比分析:

1. **整体绩效差异**: 两种条件下的加权总分对比及其意义
2. **维度差异分析**: 哪些维度差异最大，为什么
3. **领导风格影响机制**: time_master和time_chaos分别如何影响团队行为
4. **调节效应**: 项目类型如何调节领导风格对绩效的影响
5. **理论贡献**: 对时间领导力理论的启示
6. **实践建议**: 对组织管理的具体建议

请提供结构化的对比分析报告，适合直接用于论文写作。"""


def build_comparison_summary_prompt(
    topic: str,
    eval_master: str,
    eval_chaos: str,
) -> str:
    """Build a prompt for cross-condition comparison analysis."""
    return COMPARISON_PROMPT.format(
        topic=topic,
        eval_master=eval_master,
        eval_chaos=eval_chaos,
    )

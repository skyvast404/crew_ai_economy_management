"""OKR data models and evaluation dimensions for thesis experiments.

Defines KeyResult, OKRSet, EvaluationDimension, and 3 default OKR templates
mapped to project types (urgent_launch, long_term_platform, exploratory_prototype).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------
class KeyResult(BaseModel):
    """A single measurable key result within an OKR set."""

    id: str = Field(..., min_length=1, max_length=60)
    description: str = Field(..., min_length=3)
    target: str = Field(..., min_length=1)
    weight: float = Field(..., gt=0, le=1.0)
    unit: str = Field(default="", max_length=30)


class OKRSet(BaseModel):
    """Objective + Key Results for a specific project type."""

    objective: str = Field(..., min_length=5)
    key_results: list[KeyResult] = Field(..., min_length=1)
    project_type_id: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_weights_sum(self) -> OKRSet:
        total = sum(kr.weight for kr in self.key_results)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Key result weights must sum to 1.0, got {total:.2f}"
            )
        return self


class EvaluationDimension(BaseModel):
    """A single evaluation dimension for team performance scoring."""

    id: str = Field(..., min_length=1, max_length=60)
    name_zh: str = Field(..., min_length=1)
    description: str = Field(..., min_length=5)
    weight: float = Field(..., gt=0, le=1.0)


# ---------------------------------------------------------------------------
# 8 evaluation dimensions (weights sum to 1.0)
# ---------------------------------------------------------------------------
EVALUATION_DIMENSIONS: dict[str, EvaluationDimension] = {
    "task_completion": EvaluationDimension(
        id="task_completion",
        name_zh="任务完成质量",
        description="团队讨论是否围绕OKR展开，行动计划是否清晰可执行",
        weight=0.20,
    ),
    "collaboration": EvaluationDimension(
        id="collaboration",
        name_zh="协作效率",
        description="成员间是否有效配合，信息是否充分共享",
        weight=0.15,
    ),
    "decision_quality": EvaluationDimension(
        id="decision_quality",
        name_zh="决策质量",
        description="决策过程是否理性，是否考虑多方意见",
        weight=0.15,
    ),
    "innovation": EvaluationDimension(
        id="innovation",
        name_zh="创新表现",
        description="是否提出创新方案，是否有建设性分歧",
        weight=0.10,
    ),
    "morale": EvaluationDimension(
        id="morale",
        name_zh="团队士气",
        description="成员参与度、积极性、心理安全感",
        weight=0.10,
    ),
    "communication": EvaluationDimension(
        id="communication",
        name_zh="沟通有效性",
        description="信息传递是否清晰，是否有误解或冲突",
        weight=0.10,
    ),
    "risk_management": EvaluationDimension(
        id="risk_management",
        name_zh="风险应对",
        description="是否识别风险，应对策略是否合理",
        weight=0.10,
    ),
    "goal_alignment": EvaluationDimension(
        id="goal_alignment",
        name_zh="目标对齐度",
        description="讨论是否始终围绕核心目标",
        weight=0.10,
    ),
}


def get_dimension_weights() -> dict[str, float]:
    """Return dimension_id -> weight mapping."""
    return {d.id: d.weight for d in EVALUATION_DIMENSIONS.values()}


def validate_dimensions_weights() -> bool:
    """Check that all dimension weights sum to 1.0."""
    total = sum(d.weight for d in EVALUATION_DIMENSIONS.values())
    return abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# 3 default OKR templates
# ---------------------------------------------------------------------------
DEFAULT_OKRS: dict[str, OKRSet] = {
    "urgent_launch": OKRSet(
        objective="Q3完成产品2.0上线，确保高质量交付",
        project_type_id="urgent_launch",
        key_results=[
            KeyResult(
                id="ul_kr1",
                description="核心功能开发完成率",
                target="≥95%",
                weight=0.35,
                unit="%",
            ),
            KeyResult(
                id="ul_kr2",
                description="上线延期天数",
                target="≤3天",
                weight=0.30,
                unit="天",
            ),
            KeyResult(
                id="ul_kr3",
                description="上线后关键Bug率",
                target="≤2%",
                weight=0.20,
                unit="%",
            ),
            KeyResult(
                id="ul_kr4",
                description="用户验收测试通过率",
                target="≥90%",
                weight=0.15,
                unit="%",
            ),
        ],
    ),
    "long_term_platform": OKRSet(
        objective="年度技术平台升级，构建可持续演进的技术底座",
        project_type_id="long_term_platform",
        key_results=[
            KeyResult(
                id="ltp_kr1",
                description="新架构模块覆盖率",
                target="≥80%",
                weight=0.30,
                unit="%",
            ),
            KeyResult(
                id="ltp_kr2",
                description="技术债务减少比例",
                target="≥50%",
                weight=0.25,
                unit="%",
            ),
            KeyResult(
                id="ltp_kr3",
                description="API文档覆盖率",
                target="≥90%",
                weight=0.25,
                unit="%",
            ),
            KeyResult(
                id="ltp_kr4",
                description="平台可用性SLA达标",
                target="≥99.9%",
                weight=0.20,
                unit="%",
            ),
        ],
    ),
    "exploratory_prototype": OKRSet(
        objective="新业务方向快速验证，发现可行的增长点",
        project_type_id="exploratory_prototype",
        key_results=[
            KeyResult(
                id="ep_kr1",
                description="完成原型方案数量",
                target="≥3个",
                weight=0.30,
                unit="个",
            ),
            KeyResult(
                id="ep_kr2",
                description="用户测试参与人数",
                target="≥50人",
                weight=0.25,
                unit="人",
            ),
            KeyResult(
                id="ep_kr3",
                description="可行性分析报告完成",
                target="100%",
                weight=0.25,
                unit="%",
            ),
            KeyResult(
                id="ep_kr4",
                description="潜在商业价值评估得分",
                target="≥70分",
                weight=0.20,
                unit="分",
            ),
        ],
    ),
}


def get_okr_for_project(project_type_id: str) -> OKRSet | None:
    """Look up a default OKR template by project type ID."""
    return DEFAULT_OKRS.get(project_type_id)


def format_okrs_for_prompt(okr_set: OKRSet) -> str:
    """Format an OKRSet into a human-readable string for LLM prompts."""
    lines = [f"目标 (Objective): {okr_set.objective}", "", "关键结果 (Key Results):"]
    for i, kr in enumerate(okr_set.key_results, 1):
        weight_pct = int(kr.weight * 100)
        lines.append(
            f"  KR{i}. {kr.description}: {kr.target}"
            f" (权重: {weight_pct}%, 单位: {kr.unit or '无'})"
        )
    return "\n".join(lines)

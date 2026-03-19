"""Project phase definitions for lifecycle-based simulation.

Each OKR project type maps to a sequence of phases. Each phase includes:
- Identification and description
- Boss prompt templates (differentiated by leadership style)
- Member prompt template (progress reporting)
- Status checker prompt (determines whether to continue or complete)
- Manipulation check prompt (behavioral coding of boss utterances)

Boss behavior differentiation:
    time_master: personality-driven structured management (emergent from backstory)
    time_chaos: personality-driven chaotic management (emergent from backstory)
    time_neutral: baseline neutral management (no strong time orientation)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Phase model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectPhase:
    """A single phase in a project lifecycle."""

    phase_id: str
    name_zh: str
    description: str
    deliverables: str


# ---------------------------------------------------------------------------
# Phase sequences per project type
# ---------------------------------------------------------------------------
PHASE_SEQUENCES: dict[str, list[ProjectPhase]] = {
    "urgent_launch": [
        ProjectPhase(
            phase_id="requirement_confirm",
            name_zh="需求确认",
            description="明确产品需求范围，确认优先级，冻结需求基线",
            deliverables="需求文档、优先级矩阵、验收标准",
        ),
        ProjectPhase(
            phase_id="dev_sprint",
            name_zh="开发冲刺",
            description="核心功能开发，每日站会同步进度，管控技术风险",
            deliverables="可运行的核心功能模块、代码审查记录",
        ),
        ProjectPhase(
            phase_id="test_verify",
            name_zh="测试验证",
            description="全面测试，修复缺陷，性能与安全验证",
            deliverables="测试报告、缺陷修复清单、性能指标",
        ),
        ProjectPhase(
            phase_id="launch_deploy",
            name_zh="上线部署",
            description="制定上线计划，灰度发布，监控回滚预案",
            deliverables="上线清单、监控仪表盘、回滚方案",
        ),
    ],
    "long_term_platform": [
        ProjectPhase(
            phase_id="arch_design",
            name_zh="架构设计",
            description="技术选型，模块划分，接口契约设计",
            deliverables="架构设计文档、技术选型报告、接口定义",
        ),
        ProjectPhase(
            phase_id="module_dev",
            name_zh="模块开发",
            description="按模块并行开发，管控依赖关系，持续集成",
            deliverables="各模块代码、单元测试、CI流水线",
        ),
        ProjectPhase(
            phase_id="integration_test",
            name_zh="集成测试",
            description="跨模块联调，端到端测试，性能压测",
            deliverables="集成测试报告、性能基线、兼容性矩阵",
        ),
        ProjectPhase(
            phase_id="doc_delivery",
            name_zh="文档交付",
            description="API文档、运维手册、培训材料编写与评审",
            deliverables="API文档、运维手册、培训材料",
        ),
    ],
    "exploratory_prototype": [
        ProjectPhase(
            phase_id="opportunity_explore",
            name_zh="机会探索",
            description="市场调研，竞品分析，用户痛点挖掘",
            deliverables="调研报告、机会地图、用户画像",
        ),
        ProjectPhase(
            phase_id="prototype_build",
            name_zh="原型构建",
            description="快速搭建最小可行原型，验证核心假设",
            deliverables="可交互原型、核心功能演示",
        ),
        ProjectPhase(
            phase_id="user_validate",
            name_zh="用户验证",
            description="用户测试，收集反馈，迭代优化",
            deliverables="用户测试报告、反馈汇总、迭代计划",
        ),
        ProjectPhase(
            phase_id="feasibility_assess",
            name_zh="可行性评估",
            description="技术可行性、商业价值、资源投入综合评估",
            deliverables="可行性评估报告、商业计划初稿",
        ),
    ],
}


def get_phases_for_project(project_type_id: str) -> list[ProjectPhase]:
    """Return the phase sequence for a project type."""
    phases = PHASE_SEQUENCES.get(project_type_id)
    if phases is None:
        raise ValueError(f"Unknown project type: {project_type_id}")
    return phases


def get_phase_names_zh(project_type_id: str) -> list[str]:
    """Return Chinese phase names for display."""
    return [p.name_zh for p in get_phases_for_project(project_type_id)]


# ---------------------------------------------------------------------------
# Boss prompt templates (differentiated by leadership style)
# ---------------------------------------------------------------------------
_BOSS_PHASE_MASTER = """你是{role_name}。

你的目标: {goal}
你的背景: {backstory}
你的性格: {personality}
你的沟通风格: {communication_style}

## 当前项目阶段: {phase_name} ({phase_idx}/{total_phases})
{phase_description}
预期交付物: {deliverables}

## 项目OKR
{okr_summary}

{previous_summary}

## 你的任务
请基于你的管理风格和性格特征，对本阶段的工作进行指导。
你自然倾向于有条理地推进工作。
发言控制在200字以内。"""

_BOSS_PHASE_CHAOS = """你是{role_name}。

你的目标: {goal}
你的背景: {backstory}
你的性格: {personality}
你的沟通风格: {communication_style}

## 当前项目阶段: {phase_name} ({phase_idx}/{total_phases})
{phase_description}
预期交付物: {deliverables}

## 项目OKR
{okr_summary}

{previous_summary}

## 你的任务
请基于你的管理风格和性格特征，对本阶段的工作进行指导。
你可能会临时调整优先级、改变之前的决定、或关注与当前阶段不太相关的事务。
发言控制在200字以内。"""

_BOSS_PHASE_NEUTRAL = """你是{role_name}。

你的目标: {goal}
你的背景: {backstory}
你的性格: {personality}
你的沟通风格: {communication_style}

## 当前项目阶段: {phase_name} ({phase_idx}/{total_phases})
{phase_description}
预期交付物: {deliverables}

## 项目OKR
{okr_summary}

{previous_summary}

## 你的任务
请基于你的管理风格，对本阶段的工作进行指导。
发言控制在200字以内。"""


def _generate_disruption(
    phase_id: str, phase_idx: int, seed: int,
) -> str:
    """Generate a deterministic disruption event for time_chaos boss.

    .. deprecated::
        No longer called by ``build_boss_phase_prompt``. Kept for backward
        compatibility / potential rollback. Chaos boss behavior is now driven
        entirely by backstory personality traits.

    Uses a hash of (seed, phase_id, phase_idx) to pick from a pool,
    ensuring reproducibility across runs with the same seed.
    """
    disruptions = [
        "客户刚刚提出了新的需求变更，要求在不延期的前提下增加一个核心功能模块。",
        "团队中一名关键成员突然请假一周，需要紧急重新分配其负责的工作。",
        "上级领导临时插入一个紧急任务，要求团队在本阶段同时处理另一个项目的紧急修复。",
        "原定的技术方案被否决，需要在短时间内评估并切换到备选方案。",
        "预算突然被削减20%，需要重新评估哪些交付物可以简化或推迟。",
        "竞争对手发布了类似产品，领导要求加速项目进度并增加差异化功能。",
        "测试环境出现严重故障，预计需要2天修复，但deadline不变。",
        "合作方突然变更了接口协议，所有对接模块需要重新适配。",
        "公司战略调整，项目优先级从P0降为P1，部分资源将被抽调支援其他项目。",
        "用户反馈收集发现原定方向与实际需求有较大偏差，需要紧急调整。",
    ]
    hash_input = f"{seed}:{phase_id}:{phase_idx}"
    hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
    return disruptions[hash_val % len(disruptions)]


def build_boss_phase_prompt(
    boss_type_id: str,
    role_name: str,
    goal: str,
    backstory: str,
    personality: str,
    communication_style: str,
    phase: ProjectPhase,
    phase_idx: int,
    total_phases: int,
    okr_summary: str,
    previous_summary: str = "",
    seed: int = 42,
) -> str:
    """Build the boss's phase directive prompt.

    Args:
        boss_type_id: "time_master", "time_chaos", or "time_neutral".
        role_name, goal, backstory, personality, communication_style: Boss role fields.
        phase: Current project phase.
        phase_idx: 1-based phase index.
        total_phases: Total number of phases.
        okr_summary: Formatted OKR text.
        previous_summary: Summary from previous phase (empty for phase 1).
        seed: Kept for backward compatibility (no longer used).
    """
    prev_section = (
        f"## 上一阶段总结\n{previous_summary}"
        if previous_summary
        else "## 项目刚启动，这是第一个阶段。"
    )

    template_map = {
        "time_master": _BOSS_PHASE_MASTER,
        "time_chaos": _BOSS_PHASE_CHAOS,
        "time_neutral": _BOSS_PHASE_NEUTRAL,
    }
    template = template_map.get(boss_type_id, _BOSS_PHASE_MASTER)

    return template.format(
        role_name=role_name,
        goal=goal,
        backstory=backstory,
        personality=personality,
        communication_style=communication_style,
        phase_name=phase.name_zh,
        phase_idx=phase_idx,
        total_phases=total_phases,
        phase_description=phase.description,
        deliverables=phase.deliverables,
        okr_summary=okr_summary,
        previous_summary=prev_section,
    )


# ---------------------------------------------------------------------------
# Member prompt template
# ---------------------------------------------------------------------------
_MEMBER_PHASE_PROMPT = """你是{role_name}。

你的性格: {personality}
你的沟通风格: {communication_style}
你的情绪倾向: {emotional_tendency}
你的价值观: {values}

## 当前项目阶段: {phase_name} ({phase_idx}/{total_phases})
{phase_description}

## 项目OKR
{okr_summary}

## 你的任务
请根据老板的指令和其他成员的汇报，汇报你在本阶段的工作情况:
1. 你负责的任务完成进展
2. 遇到的障碍或风险
3. 需要其他成员协作配合的事项
4. 对下一步工作的建议

保持你的角色特征和沟通风格。发言控制在150字以内。"""


def build_member_phase_prompt(
    role_name: str,
    personality: str,
    communication_style: str,
    emotional_tendency: str,
    values: str,
    phase: ProjectPhase,
    phase_idx: int,
    total_phases: int,
    okr_summary: str,
) -> str:
    """Build a member's phase progress report prompt."""
    return _MEMBER_PHASE_PROMPT.format(
        role_name=role_name,
        personality=personality,
        communication_style=communication_style,
        emotional_tendency=emotional_tendency,
        values=values,
        phase_name=phase.name_zh,
        phase_idx=phase_idx,
        total_phases=total_phases,
        phase_description=phase.description,
        okr_summary=okr_summary,
    )


# ---------------------------------------------------------------------------
# Status checker prompt
# ---------------------------------------------------------------------------
_STATUS_CHECKER_PROMPT = """你是项目状态评估专家。请根据本阶段所有成员的工作汇报，判断项目是否应该继续推进到下一阶段。

## 当前阶段: {phase_name} ({phase_idx}/{total_phases})
预期交付物: {deliverables}

## 判断标准
- 如果本阶段的核心交付物已基本完成，团队可以进入下一阶段 → 输出 "continue"
- 如果整个项目的OKR目标已经达成，无需继续 → 输出 "complete"
- 如果这是最后一个阶段且交付物已完成 → 输出 "complete"

## 输出格式 (严格JSON，不要包含任何其他文字)
{{"project_status": "continue"}}
或
{{"project_status": "complete"}}"""


def build_status_checker_prompt(
    phase: ProjectPhase,
    phase_idx: int,
    total_phases: int,
) -> str:
    """Build the status checker prompt for a given phase."""
    return _STATUS_CHECKER_PROMPT.format(
        phase_name=phase.name_zh,
        phase_idx=phase_idx,
        total_phases=total_phases,
        deliverables=phase.deliverables,
    )


# ---------------------------------------------------------------------------
# Manipulation check prompt (behavioral coding of boss utterances)
# ---------------------------------------------------------------------------
_MANIPULATION_CHECK_PROMPT = """你是组织行为学编码专家。请对以下项目中领导者的所有发言进行行为编码。

## 编码维度（基于 Mohammed & Nadkarni 2011 时间领导力量表）
1. time_urgency: 时间紧迫感表达（提及 deadline、紧急、加快等）
2. time_allocation: 时间分配行为（为任务设定时间框架、分配时间节点）
3. schedule_coordination: 进度协调行为（同步进度、协调节奏差异）
4. priority_consistency: 优先级一致性（优先级是否前后一致、是否频繁变动）
5. plan_clarity: 计划清晰度（指令是否明确、是否有歧义或矛盾）
6. disruption_frequency: 干扰频率（临时插入任务、改变方向的次数）

## 领导者发言记录
{boss_utterances}

## 输出格式（严格 JSON）
{{"dimensions": {{"time_urgency": {{"score": 0, "count": 0, "examples": []}}, "time_allocation": {{"score": 0, "count": 0, "examples": []}}, "schedule_coordination": {{"score": 0, "count": 0, "examples": []}}, "priority_consistency": {{"score": 0, "count": 0, "examples": []}}, "plan_clarity": {{"score": 0, "count": 0, "examples": []}}, "disruption_frequency": {{"score": 0, "count": 0, "examples": []}}}}}}

score: 0-100，count: 该行为出现的次数，examples: 引用原文的具体片段（最多3条）。"""


def build_manipulation_check_prompt(boss_utterances: str) -> str:
    """Build the manipulation check prompt with boss utterances."""
    return _MANIPULATION_CHECK_PROMPT.format(boss_utterances=boss_utterances)

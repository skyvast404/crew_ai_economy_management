"""📝 论文实验 — Thesis Experiment Page

Streamlit page for running controlled experiments comparing
time_master vs time_chaos boss types on team performance.

Uses phase-based project lifecycle simulation where teams progress
through project stages (e.g., requirements → development → testing → launch).

4 Tabs:
  1. 实验配置 — topic, project type, phase count, OKR editor, team preview
  2. 实验运行 — dual-column progress, streaming output
  3. 绩效评估 — 8-dimension radar chart, detail table, findings
  4. 论文素材导出 — structured data, charts, markdown report
"""

import logging
import re
import threading
import time

from crewai import Agent, Crew, Process, Task
from crewai.types.streaming import CrewStreamingOutput, StreamChunkType
from lib_custom.chat_store import ChatMessageStore
from lib_custom.default_team import DEFAULT_TEAM_MEMBERS, create_default_team
from lib_custom.experiment_runner import (
    DimensionScore,
    ExperimentConfig,
    SingleRunResult,
    ThesisExperimentResult,
    build_comparison_summary_prompt,
    extract_messages_as_dicts,
    extract_transcript,
    find_evaluator_output,
    parse_evaluation,
)
from lib_custom.llm_config import create_primary_llm
from lib_custom.okr_models import (
    DEFAULT_OKRS,
    EVALUATION_DIMENSIONS,
    OKRSet,
    format_okrs_for_prompt,
)
from lib_custom.personality_types import BOSS_TYPES, PERSONALITY_TYPES
from lib_custom.project_phases import get_phase_names_zh
from lib_custom.runtime_state import (
    STATE as RUNTIME_STATE,
    ensure_event_handlers_registered,
)
from lib_custom.thesis_crew_builder import build_thesis_crew
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st


logger = logging.getLogger(__name__)

_BOSS_ICONS: dict[str, str] = {
    "time_master": "🏆",
    "time_chaos": "🌪️",
    "time_neutral": "⚖️",
}

ensure_event_handlers_registered()

st.set_page_config(page_title="📝 论文实验", page_icon="📝", layout="wide")
st.title("📝 论文实验：时间驾驭能力与团队绩效")
st.caption("对比 time_master / time_neutral / time_chaos 三种老板对同一团队项目执行绩效的影响")


# ---------------------------------------------------------------------------
# Helper functions (must be defined before tab code uses them)
# ---------------------------------------------------------------------------
def _render_partial_messages(store: ChatMessageStore):
    """Render a compact view of messages from a store."""
    messages = store.get_all()
    completed = [
        m for m in messages if m.msg_type == "completed" and m.role != "system"
    ]
    if not completed:
        if store.error:
            st.error(f"错误: {store.error}")
        elif store.done:
            st.info("无输出")
        else:
            st.caption("等待中...")
        return

    st.caption(f"共 {len(completed)} 条发言")
    for msg in completed[-4:]:
        preview = (
            msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
        )
        st.markdown(f"**{msg.role}**: {preview}")


def _render_full_messages(store: ChatMessageStore):
    """Render all messages from a store in a scrollable container."""
    messages = store.get_all()
    completed = [
        m for m in messages if m.msg_type == "completed" and m.role != "system"
    ]
    if not completed:
        if store.error:
            st.error(f"错误: {store.error}")
        elif store.done:
            st.info("无输出")
        else:
            st.caption("等待中...")
        return

    st.caption(f"共 {len(completed)} 条发言")
    with st.container(height=400):
        for msg in completed:
            st.markdown(f"**{msg.role}**: {msg.content}")


def _build_experiment_result():
    """Parse stores into ThesisExperimentResult and save to session state."""
    stores = st.session_state.exp_stores
    if not stores:
        return

    runs: dict[str, SingleRunResult] = {}
    for boss_type_id in ["time_master", "time_chaos", "time_neutral"]:
        store = stores.get(boss_type_id)
        if not store:
            continue
        transcript = extract_transcript(store)
        eval_raw = find_evaluator_output(store)
        evaluation = parse_evaluation(eval_raw)
        runs[boss_type_id] = SingleRunResult(
            boss_type_id=boss_type_id,
            transcript=transcript,
            evaluation_raw=eval_raw,
            evaluation=evaluation,
            messages=extract_messages_as_dicts(store),
            elapsed_seconds=0.0,
        )

    comp_store = stores.get("__comparison__")
    comparison_summary = ""
    if comp_store:
        comp_msgs = comp_store.get_all()
        comparison_summary = "\n\n".join(
            m.content for m in comp_msgs if m.msg_type == "completed"
        )

    okrs = DEFAULT_OKRS.get(st.session_state.exp_project_type)
    if okrs is None:
        return

    team = create_default_team("time_master")

    result = ThesisExperimentResult(
        config=ExperimentConfig(
            topic=st.session_state.exp_topic,
            okrs=okrs,
            team=team,
            max_phases=st.session_state.exp_max_phases,
        ),
        runs=runs,
        comparison_summary=comparison_summary,
    )
    st.session_state.exp_result = result


def _build_markdown_table(result: ThesisExperimentResult) -> str:
    """Build a Markdown table of dimension scores for all available boss types."""
    available = [
        (bt, result.runs[bt])
        for bt in ["time_master", "time_neutral", "time_chaos"]
        if bt in result.runs
    ]
    if len(available) < 2:
        return "数据不完整（至少需要2组）"

    header_names = " | ".join(bt for bt, _ in available)
    header_sep = " | ".join("---" for _ in available)
    lines = [
        f"| 维度 | 权重 | {header_names} |",
        f"|------|------|{header_sep}|",
    ]

    for dim_id, dim in EVALUATION_DIMENSIONS.items():
        weight_pct = int(dim.weight * 100)
        scores = [
            run.evaluation.dimensions.get(dim_id, DimensionScore(score=0)).score
            for _, run in available
        ]
        score_cells = " | ".join(str(s) for s in scores)
        lines.append(f"| {dim.name_zh} | {weight_pct}% | {score_cells} |")

    totals = [run.evaluation.overall_score for _, run in available]
    total_cells = " | ".join(f"**{t:.1f}**" for t in totals)
    lines.append(f"| **加权总分** | 100% | {total_cells} |")
    return "\n".join(lines)


_LATEX_SPECIAL = re.compile(r"([\\&%$#_{}])")
_LATEX_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters in text (single-pass regex)."""
    return _LATEX_SPECIAL.sub(lambda m: _LATEX_MAP[m.group(1)], text)


def _build_latex_table(result: ThesisExperimentResult) -> str:
    """Build a LaTeX table of dimension scores for all available boss types."""
    available = [
        (bt, result.runs[bt])
        for bt in ["time_master", "time_neutral", "time_chaos"]
        if bt in result.runs
    ]
    col_count = len(available) + 2  # dim + weight + N boss types
    col_spec = "lc" + "c" * len(available)
    header_names = " & ".join(
        _latex_escape(bt) for bt, _ in available
    )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{团队绩效8维度评分对比}",
        r"\label{tab:performance}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"维度 & 权重 & {header_names} \\\\",
        r"\midrule",
    ]

    if not available:
        lines.append(r"数据不完整 \\")
    else:
        for dim_id, dim in EVALUATION_DIMENSIONS.items():
            weight_pct = int(dim.weight * 100)
            scores = [
                run.evaluation.dimensions.get(
                    dim_id, DimensionScore(score=0)
                ).score
                for _, run in available
            ]
            score_cells = " & ".join(str(s) for s in scores)
            lines.append(
                f"{_latex_escape(dim.name_zh)} & {weight_pct}\\% "
                f"& {score_cells} \\\\"
            )
        lines.append(r"\midrule")
        totals = [run.evaluation.overall_score for _, run in available]
        total_cells = " & ".join(f"{t:.1f}" for t in totals)
        lines.append(f"加权总分 & 100\\% & {total_cells} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _build_full_report(result: ThesisExperimentResult) -> str:
    """Build a complete experiment report in Markdown."""
    max_phases = result.config.max_phases or 0
    sections: list[str] = [
        "# 论文实验报告：时间驾驭能力与团队绩效",
        "",
        f"**实验时间**: {result.timestamp}",
        f"**项目议题**: {result.config.topic}",
        f"**项目类型**: {result.config.okrs.project_type_id}",
        f"**最大阶段数**: {max_phases}",
        f"**团队规模**: {len(result.config.team.members)} 人",
        "",
        "---",
        "",
        "## 研究设计",
        "",
        "- **自变量**: 老板时间管理类型 (time_master / time_neutral / time_chaos)",
        "- **因变量**: 团队绩效 (8维度评分)",
        "- **调节变量**: 项目类型",
        "- **模拟模式**: 项目生命周期推进（阶段制）",
        "",
        "## OKR 目标",
        "",
        format_okrs_for_prompt(result.config.okrs),
        "",
        "---",
        "",
        "## 绩效评分对比",
        "",
        _build_markdown_table(result),
        "",
    ]

    for boss_type_id in ["time_master", "time_neutral", "time_chaos"]:
        run = result.runs.get(boss_type_id)
        if not run:
            continue
        boss_info = BOSS_TYPES.get(boss_type_id)
        label = boss_info.name_zh if boss_info else boss_type_id
        sections.append(f"## {label} 关键发现")
        sections.append("")
        sections.extend(f"- {f}" for f in run.evaluation.key_findings)
        if run.evaluation.boss_impact_analysis:
            sections.append("")
            sections.append(
                f"**领导风格影响分析**: {run.evaluation.boss_impact_analysis}"
            )
        sections.append("")

    if result.comparison_summary:
        sections.extend([
            "---",
            "",
            "## 跨条件对比分析",
            "",
            result.comparison_summary,
        ])

    return "\n".join(sections)


def _run_single_experiment(
    boss_type_id: str,
    topic: str,
    okrs: OKRSet,
    max_phases: int,
    store: ChatMessageStore,
    config: dict,
):
    """Run a single experiment for one boss type (called in background thread)."""
    try:
        llm = create_primary_llm()
    except Exception as e:
        store.mark_error(f"LLM初始化失败: {e}")
        return

    try:
        team = create_default_team(boss_type_id)
        RUNTIME_STATE.active_store = store
        RUNTIME_STATE.set_current_prefix(boss_type_id)

        crew = build_thesis_crew(
            team=team,
            boss_type_id=boss_type_id,
            topic=topic,
            okrs=okrs,
            max_phases=max_phases,
            llm=llm,
            config=config,
        )

        start_time = time.time()
        stream_enabled = config.get("stream", False)
        if stream_enabled:
            streaming = crew.kickoff()
            if isinstance(streaming, CrewStreamingOutput):
                buffers: dict[str, str] = {}
                for chunk in streaming:
                    if store.cancelled:
                        break
                    if chunk.chunk_type != StreamChunkType.TEXT:
                        continue
                    if not chunk.content:
                        continue
                    msg_key = (
                        f"{boss_type_id}:{chunk.task_id}"
                        if chunk.task_id
                        else f"{boss_type_id}:{chunk.agent_id}:{chunk.task_index}"
                    )
                    prev = buffers.get(msg_key, "")
                    new_content = prev + chunk.content
                    buffers[msg_key] = new_content
                    store.upsert(
                        key=msg_key,
                        role=chunk.agent_role or "assistant",
                        content=new_content,
                        msg_type="stream",
                    )
                    RUNTIME_STATE.set_progress(
                        last_update=str(time.time()),
                    )
                try:
                    _ = streaming.result
                except Exception:
                    logger.debug("Streaming result access failed (cancelled?)")
            else:
                logger.warning(
                    "stream enabled but kickoff() did not return CrewStreamingOutput"
                )
        else:
            crew.kickoff()
        elapsed = time.time() - start_time

        store.finalize_streaming()
        logger.info(
            "Experiment completed for %s in %.1fs", boss_type_id, elapsed
        )
        store.mark_done()
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error("Experiment failed for %s: %s", boss_type_id, error_msg)
        store.mark_error(error_msg)
    finally:
        RUNTIME_STATE.active_store = None
        RUNTIME_STATE.set_current_prefix("")


def _run_thesis_experiment_thread(
    topic: str,
    project_type: str,
    max_phases: int,
    stores: dict[str, ChatMessageStore],
    config: dict,
):
    """Background thread: run all boss types sequentially, then compare."""
    okrs = DEFAULT_OKRS[project_type]
    boss_types = [bt for bt in BOSS_TYPES if bt in stores]

    RUNTIME_STATE.set_llm(
        status="idle",
        call_count=0,
        completed_count=0,
        failed_count=0,
        agent_role="",
        model="",
        call_started_at="",
    )

    for idx, boss_type_id in enumerate(boss_types):
        store = stores[boss_type_id]
        if store.cancelled:
            break

        RUNTIME_STATE.set_progress(
            step=str(idx + 1),
            total=str(len(boss_types) + 1),
            label=f"运行: {BOSS_TYPES[boss_type_id].name_zh}",
            live=f"构建 {boss_type_id} 实验...",
            last_update=str(time.time()),
        )

        _run_single_experiment(
            boss_type_id, topic, okrs, max_phases, store, config
        )

    # Run comparison
    comp_store = stores.get("__comparison__")
    if comp_store and not comp_store.cancelled:
        RUNTIME_STATE.set_progress(
            step=str(len(boss_types) + 1),
            total=str(len(boss_types) + 1),
            label="跨条件对比分析",
            live="生成对比分析...",
            last_update=str(time.time()),
        )

        eval_master = ""
        eval_chaos = ""
        eval_neutral = ""
        for bt, store_ref in stores.items():
            if bt == "__comparison__":
                continue
            ev = find_evaluator_output(store_ref) if store_ref else ""
            if bt == "time_master":
                eval_master = ev
            elif bt == "time_chaos":
                eval_chaos = ev
            elif bt == "time_neutral":
                eval_neutral = ev

        try:
            RUNTIME_STATE.active_store = comp_store
            RUNTIME_STATE.set_current_prefix("__comparison__")
            llm = create_primary_llm()
            prompt = build_comparison_summary_prompt(
                topic, eval_master, eval_chaos, eval_neutral
            )

            agent = Agent(
                role="跨条件对比分析师",
                goal="对比分析不同老板类型对团队绩效的差异化影响",
                backstory="你是资深组织行为学研究者，专注时间领导力理论。",
                verbose=False,
                allow_delegation=False,
                llm=llm,
                max_iter=5,
                max_execution_time=120,
            )
            task = Task(
                description=prompt,
                expected_output="结构化的跨条件对比分析报告",
                agent=agent,
            )
            comparison_crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
                max_rpm=10,
                stream=False,
            )
            comparison_crew.kickoff()
            comp_store.finalize_streaming()
            comp_store.mark_done()
        except Exception as e:
            comp_store.mark_error(f"{type(e).__name__}: {e}")
        finally:
            RUNTIME_STATE.active_store = None
            RUNTIME_STATE.set_current_prefix("")

    RUNTIME_STATE.set_progress(done="true", last_update=str(time.time()))


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "exp_topic": "Q3产品发布计划讨论",
    "exp_project_type": "urgent_launch",
    "exp_max_phases": 4,
    "exp_running": False,
    "exp_stores": {},
    "exp_result": None,
    "exp_worker": None,
}
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_config, tab_run, tab_eval, tab_export = st.tabs(
    ["⚙️ 实验配置", "▶️ 实验运行", "📊 绩效评估", "📥 论文素材导出"]
)


# ===== TAB 1: 实验配置 =====
with tab_config:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📋 实验参数")
        topic = st.text_input(
            "项目议题",
            value=st.session_state.exp_topic,
            help="团队需要推进的项目主题",
        )
        st.session_state.exp_topic = topic

        project_type = st.selectbox(
            "项目类型（调节变量）",
            options=list(DEFAULT_OKRS.keys()),
            format_func=lambda x: {
                "urgent_launch": "🚀 紧急上线",
                "long_term_platform": "🏛️ 长期平台建设",
                "exploratory_prototype": "🧪 探索性原型",
            }.get(x, x),
            index=list(DEFAULT_OKRS.keys()).index(
                st.session_state.exp_project_type
            ),
        )
        st.session_state.exp_project_type = project_type

        max_phases = st.slider(
            "最大项目阶段数",
            min_value=2,
            max_value=8,
            value=st.session_state.exp_max_phases,
            help="项目最多经历的阶段数（可提前自然结束）",
        )
        st.session_state.exp_max_phases = max_phases

        # Phase names preview
        phase_names = get_phase_names_zh(project_type)
        active_phases = phase_names[:max_phases]
        st.markdown("**项目阶段流程:**")
        st.markdown(" → ".join(active_phases))

        # OKR preview
        st.subheader("📌 OKR 目标")
        okr = DEFAULT_OKRS.get(project_type)
        if okr:
            st.markdown(f"**目标**: {okr.objective}")
            for i, kr in enumerate(okr.key_results, 1):
                weight_pct = int(kr.weight * 100)
                st.markdown(
                    f"- **KR{i}**: {kr.description} → {kr.target} "
                    f"(权重 {weight_pct}%)"
                )

        # Research design summary
        st.subheader("🔬 研究设计")
        st.markdown("""
| 变量 | 说明 |
|------|------|
| **自变量(IV)** | 老板时间管理类型 (time_master / time_neutral / time_chaos) |
| **因变量(DV)** | 团队绩效（8维度评分） |
| **调节变量** | 项目类型 |
| **模拟模式** | 项目生命周期推进（阶段制，可自由结束） |
| **实验设计** | 3 (boss) × 1 (project) = 3 组对比 |
""")

    with col_right:
        st.subheader("👥 默认团队 (12人)")
        for i in range(0, 12, 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(DEFAULT_TEAM_MEMBERS):
                    member = DEFAULT_TEAM_MEMBERS[idx]
                    ptype = PERSONALITY_TYPES.get(member.personality_type_id)
                    if ptype:
                        with col:
                            st.markdown(
                                f"**{ptype.icon} {member.name}**\n\n"
                                f"_{ptype.name_zh}_"
                            )
                            dims = ptype.dimensions
                            st.caption(
                                f"紧迫感: {dims.urgency} · "
                                f"行动: {dims.action_pattern} · "
                                f"时间: {dims.time_orientation}"
                            )

        st.divider()
        st.subheader("👔 两种老板类型")
        for boss_id, boss in BOSS_TYPES.items():
            with st.expander(
                f"{boss.name_zh} ({boss_id})", expanded=False
            ):
                st.markdown(boss.description)
                st.markdown("**特征**: " + "、".join(boss.traits))

        st.subheader("📏 评估维度 (8维)")
        for dim in EVALUATION_DIMENSIONS.values():
            weight_pct = int(dim.weight * 100)
            st.caption(
                f"• {dim.name_zh} ({weight_pct}%): {dim.description}"
            )


# ===== TAB 2: 实验运行 =====
with tab_run:
    col_start, col_cancel = st.columns([3, 1])
    with col_start:
        start_btn = st.button(
            "🚀 开始实验",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.exp_running,
        )
    with col_cancel:
        cancel_btn = st.button(
            "🛑 取消",
            type="secondary",
            use_container_width=True,
            disabled=not st.session_state.exp_running,
        )

    if cancel_btn and st.session_state.exp_running:
        for s in st.session_state.exp_stores.values():
            s.mark_cancelled()
        st.session_state.exp_running = False
        st.warning("⚠️ 实验已取消")
        st.rerun()

    if start_btn and not st.session_state.exp_running:
        stores: dict[str, ChatMessageStore] = {
            bt: ChatMessageStore() for bt in BOSS_TYPES
        }
        stores["__comparison__"] = ChatMessageStore()
        st.session_state.exp_stores = stores
        st.session_state.exp_running = True
        st.session_state.exp_result = None
        st.session_state.exp_started_at = time.time()

        config = {
            "agent_timeout": 120,
            "max_iterations": 5,
            "context_window": 30,
            "stream": True,
            "seed": 42,
        }

        RUNTIME_STATE.set_progress(
            step="0",
            total="3",
            label="初始化实验",
            live="准备中...",
            last_update=str(time.time()),
        )

        thread = threading.Thread(
            target=_run_thesis_experiment_thread,
            args=(
                st.session_state.exp_topic,
                st.session_state.exp_project_type,
                st.session_state.exp_max_phases,
                stores,
                config,
            ),
            daemon=True,
        )
        thread.start()
        st.session_state.exp_worker = thread
        st.rerun()

    # Progress display
    stores = st.session_state.exp_stores
    if st.session_state.exp_running and stores:
        comp_store = stores.get("__comparison__")

        all_done = all(
            s.done for key, s in stores.items() if key != "__comparison__"
        )
        comp_done = comp_store.done if comp_store else True

        if not (all_done and comp_done):
            progress_info = RUNTIME_STATE.snapshot_progress()
            llm_info = RUNTIME_STATE.snapshot_llm()

            label = progress_info.get("label", "准备中...")
            step = int(progress_info.get("step", "0") or "0")
            total = max(int(progress_info.get("total", "3") or "3"), 1)
            live = progress_info.get("live", "等待启动")
            started_at = st.session_state.get(
                "exp_started_at", time.time()
            )
            elapsed = int(time.time() - started_at)
            completed_count = int(llm_info.get("completed_count", 0))
            call_count = int(llm_info.get("call_count", 0))

            worker = st.session_state.get("exp_worker")
            if worker is not None and not worker.is_alive():
                for s in stores.values():
                    if not s.done and not s.error:
                        s.mark_error("后台线程已退出")
                st.session_state.exp_running = False
                st.error("❌ 后台执行线程异常退出，请重试")
                st.rerun()

            st.info(f"🔄 实验进行中 — {label}")
            st.progress(
                min(max(step / total, 0.0), 1.0),
                text=f"进度: {step}/{total}",
            )
            st.caption(
                f"已运行: {elapsed}s | "
                f"LLM调用: {completed_count}/{call_count} | {live}"
            )

            boss_cols = st.columns(len(BOSS_TYPES))
            for col, bt in zip(boss_cols, BOSS_TYPES):
                with col:
                    icon = _BOSS_ICONS.get(bt, "👔")
                    st.markdown(f"### {icon} {bt}")
                    bt_store = stores.get(bt)
                    if bt_store:
                        _render_full_messages(bt_store)

            time.sleep(1.5)
            st.rerun()
        else:
            st.session_state.exp_running = False
            _build_experiment_result()
            st.success(
                "✅ 实验完成！请切换到「📊 绩效评估」查看结果。"
            )

    if not st.session_state.exp_running and stores:
        any_error = any(s.error for s in stores.values())
        if any_error:
            for key, s in stores.items():
                if s.error:
                    st.error(f"{key}: {s.error}")

        boss_cols_final = st.columns(len(BOSS_TYPES))
        for col, bt in zip(boss_cols_final, BOSS_TYPES):
            with col:
                icon = _BOSS_ICONS.get(bt, "👔")
                st.markdown(f"### {icon} {bt}")
                bt_store = stores.get(bt)
                if bt_store:
                    _render_full_messages(bt_store)


# ===== TAB 3: 绩效评估 =====
with tab_eval:
    result: ThesisExperimentResult | None = st.session_state.exp_result

    if result is None:
        st.info(
            "👈 请先在「⚙️ 实验配置」中设置参数，"
            "然后在「▶️ 实验运行」中启动实验"
        )
    else:
        available_runs = [
            (bt, result.runs[bt])
            for bt in ["time_master", "time_neutral", "time_chaos"]
            if bt in result.runs
        ]

        if len(available_runs) < 2:
            st.warning("实验数据不完整，至少需要2组老板类型的结果")
        else:
            # Overall scores
            st.subheader("📊 加权总分对比")
            score_cols = st.columns(len(available_runs))
            for col, (bt, run) in zip(score_cols, available_runs):
                icon = _BOSS_ICONS.get(bt, "👔")
                with col:
                    st.metric(
                        f"{icon} {bt}",
                        f"{run.evaluation.overall_score:.1f}",
                    )

            st.divider()

            # Radar chart
            _RADAR_COLORS = {
                "time_master": ("#2196F3", "rgba(33, 150, 243, 0.15)"),
                "time_neutral": ("#4CAF50", "rgba(76, 175, 80, 0.15)"),
                "time_chaos": ("#FF5722", "rgba(255, 87, 34, 0.15)"),
            }
            _RADAR_LABELS = {
                "time_master": "time_master (高效管理)",
                "time_neutral": "time_neutral (中性基线)",
                "time_chaos": "time_chaos (混乱管理)",
            }

            st.subheader("🕸️ 8维度雷达图对比")
            dim_ids = list(EVALUATION_DIMENSIONS.keys())
            dim_names = [
                EVALUATION_DIMENSIONS[d].name_zh for d in dim_ids
            ]

            fig = go.Figure()
            for bt, run in available_runs:
                scores = [
                    run.evaluation.dimensions.get(
                        d, DimensionScore(score=0)
                    ).score
                    for d in dim_ids
                ]
                line_color, fill_color = _RADAR_COLORS.get(
                    bt, ("#9E9E9E", "rgba(158, 158, 158, 0.15)")
                )
                fig.add_trace(
                    go.Scatterpolar(
                        r=[*scores, scores[0]],
                        theta=[*dim_names, dim_names[0]],
                        fill="toself",
                        name=_RADAR_LABELS.get(bt, bt),
                        line={"color": line_color},
                        fillcolor=fill_color,
                    )
                )
            fig.update_layout(
                polar={
                    "radialaxis": {"visible": True, "range": [0, 100]}
                },
                showlegend=True,
                title="团队绩效 8 维度雷达图",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Dimension detail table
            st.subheader("📋 维度明细对比")
            table_data: list[dict] = []
            for dim_id in dim_ids:
                dim = EVALUATION_DIMENSIONS[dim_id]
                weight_pct = int(dim.weight * 100)
                row_data: dict = {"维度": f"{dim.name_zh} ({weight_pct}%)"}
                for bt, run in available_runs:
                    ds = run.evaluation.dimensions.get(
                        dim_id, DimensionScore(score=0, evidence="无数据")
                    )
                    row_data[bt] = ds.score
                    row_data[f"{bt}_证据"] = ds.evidence[:80]
                table_data.append(row_data)
            st.dataframe(table_data, use_container_width=True)

            # Key findings
            st.subheader("🔍 关键发现")
            finding_cols = st.columns(len(available_runs))
            for col, (bt, run) in zip(finding_cols, available_runs):
                with col:
                    st.markdown(f"**{bt} 关键发现:**")
                    for finding in run.evaluation.key_findings:
                        st.markdown(f"- {finding}")
                    if run.evaluation.boss_impact_analysis:
                        st.info(run.evaluation.boss_impact_analysis)

            # Comparison summary
            if result.comparison_summary:
                st.divider()
                st.subheader("📝 跨条件对比分析")
                st.markdown(result.comparison_summary)


# ===== TAB 4: 论文素材导出 =====
with tab_export:
    result_export: ThesisExperimentResult | None = (
        st.session_state.exp_result
    )

    if result_export is None:
        st.info("请先运行实验以生成可导出的数据")
    else:
        st.subheader("📊 数据导出")

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("#### Markdown 数据表")
            md_table = _build_markdown_table(result_export)
            st.code(md_table, language="markdown")
            st.download_button(
                "📥 下载 Markdown 表格",
                data=md_table,
                file_name="thesis_results_table.md",
                mime="text/markdown",
                key="dl_md_table",
            )

        with col_e2:
            st.markdown("#### LaTeX 数据表")
            latex_table = _build_latex_table(result_export)
            st.code(latex_table, language="latex")
            st.download_button(
                "📥 下载 LaTeX 表格",
                data=latex_table,
                file_name="thesis_results_table.tex",
                mime="text/plain",
                key="dl_latex_table",
            )

        st.divider()

        st.subheader("📄 完整实验报告")
        report_md = _build_full_report(result_export)
        st.download_button(
            "📥 下载完整报告 (Markdown)",
            data=report_md,
            file_name="thesis_experiment_report.md",
            mime="text/markdown",
            key="dl_full_report",
        )

        st.subheader("🗂️ 原始数据")
        raw_json = result_export.model_dump_json(indent=2)
        st.download_button(
            "📥 下载原始数据 (JSON)",
            data=raw_json,
            file_name="thesis_experiment_raw.json",
            mime="application/json",
            key="dl_raw_json",
        )

        st.subheader("📈 图表导出")
        st.caption(
            "雷达图可在「📊 绩效评估」标签页中右键另存为图片"
        )

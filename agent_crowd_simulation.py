"""Agent 人群模拟研究平台 - Agent Crowd Simulation Platform

A Streamlit app that uses crewAI agents to simulate multi-agent crowd interactions.
Supports comparing the same topic under multiple leadership styles.
"""

import logging
import os
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from crewai import Crew
from crewai.types.streaming import CrewStreamingOutput, StreamChunkType

from lib_custom.chat_store import ChatMessage, ChatMessageStore
from lib_custom.crew_builder import CrewBuilder, build_comparison_crew
from lib_custom.leadership_styles import LEADERSHIP_STYLES, apply_style_to_roles
from lib_custom.llm_config import create_primary_llm
from lib_custom.role_repository import RoleRepository
from lib_custom.runtime_state import STATE as RUNTIME_STATE
from lib_custom.runtime_state import ensure_event_handlers_registered


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def validate_api_endpoint() -> tuple[bool, str]:
    """Validate that the API endpoint is reachable.

    Returns:
        (is_valid, error_message)
    """
    base_url = os.getenv("OPENAI_BASE_URL", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model_name = os.getenv("OPENAI_MODEL_NAME", "")

    if not api_key:
        return False, "❌ OPENAI_API_KEY 未设置，请检查 .env 文件"

    if not base_url:
        return False, "❌ OPENAI_BASE_URL 未设置，请检查 .env 文件"

    # Log model configuration for debugging
    if model_name:
        logger.info(f"Using model: {model_name}")

    # Health-check: validate connectivity and auth via /models endpoint
    is_local = "127.0.0.1" in base_url or "localhost" in base_url
    timeout = 5 if is_local else 10

    try:
        test_url = base_url.rstrip("/") + "/models"
        req = Request(
            test_url,
            headers={
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urlopen(req, timeout=timeout) as response:
            response.read()
        return True, ""
    except HTTPError as e:
        if e.code == 401:
            return (
                False,
                f"❌ API 鉴权失败 (401 Unauthorized): {base_url}\n"
                "请检查 OPENAI_API_KEY 是否正确。",
            )
        if e.code == 403:
            return (
                False,
                f"❌ API 无权限访问 (403 Forbidden): {base_url}\n"
                "请检查 API Key 权限配置。",
            )
        return (
            False,
            f"❌ API 返回 HTTP {e.code}: {base_url}\n"
            f"错误: {e.reason}",
        )
    except URLError as e:
        return False, f"❌ 无法连接到 API: {base_url}\n错误: {e.reason}"
    except TimeoutError:
        return False, f"❌ 连接超时: {base_url}\n请检查服务是否正常"
    except Exception as e:
        return False, f"❌ 连接错误: {str(e)}"


ensure_event_handlers_registered()


# ---------------------------------------------------------------------------
# Avatar mapping — built dynamically from role config
# ---------------------------------------------------------------------------

def get_role_avatars() -> dict[str, str]:
    """Build avatar mapping from current role configuration."""
    try:
        repo = RoleRepository()
        db = repo.load_roles()
        avatars = {role.role_name: role.avatar for role in db.roles}
        avatars["system"] = "⚙️"
        avatars["跨风格对比分析师"] = "📊"
        return avatars
    except Exception:
        return {"system": "⚙️"}


# ---------------------------------------------------------------------------
# Background thread runner
# ---------------------------------------------------------------------------

def _render_results_tabs(
    style_stores: dict[str, ChatMessageStore],
    style_ids: list[str],
):
    """Render results in tabs — one per style + optional comparison tab."""
    sim_topic = st.session_state.sim_topic
    sim_rounds = st.session_state.sim_num_rounds

    tab_names = [LEADERSHIP_STYLES[sid].style_name for sid in style_ids]
    has_comparison = "__comparison__" in style_stores
    if has_comparison:
        tab_names.append("📊 跨风格对比分析")

    tabs = st.tabs(tab_names)

    for i, sid in enumerate(style_ids):
        style = LEADERSHIP_STYLES[sid]
        store = style_stores[sid]
        messages = store.get_all()

        with tabs[i]:
            if store.cancelled:
                st.warning("⚠️ 此模拟已被取消")
            elif store.error:
                st.error(f"模拟出错: {store.error}")
            else:
                render_messages(messages)
                render_analyst(messages)

            md = format_messages_as_markdown(
                messages, sim_topic, sim_rounds, style_name=style.style_name,
            )
            st.download_button(
                f"📥 下载 {style.style_name} 记录",
                data=md,
                file_name=f"simulation_{sid}.md",
                mime="text/markdown",
                key=f"dl_{sid}",
            )

    if has_comparison:
        comp_store = style_stores["__comparison__"]
        comp_messages = comp_store.get_all()
        with tabs[-1]:
            if comp_store.error:
                st.error(f"对比分析出错: {comp_store.error}")
            else:
                for msg in comp_messages:
                    if msg.msg_type == "completed":
                        st.markdown(msg.content)

            comp_text = "\n\n".join(
                msg.content for msg in comp_messages if msg.msg_type == "completed"
            )
            st.download_button(
                "📥 下载跨风格对比分析",
                data=comp_text,
                file_name="comparison_analysis.md",
                mime="text/markdown",
                key="dl_comparison",
            )


# ---------------------------------------------------------------------------
# Render chat messages in Streamlit
# ---------------------------------------------------------------------------

def get_conversation_role_names() -> set[str]:
    """Get current conversation role names from repository."""
    try:
        repo = RoleRepository()
        db = repo.load_roles()
        return {role.role_name for role in db.get_conversation_roles()}
    except Exception:
        return {"老板 (Boss)", "资深员工 (Senior)", "新人 (Newbie)"}


def format_messages_as_markdown(
    messages: list[ChatMessage],
    topic: str,
    num_rounds: int,
    style_name: str = "",
) -> str:
    """Format finished messages into a Markdown document for export."""
    avatars = get_role_avatars()
    conversation_roles = get_conversation_role_names()

    header = f"# Agent 人群模拟记录 — {style_name}" if style_name else "# Agent 人群模拟记录"
    lines = [header, f"\n## 话题：{topic}", f"\n对话轮数：{num_rounds}", ""]

    num_conv_roles = len(conversation_roles) or 3
    current_round = 0
    msgs_in_round = 0
    analyst_content = ""

    for msg in messages:
        if msg.msg_type != "completed":
            continue
        if msg.role not in conversation_roles:
            if "分析" in msg.role and msg.msg_type == "completed":
                analyst_content = msg.content
            continue
        if msgs_in_round % num_conv_roles == 0:
            current_round += 1
            lines.append(f"\n### 第 {current_round} 轮\n")
        msgs_in_round += 1
        avatar = avatars.get(msg.role, "")
        lines.append(f"**{avatar} {msg.role}:**\n")
        lines.append(f"{msg.content}\n")

    if analyst_content:
        lines.append("\n---\n")
        lines.append("### 分析总结\n")
        lines.append(analyst_content)

    return "\n".join(lines)


def render_messages(messages: list[ChatMessage]):
    """Render conversation messages (excludes analyst output)."""
    avatars = get_role_avatars()
    for msg in messages:
        if msg.msg_type == "started":
            continue
        if "分析" in msg.role:
            continue
        avatar = avatars.get(msg.role, "💬")
        if msg.role == "system":
            st.caption(msg.content)
        else:
            with st.chat_message(msg.role, avatar=avatar):
                st.markdown(f"**{msg.role}**")
                st.write(msg.content)


def render_analyst(messages: list[ChatMessage]):
    """Render the analyst output in a separate expander."""
    for msg in messages:
        if "分析" in msg.role and msg.msg_type == "completed":
            with st.expander("📊 群体互动分析", expanded=False):
                st.markdown(msg.content)
            return


def extract_conversation_text(messages: list[ChatMessage]) -> str:
    """Extract completed conversation messages as plain text."""
    parts = []
    for msg in messages:
        if msg.msg_type == "completed" and msg.role != "system":
            parts.append(f"[{msg.role}]: {msg.content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Multi-style simulation runner (synchronous, called from background thread)
# ---------------------------------------------------------------------------

def run_multi_style_simulation(
    topic: str,
    num_rounds: int,
    selected_style_ids: list[str],
    style_stores: dict[str, ChatMessageStore],
    progress_callback,
    config: dict,
):
    """Run simulation for each selected style sequentially.

    Populates style_stores with messages for each style, then runs comparison.
    """
    repo = RoleRepository()
    base_db = repo.load_roles()
    total_steps = len(selected_style_ids) + (1 if len(selected_style_ids) > 1 else 0)
    style_conversations: dict[str, str] = {}

    # Bind to the same OpenAI-compatible endpoint/model shown in the sidebar.
    # Without this, CrewAI may fall back to a default provider/config and hang
    # before emitting any task events (UI then shows "启动超时").
    try:
        llm = create_primary_llm()
    except Exception as e:
        raise RuntimeError(f"LLM 初始化失败: {type(e).__name__}: {e}") from e

    stream_enabled = bool(config.get("stream", True))

    for idx, style_id in enumerate(selected_style_ids):
        style = LEADERSHIP_STYLES[style_id]
        store = style_stores[style_id]

        # Check if cancelled before starting
        if store.cancelled:
            logger.info(f"Simulation cancelled before starting {style.style_name}")
            break

        try:
            styled_db = apply_style_to_roles(base_db, style)
            builder = CrewBuilder(styled_db, config, llm=llm)
            RUNTIME_STATE.set_progress(
                live=f"构建任务与角色: {style.style_name}",
                last_update=str(time.time()),
            )
            crew = builder.build_crew(topic, num_rounds)
            RUNTIME_STATE.active_store = store
            RUNTIME_STATE.set_current_prefix(style_id)
            progress_callback(idx, total_steps, style.style_name)
            logger.info(f"Starting simulation for style: {style.style_name}")
            logger.info(f"Topic: {topic}, Rounds: {num_rounds}")

            # Run crew with timeout monitoring
            start_time = time.time()
            if stream_enabled:
                streaming = crew.kickoff()
                if isinstance(streaming, CrewStreamingOutput):
                    buffers: dict[str, str] = {}
                    for chunk in streaming:
                        if store.cancelled:
                            break
                        if chunk.chunk_type != StreamChunkType.TEXT:
                            # Tool call chunks can be shown in terminal logs; keep UI clean for now.
                            continue
                        if not chunk.content:
                            continue
                        msg_key = f"{style_id}:{chunk.task_id}" if chunk.task_id else f"{style_id}:{chunk.agent_id}:{chunk.task_index}"
                        prev = buffers.get(msg_key, "")
                        new = prev + chunk.content
                        buffers[msg_key] = new
                        store.upsert(key=msg_key, role=chunk.agent_role or "assistant", content=new, msg_type="stream")
                        RUNTIME_STATE.set_progress(last_update=str(time.time()))
                    # Ensure we can access the final result (may raise if cancelled mid-stream).
                    try:
                        _ = streaming.result
                    except Exception:
                        pass
                else:
                    # Fallback: non-streaming output
                    crew.kickoff()
            else:
                crew.kickoff()
            elapsed = time.time() - start_time
            store.finalize_streaming()

            logger.info(f"Simulation completed for {style.style_name} in {elapsed:.1f}s")
            store.mark_done()
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Simulation failed for {style.style_name}: {error_msg}")

            store.mark_error(error_msg)
        finally:
            RUNTIME_STATE.active_store = None
            RUNTIME_STATE.set_current_prefix("")

        style_conversations[style.style_name] = extract_conversation_text(
            store.get_all()
        )

    # Run cross-style comparison if multiple styles selected
    if len(selected_style_ids) > 1:
        comparison_store = style_stores.get("__comparison__")
        if comparison_store is not None:
            try:
                RUNTIME_STATE.active_store = comparison_store
                RUNTIME_STATE.set_current_prefix("__comparison__")
                progress_callback(
                    len(selected_style_ids), total_steps, "跨风格对比分析"
                )
                logger.info("Starting cross-style comparison analysis")
                RUNTIME_STATE.set_progress(
                    live="构建跨风格对比分析任务",
                    last_update=str(time.time()),
                )
                comparison_crew = build_comparison_crew(topic, style_conversations, llm=llm, stream=stream_enabled)

                start_time = time.time()
                if stream_enabled:
                    streaming = comparison_crew.kickoff()
                    if isinstance(streaming, CrewStreamingOutput):
                        buffers: dict[str, str] = {}
                        for chunk in streaming:
                            if comparison_store.cancelled:
                                break
                            if chunk.chunk_type != StreamChunkType.TEXT:
                                continue
                            if not chunk.content:
                                continue
                            msg_key = (
                                f"__comparison__:{chunk.task_id}"
                                if chunk.task_id
                                else f"__comparison__:{chunk.agent_id}:{chunk.task_index}"
                            )
                            prev = buffers.get(msg_key, "")
                            new = prev + chunk.content
                            buffers[msg_key] = new
                            comparison_store.upsert(
                                key=msg_key,
                                role=chunk.agent_role or "跨风格对比分析师",
                                content=new,
                                msg_type="stream",
                            )
                            RUNTIME_STATE.set_progress(last_update=str(time.time()))
                        try:
                            _ = streaming.result
                        except Exception:
                            pass
                    else:
                        comparison_crew.kickoff()
                else:
                    comparison_crew.kickoff()
                elapsed = time.time() - start_time
                comparison_store.finalize_streaming()

                logger.info(f"Comparison analysis completed in {elapsed:.1f}s")
                comparison_store.mark_done()
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Comparison analysis failed: {error_msg}")

                comparison_store.mark_error(error_msg)
            finally:
                RUNTIME_STATE.active_store = None
                RUNTIME_STATE.set_current_prefix("")


# ---------------------------------------------------------------------------
# Background thread wrapper for multi-style simulation
# ---------------------------------------------------------------------------

def _run_in_thread(topic, num_rounds, selected_style_ids, style_stores, config):
    """Background thread entry point for multi-style simulation."""

    RUNTIME_STATE.set_llm(
        status="idle",
        call_count=0,
        completed_count=0,
        failed_count=0,
        agent_role="",
        model="",
        call_started_at="",
    )

    def progress_callback(step, total, label):
        RUNTIME_STATE.set_progress(
            step=str(step + 1),
            total=str(total),
            label=label,
            last_update=str(time.time()),
        )

    try:
        run_multi_style_simulation(
            topic, num_rounds, selected_style_ids, style_stores, progress_callback, config
        )
    except Exception as e:
        logger.exception("Simulation thread crashed before completion")
        error_msg = f"运行线程异常: {type(e).__name__}: {str(e)}"
        RUNTIME_STATE.set_progress(live=error_msg, last_update=str(time.time()))
        for store in style_stores.values():
            if not store.done:
                store.mark_error(error_msg)
    finally:
        RUNTIME_STATE.set_progress(done="true", last_update=str(time.time()))


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Agent 人群模拟研究",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Agent 人群模拟研究")
    st.caption("基于 LLM 多智能体的群体行为模拟 — 领导风格对比分析")

    # -- Sidebar --
    with st.sidebar:
        st.header("⚙️ 设置")
        topic = st.text_input(
            "会议话题",
            value="项目延期了，谁来背锅？",
            help="输入一个模拟场景话题",
        )
        num_rounds = st.slider(
            "对话轮数", min_value=1, max_value=10, value=3,
            help="每轮角色各发言一次，轮数越多剧情越丰富",
        )

        with st.expander("🔧 高级设置", expanded=False):
            stream_output = st.checkbox(
                "流式输出（推荐）",
                value=True,
                help="开启后，LLM 的输出会在前端逐字显示，等待体验更好；关闭后只在完成时一次性显示。",
            )
            agent_timeout = st.slider(
                "Agent超时时间（秒）",
                min_value=30,
                max_value=300,
                value=120,
                step=30,
                help="单个Agent的最大执行时间，超时后会终止",
            )
            max_iterations = st.slider(
                "最大迭代次数",
                min_value=1,
                max_value=10,
                value=5,
                help="防止Agent陷入无限循环",
            )
            context_window = st.slider(
                "上下文窗口（任务数）",
                min_value=2,
                max_value=12,
                value=4,
                step=2,
                help="每个Agent能看到的历史任务数量，越大消耗越多token",
            )

        st.divider()

        # Model status display
        st.subheader("模型配置")
        primary_model = os.getenv("OPENAI_MODEL_NAME", "")
        primary_base_url = os.getenv("OPENAI_BASE_URL", "")
        primary_api_key = os.getenv("OPENAI_API_KEY", "")

        st.caption(f"接口: `{primary_base_url or '未配置'}`")
        st.caption(f"模型: **{primary_model or '未配置'}**")
        if not primary_api_key or not primary_model:
            st.warning("⚠️ 无可用模型，请检查 .env 配置")

        st.divider()
        st.subheader("领导风格选择")
        style_options = {
            sid: s.style_name for sid, s in LEADERSHIP_STYLES.items()
        }
        selected_styles = st.multiselect(
            "选择要对比的领导风格（至少1个）",
            options=list(style_options.keys()),
            default=list(style_options.keys()),
            format_func=lambda sid: f"{style_options[sid]}",
        )

        # Show style descriptions
        for sid in selected_styles:
            style = LEADERSHIP_STYLES[sid]
            st.caption(f"**{style.style_name}**: {style.description}")

        start_btn = st.button(
            "🚀 开始模拟",
            type="primary",
            use_container_width=True,
            disabled=len(selected_styles) == 0,
        )

        st.divider()
        st.subheader("角色介绍")
        avatars = get_role_avatars()
        for role_name, avatar in avatars.items():
            if role_name not in ("system", "跨风格对比分析师"):
                st.write(f"{avatar} **{role_name}**")

    # -- Initialize session state --
    if "style_stores" not in st.session_state:
        st.session_state.style_stores = {}
    if "running" not in st.session_state:
        st.session_state.running = False
    if "sim_topic" not in st.session_state:
        st.session_state.sim_topic = ""
    if "sim_num_rounds" not in st.session_state:
        st.session_state.sim_num_rounds = 3
    if "selected_style_ids" not in st.session_state:
        st.session_state.selected_style_ids = []
    if "agent_timeout" not in st.session_state:
        st.session_state.agent_timeout = 120
    if "max_iterations" not in st.session_state:
        st.session_state.max_iterations = 5
    if "context_window" not in st.session_state:
        st.session_state.context_window = 4
    if "stream_output" not in st.session_state:
        st.session_state.stream_output = True
    if "worker_thread" not in st.session_state:
        st.session_state.worker_thread = None

    # -- Handle start button --
    if start_btn and not st.session_state.running and selected_styles:
        # Validate API endpoint before starting
        is_valid, error_msg = validate_api_endpoint()
        if not is_valid:
            st.error(error_msg)
            st.info("💡 提示：如果使用本地代理，请确保代理服务正在运行")
            st.stop()

        stores: dict[str, ChatMessageStore] = {}
        for sid in selected_styles:
            stores[sid] = ChatMessageStore()
        if len(selected_styles) > 1:
            stores["__comparison__"] = ChatMessageStore()

        st.session_state.style_stores = stores
        st.session_state.running = True
        st.session_state.sim_topic = topic
        st.session_state.sim_num_rounds = num_rounds
        st.session_state.selected_style_ids = list(selected_styles)
        st.session_state.agent_timeout = agent_timeout
        st.session_state.max_iterations = max_iterations
        st.session_state.context_window = context_window
        st.session_state.stream_output = stream_output
        st.session_state.sim_started_at = time.time()

        total_steps = len(selected_styles) + (1 if len(selected_styles) > 1 else 0)
        RUNTIME_STATE.set_progress(
            step="0",
            total=str(max(total_steps, 1)),
            label="初始化任务",
            live="等待第一个角色开始",
            last_update=str(time.time()),
        )
        RUNTIME_STATE.set_llm(
            status="idle",
            call_count=0,
            completed_count=0,
            failed_count=0,
            agent_role="",
            model="",
            call_started_at="",
        )

        config = {
            "agent_timeout": agent_timeout,
            "max_iterations": max_iterations,
            "context_window": context_window,
            "stream": stream_output,
        }

        thread = threading.Thread(
            target=_run_in_thread,
            args=(topic, num_rounds, list(selected_styles), stores, config),
            daemon=True,
        )
        thread.start()
        st.session_state.worker_thread = thread

    # -- Main area: render based on state --
    style_stores = st.session_state.style_stores
    style_ids = st.session_state.selected_style_ids

    if st.session_state.running and style_stores:
        # Check if all stores are done
        all_done = all(
            store.done for sid, store in style_stores.items()
            if sid != "__comparison__"
        )
        comparison_store = style_stores.get("__comparison__")
        comparison_done = comparison_store.done if comparison_store else True

        if not (all_done and comparison_done):
            _progress_info = RUNTIME_STATE.snapshot_progress()
            _llm_call_info = RUNTIME_STATE.snapshot_llm()
            label = _progress_info.get("label", "准备中...")
            step = int(_progress_info.get("step", "0") or "0")
            total = max(int(_progress_info.get("total", "1") or "1"), 1)
            live = _progress_info.get("live", "等待任务启动")
            last_update = float(_progress_info.get("last_update", "0") or "0")
            started_at = st.session_state.get("sim_started_at", time.time())
            elapsed = int(time.time() - started_at)
            idle_seconds = int(time.time() - last_update) if last_update > 0 else elapsed
            progress_ratio = min(max(step / total, 0.0), 1.0)
            worker = st.session_state.get("worker_thread")

            # Guardrail: if worker thread has exited but stores are not done, fail fast with diagnostics.
            if worker is not None and not worker.is_alive():
                fail_msg = "后台执行线程已退出，任务未完成。请重试；若重复出现，请检查终端日志。"
                for store in style_stores.values():
                    if not store.done:
                        store.mark_error(fail_msg)
                st.session_state.running = False
                st.error(f"❌ {fail_msg}")
                st.rerun()

            # Guardrail: no update for too long while still at initial phase usually means startup failure.
            if step == 0 and idle_seconds > 45:
                timeout_msg = f"启动超时：{idle_seconds}s 内未收到任务事件。请检查模型接口与日志。"
                for store in style_stores.values():
                    if not store.done:
                        store.mark_error(timeout_msg)
                st.session_state.running = False
                st.error(f"❌ {timeout_msg}")
                st.rerun()

            col1, col2 = st.columns([3, 1])
            with col1:
                llm_status = _llm_call_info.get("status", "idle")
                call_count = int(_llm_call_info.get("call_count", 0))
                completed_count = int(_llm_call_info.get("completed_count", 0))
                failed_count = int(_llm_call_info.get("failed_count", 0))

                if llm_status == "sending":
                    call_started = float(_llm_call_info.get("call_started_at", 0) or 0)
                    waiting_sec = int(time.time() - call_started) if call_started > 0 else 0
                    st.warning(f"📡 模拟进行中 — {label}　|　正在等待 LLM 响应... ({waiting_sec}s)")
                elif llm_status == "failed":
                    st.error(f"⚠️ 模拟进行中 — {label}　|　上次请求失败，重试中...")
                else:
                    st.info(f"🔄 模拟进行中 — {label}")

                st.progress(progress_ratio, text=f"阶段进度: {step}/{total}")
                st.caption(
                    f"已运行: {elapsed}s | LLM 调用: {completed_count}/{call_count} 完成"
                    + (f" · {failed_count} 失败" if failed_count else "")
                    + f" | {live}"
                )
            with col2:
                if st.button("🛑 取消模拟", type="secondary", use_container_width=True):
                    # Mark all stores as cancelled
                    for store in style_stores.values():
                        store.mark_cancelled()
                    st.session_state.running = False
                    st.warning("⚠️ 模拟已取消")
                    st.rerun()

            # Stream partial outputs while running (tabs update on each rerun).
            _render_results_tabs(style_stores, style_ids)
            time.sleep(1.0)
            st.rerun()

        # All done — stop polling
        st.session_state.running = False

        # Check if any were cancelled
        any_cancelled = any(store.cancelled for store in style_stores.values())
        if any_cancelled:
            st.warning("⚠️ 模拟已被用户取消")
        else:
            st.success("✅ 所有风格模拟完成！")

    if style_stores and not st.session_state.running:
        _render_results_tabs(style_stores, style_ids)

    elif not style_stores:
        st.info("👈 在左侧选择领导风格并输入话题，点击「开始模拟」")


main()

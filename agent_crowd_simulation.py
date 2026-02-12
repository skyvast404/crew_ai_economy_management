"""Agent 人群模拟研究平台 - Agent Crowd Simulation Platform

A Streamlit app that uses crewAI agents to simulate multi-agent crowd interactions.
Supports comparing the same topic under multiple leadership styles.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from crewai import Crew
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.task_events import TaskCompletedEvent, TaskStartedEvent

from lib_custom.crew_builder import CrewBuilder, build_comparison_crew
from lib_custom.leadership_styles import LEADERSHIP_STYLES, apply_style_to_roles
from lib_custom.llm_config import create_openrouter_llm, get_available_llms
from lib_custom.role_repository import RoleRepository


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
        if "gemini" in model_name.lower() and "127.0.0.1" not in base_url:
            logger.warning(
                f"Model '{model_name}' appears to be Gemini but URL is not local proxy. "
                "Ensure your proxy correctly translates OpenAI API to Gemini."
            )

    # Check if it's a local proxy
    if "127.0.0.1" in base_url or "localhost" in base_url:
        try:
            # Use an authenticated OpenAI-compatible endpoint to validate both connectivity and auth.
            test_url = base_url.rstrip("/") + "/models"
            req = Request(
                test_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
            )
            with urlopen(req, timeout=5) as response:
                response.read()
            return True, ""
        except HTTPError as e:
            if e.code == 401:
                return (
                    False,
                    f"❌ 本地代理鉴权失败 (401 Unauthorized): {base_url}\n"
                    "请检查 OPENAI_API_KEY 是否正确，或代理是否需要不同的密钥格式。",
                )
            if e.code == 403:
                return (
                    False,
                    f"❌ 本地代理无权限访问 (403 Forbidden): {base_url}\n"
                    "请检查 API Key 权限或代理访问控制配置。",
                )
            return (
                False,
                f"❌ 本地代理返回 HTTP {e.code}: {base_url}\n"
                f"错误: {e.reason}",
            )
        except URLError as e:
            return False, f"❌ 无法连接到本地代理: {base_url}\n请确保代理服务正在运行\n错误: {e.reason}"
        except TimeoutError:
            return False, f"❌ 连接超时: {base_url}\n请检查代理服务是否正常"
        except Exception as e:
            return False, f"❌ 连接错误: {str(e)}"

    # For remote endpoints, assume they're valid (will fail later with better error)
    return True, ""


# ---------------------------------------------------------------------------
# Message store: thread-safe shared state between crew thread and Streamlit
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    role: str
    content: str
    msg_type: str
    timestamp: float = field(default_factory=time.time)


class ChatMessageStore:
    """Thread-safe container for chat messages shared between threads."""

    def __init__(self):
        self._messages: list[ChatMessage] = []
        self._lock = threading.Lock()
        self._done = False
        self._error: str | None = None
        self._cancelled = False

    def add(self, msg: ChatMessage):
        with self._lock:
            self._messages = [*self._messages, msg]

    def get_all(self) -> list[ChatMessage]:
        with self._lock:
            return list(self._messages)

    def mark_done(self):
        with self._lock:
            self._done = True

    def mark_error(self, error: str):
        with self._lock:
            self._error = error
            self._done = True

    def mark_cancelled(self):
        with self._lock:
            self._cancelled = True
            self._done = True

    @property
    def done(self) -> bool:
        with self._lock:
            return self._done

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled


# ---------------------------------------------------------------------------
# Module-level active store pointer — handlers write here (registered once)
# ---------------------------------------------------------------------------

_active_store: ChatMessageStore | None = None


# ---------------------------------------------------------------------------
# Register event handlers ONCE at module level (singleton event bus)
# ---------------------------------------------------------------------------

@crewai_event_bus.on(AgentExecutionStartedEvent)
def _on_agent_started(source, event: AgentExecutionStartedEvent):
    store = _active_store
    if store is not None:
        _progress_info["live"] = f"{event.agent.role} 开始发言"
        _progress_info["last_update"] = str(time.time())
        store.add(ChatMessage(
            role=event.agent.role,
            content=f"*{event.agent.role} 开始发言...*",
            msg_type="started",
        ))


@crewai_event_bus.on(AgentExecutionCompletedEvent)
def _on_agent_completed(source, event: AgentExecutionCompletedEvent):
    store = _active_store
    if store is not None:
        _progress_info["live"] = f"{event.agent.role} 发言完成"
        _progress_info["last_update"] = str(time.time())
        store.add(ChatMessage(
            role=event.agent.role,
            content=event.output,
            msg_type="completed",
        ))


@crewai_event_bus.on(TaskStartedEvent)
def _on_task_started(source, event: TaskStartedEvent):
    store = _active_store
    if store is not None:
        desc = ""
        if event.task and hasattr(event.task, "description"):
            desc = event.task.description[:80]
        _progress_info["live"] = f"任务开始: {desc}" if desc else "任务开始"
        _progress_info["last_update"] = str(time.time())
        store.add(ChatMessage(
            role="system",
            content=f"📋 任务开始: {desc}...",
            msg_type="task_started",
        ))


@crewai_event_bus.on(TaskCompletedEvent)
def _on_task_completed(source, event: TaskCompletedEvent):
    store = _active_store
    if store is not None:
        _progress_info["live"] = "任务完成"
        _progress_info["last_update"] = str(time.time())
        store.add(ChatMessage(
            role="system",
            content="✅ 任务完成",
            msg_type="task_completed",
        ))


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
    global _active_store
    repo = RoleRepository()
    base_db = repo.load_roles()
    total_steps = len(selected_style_ids) + 1  # +1 for comparison
    style_conversations: dict[str, str] = {}

    for idx, style_id in enumerate(selected_style_ids):
        style = LEADERSHIP_STYLES[style_id]
        store = style_stores[style_id]

        # Check if cancelled before starting
        if store.cancelled:
            logger.info(f"Simulation cancelled before starting {style.style_name}")
            break

        styled_db = apply_style_to_roles(base_db, style)
        builder = CrewBuilder(styled_db, config)
        crew = builder.build_crew(topic, num_rounds)

        try:
            _active_store = store
            progress_callback(idx, total_steps, style.style_name)
            logger.info(f"Starting simulation for style: {style.style_name}")
            logger.info(f"Topic: {topic}, Rounds: {num_rounds}")

            # Run crew with timeout monitoring
            start_time = time.time()
            crew.kickoff()
            elapsed = time.time() - start_time

            logger.info(f"Simulation completed for {style.style_name} in {elapsed:.1f}s")
            store.mark_done()
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Simulation failed for {style.style_name}: {error_msg}")

            # Attempt fallback with OpenRouter
            fallback_llm = create_openrouter_llm()
            if fallback_llm is not None:
                logger.info(f"Retrying {style.style_name} with OpenRouter fallback")
                _llm_route_info["active_provider"] = "openrouter"
                store.add(ChatMessage(
                    role="system",
                    content="⚠️ 主模型调用失败，正在切换到 OpenRouter 备选模型...",
                    msg_type="task_started",
                ))
                try:
                    fallback_builder = CrewBuilder(styled_db, config, llm=fallback_llm)
                    fallback_crew = fallback_builder.build_crew(topic, num_rounds)
                    start_time = time.time()
                    fallback_crew.kickoff()
                    elapsed = time.time() - start_time
                    logger.info(f"Fallback completed for {style.style_name} in {elapsed:.1f}s")
                    store.add(ChatMessage(
                        role="system",
                        content="✅ OpenRouter 备选模型完成模拟",
                        msg_type="task_completed",
                    ))
                    store.mark_done()
                except Exception as fallback_err:
                    fallback_msg = f"{type(fallback_err).__name__}: {str(fallback_err)}"
                    logger.error(f"Fallback also failed for {style.style_name}: {fallback_msg}")
                    store.mark_error(f"主模型: {error_msg}\nOpenRouter: {fallback_msg}")
            else:
                store.mark_error(error_msg)
        finally:
            _active_store = None

        style_conversations[style.style_name] = extract_conversation_text(
            store.get_all()
        )

    # Run cross-style comparison if multiple styles selected
    if len(selected_style_ids) > 1:
        comparison_store = style_stores.get("__comparison__")
        if comparison_store is not None:
            try:
                _active_store = comparison_store
                progress_callback(
                    len(selected_style_ids), total_steps, "跨风格对比分析"
                )
                logger.info("Starting cross-style comparison analysis")
                comparison_crew = build_comparison_crew(topic, style_conversations)

                start_time = time.time()
                comparison_crew.kickoff()
                elapsed = time.time() - start_time

                logger.info(f"Comparison analysis completed in {elapsed:.1f}s")
                comparison_store.mark_done()
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Comparison analysis failed: {error_msg}")

                fallback_llm = create_openrouter_llm()
                if fallback_llm is not None:
                    logger.info("Retrying comparison with OpenRouter fallback")
                    _llm_route_info["active_provider"] = "openrouter"
                    comparison_store.add(ChatMessage(
                        role="system",
                        content="⚠️ 主模型调用失败，正在切换到 OpenRouter 备选模型...",
                        msg_type="task_started",
                    ))
                    try:
                        fallback_crew = build_comparison_crew(
                            topic, style_conversations, llm=fallback_llm,
                        )
                        start_time = time.time()
                        fallback_crew.kickoff()
                        elapsed = time.time() - start_time
                        logger.info(f"Fallback comparison completed in {elapsed:.1f}s")
                        comparison_store.mark_done()
                    except Exception as fallback_err:
                        fallback_msg = f"{type(fallback_err).__name__}: {str(fallback_err)}"
                        logger.error(f"Fallback comparison also failed: {fallback_msg}")
                        comparison_store.mark_error(f"主模型: {error_msg}\nOpenRouter: {fallback_msg}")
                else:
                    comparison_store.mark_error(error_msg)
            finally:
                _active_store = None


# ---------------------------------------------------------------------------
# Background thread wrapper for multi-style simulation
# ---------------------------------------------------------------------------

_progress_info: dict[str, str] = {
    "step": "0",
    "total": "1",
    "label": "准备中...",
    "live": "等待任务启动",
}
_llm_route_info: dict[str, str] = {"active_provider": "primary"}


def _run_in_thread(topic, num_rounds, selected_style_ids, style_stores, config):
    """Background thread entry point for multi-style simulation."""

    def progress_callback(step, total, label):
        _progress_info["step"] = str(step + 1)
        _progress_info["total"] = str(total)
        _progress_info["label"] = label
        _progress_info["last_update"] = str(time.time())

    try:
        run_multi_style_simulation(
            topic, num_rounds, selected_style_ids, style_stores, progress_callback, config
        )
    finally:
        _progress_info["done"] = "true"


def _format_llm_route_label() -> str:
    """Return a human-readable current LLM route label."""
    provider = _llm_route_info.get("active_provider", "primary")
    if provider == "openrouter":
        return "OpenRouter 备选模型"
    return "主模型接口"


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
        available_llms = get_available_llms()
        primary_model = os.getenv("OPENAI_MODEL_NAME", "未配置")
        primary_base_url = os.getenv("OPENAI_BASE_URL", "未配置")
        openrouter_model = os.getenv("OPENROUTER_MODEL_NAME", "")
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        openrouter_in_available = any(label == "OpenRouter" for label, _ in available_llms)

        st.caption(f"当前使用: **{_format_llm_route_label()}**")
        st.caption(f"主接口: `{primary_base_url}`")
        st.caption(f"主模型: **{primary_model}**")
        if openrouter_key and openrouter_model and openrouter_in_available:
            st.caption("备选接口: `https://openrouter.ai/api/v1`")
            st.caption(f"备选模型: **OpenRouter/{openrouter_model}** ✅")
        elif openrouter_key and openrouter_model:
            st.caption(f"备选模型: **OpenRouter/{openrouter_model}** ⚠️ 已配置但当前环境不可用")
        else:
            st.caption("备选模型: 未配置 ❌")
        if len(available_llms) == 0:
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
        st.session_state.sim_started_at = time.time()

        _progress_info.clear()
        total_steps = len(selected_styles) + (1 if len(selected_styles) > 1 else 0)
        _progress_info["step"] = "0"
        _progress_info["total"] = str(max(total_steps, 1))
        _progress_info["label"] = "初始化任务"
        _progress_info["live"] = "等待第一个角色开始"
        _progress_info["last_update"] = str(time.time())
        _llm_route_info["active_provider"] = "primary"

        config = {
            "agent_timeout": agent_timeout,
            "max_iterations": max_iterations,
            "context_window": context_window,
        }

        thread = threading.Thread(
            target=_run_in_thread,
            args=(topic, num_rounds, list(selected_styles), stores, config),
            daemon=True,
        )
        thread.start()

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
            label = _progress_info.get("label", "准备中...")
            step = int(_progress_info.get("step", "0") or "0")
            total = max(int(_progress_info.get("total", "1") or "1"), 1)
            live = _progress_info.get("live", "等待任务启动")
            started_at = st.session_state.get("sim_started_at", time.time())
            elapsed = int(time.time() - started_at)
            progress_ratio = min(max(step / total, 0.0), 1.0)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"🔄 模拟进行中 — {label}")
                st.progress(progress_ratio, text=f"阶段进度: {step}/{total}")
                st.caption(f"已运行: {elapsed}s | 最近动作: {live}")
            with col2:
                if st.button("🛑 取消模拟", type="secondary", use_container_width=True):
                    # Mark all stores as cancelled
                    for store in style_stores.values():
                        store.mark_cancelled()
                    st.session_state.running = False
                    st.warning("⚠️ 模拟已取消")
                    st.rerun()
            time.sleep(1.5)
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

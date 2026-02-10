"""Agent 人群模拟研究平台 - Agent Crowd Simulation Platform

A Streamlit app that uses crewAI agents to simulate multi-agent crowd interactions.
Each agent plays a different social role and responds to a user-defined topic.
"""

import threading
import time
from dataclasses import dataclass, field

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Crew, Process, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.task_events import TaskCompletedEvent, TaskStartedEvent

from lib_custom.crew_builder import CrewBuilder
from lib_custom.role_repository import RoleRepository


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

    @property
    def done(self) -> bool:
        with self._lock:
            return self._done

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error


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
        store.add(ChatMessage(
            role=event.agent.role,
            content=f"*{event.agent.role} 开始发言...*",
            msg_type="started",
        ))


@crewai_event_bus.on(AgentExecutionCompletedEvent)
def _on_agent_completed(source, event: AgentExecutionCompletedEvent):
    store = _active_store
    if store is not None:
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
        store.add(ChatMessage(
            role="system",
            content=f"📋 任务开始: {desc}...",
            msg_type="task_started",
        ))


@crewai_event_bus.on(TaskCompletedEvent)
def _on_task_completed(source, event: TaskCompletedEvent):
    store = _active_store
    if store is not None:
        store.add(ChatMessage(
            role="system",
            content="✅ 任务完成",
            msg_type="task_completed",
        ))


# ---------------------------------------------------------------------------
# Avatar mapping for each role
# ---------------------------------------------------------------------------

ROLE_AVATARS = {
    "老板 (Boss)": "👔",
    "资深员工 (Senior)": "🦊",
    "新人 (Newbie)": "🐣",
    "HR": "🎭",
    "分析师 (Analyst)": "📊",
    "system": "⚙️",
}


# ---------------------------------------------------------------------------
# Build the crew: 4 social roles + sequential task chain
# ---------------------------------------------------------------------------

def build_crew(topic: str, num_rounds: int = 3) -> Crew:
    """Create a Crew with agents from role configuration.

    Generates num_rounds × N conversation tasks plus a final analyst task.
    """
    repo = RoleRepository()
    db = repo.load_roles()
    builder = CrewBuilder(db)
    return builder.build_crew(topic, num_rounds)


# ---------------------------------------------------------------------------
# Background thread runner
# ---------------------------------------------------------------------------

def run_crew_in_background(crew: Crew, store: ChatMessageStore):
    """Run crew.kickoff() in a background thread."""
    global _active_store
    try:
        _active_store = store
        crew.kickoff()
        store.mark_done()
    except Exception as e:
        store.mark_error(str(e))
    finally:
        _active_store = None


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
        return {"老板 (Boss)", "资深员工 (Senior)", "新人 (Newbie)", "HR"}


def format_messages_as_markdown(
    messages: list[ChatMessage], topic: str, num_rounds: int,
) -> str:
    """Format finished messages into a Markdown document for export."""
    lines = [
        "# Agent 人群模拟记录",
        f"\n## 话题：{topic}",
        f"\n对话轮数：{num_rounds}",
        "",
    ]

    current_round = 0
    msgs_in_round = 0
    analyst_content = ""

    for msg in messages:
        if msg.msg_type != "completed":
            continue

        if msg.role == "分析师 (Analyst)":
            analyst_content = msg.content
            continue

        conversation_roles = get_conversation_role_names()
        if msg.role not in conversation_roles:
            continue

        if msgs_in_round % 4 == 0:
            current_round += 1
            lines.append(f"\n### 第 {current_round} 轮\n")
        msgs_in_round += 1

        avatar = ROLE_AVATARS.get(msg.role, "")
        lines.append(f"**{avatar} {msg.role}:**\n")
        lines.append(f"{msg.content}\n")

    if analyst_content:
        lines.append("\n---\n")
        lines.append("### 分析总结\n")
        lines.append(analyst_content)

    return "\n".join(lines)


def render_messages(messages: list[ChatMessage]):
    """Render conversation messages (excludes analyst output)."""
    for msg in messages:
        if msg.msg_type == "started":
            continue
        if msg.role == "分析师 (Analyst)":
            continue
        avatar = ROLE_AVATARS.get(msg.role, "💬")
        if msg.role == "system":
            st.caption(msg.content)
        else:
            with st.chat_message(msg.role, avatar=avatar):
                st.markdown(f"**{msg.role}**")
                st.write(msg.content)


def render_analyst(messages: list[ChatMessage]):
    """Render the analyst output in a separate expander."""
    for msg in messages:
        if msg.role == "分析师 (Analyst)" and msg.msg_type == "completed":
            with st.expander("📊 群体互动分析", expanded=False):
                st.markdown(msg.content)
            return


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
    st.caption("基于 LLM 多智能体的群体行为模拟与分析")

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
            help="每轮4个角色各发言一次，轮数越多剧情越丰富",
        )
        start_btn = st.button(
            "🚀 开始模拟", type="primary", use_container_width=True,
        )

        st.divider()
        st.subheader("角色介绍")
        for role, avatar in ROLE_AVATARS.items():
            if role != "system":
                st.write(f"{avatar} **{role}**")

    # -- Initialize session state --
    if "store" not in st.session_state:
        st.session_state.store = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "finished_messages" not in st.session_state:
        st.session_state.finished_messages = []
    if "sim_topic" not in st.session_state:
        st.session_state.sim_topic = ""
    if "sim_num_rounds" not in st.session_state:
        st.session_state.sim_num_rounds = 3

    # -- Handle start button --
    if start_btn and not st.session_state.running:
        st.session_state.store = ChatMessageStore()
        st.session_state.running = True
        st.session_state.finished_messages = []
        st.session_state.sim_topic = topic
        st.session_state.sim_num_rounds = num_rounds

        crew = build_crew(topic, num_rounds)

        thread = threading.Thread(
            target=run_crew_in_background,
            args=(crew, st.session_state.store),
            daemon=True,
        )
        thread.start()

    # -- Main area: render based on state --
    if st.session_state.running and st.session_state.store is not None:
        store: ChatMessageStore = st.session_state.store

        if not store.done:
            st.info("🔄 模拟进行中，请稍候...")
            render_messages(store.get_all())
            time.sleep(1.0)
            st.rerun()

        # Crew finished — save results and stop polling
        final_messages = store.get_all()
        st.session_state.finished_messages = final_messages
        st.session_state.running = False

        if store.error:
            st.error(f"模拟出错: {store.error}")
        else:
            st.success("✅ 模拟完成！")
        render_messages(final_messages)
        render_analyst(final_messages)
        md = format_messages_as_markdown(
            final_messages,
            st.session_state.sim_topic,
            st.session_state.sim_num_rounds,
        )
        st.download_button(
            "📥 下载对话记录 (Markdown)",
            data=md,
            file_name="agent_crowd_simulation.md",
            mime="text/markdown",
        )

    elif st.session_state.finished_messages:
        st.success("✅ 模拟完成！")
        render_messages(st.session_state.finished_messages)
        render_analyst(st.session_state.finished_messages)
        md = format_messages_as_markdown(
            st.session_state.finished_messages,
            st.session_state.sim_topic,
            st.session_state.sim_num_rounds,
        )
        st.download_button(
            "📥 下载对话记录 (Markdown)",
            data=md,
            file_name="agent_crowd_simulation.md",
            mime="text/markdown",
        )

    else:
        st.info("👈 在左侧输入话题，点击「开始模拟」开始多智能体群体模拟")


main()

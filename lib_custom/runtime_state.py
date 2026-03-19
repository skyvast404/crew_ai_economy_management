"""Runtime shared state for Streamlit reruns.

Streamlit reruns the main script in a fresh module namespace. Module globals in
the main script therefore do NOT reliably persist across reruns, especially
while a background thread is still running.

This module is imported as a normal Python module, so it is cached in
sys.modules and persists across Streamlit reruns. Put thread-shared state here.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
import threading
import time

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)
from crewai.events.types.task_events import TaskCompletedEvent, TaskStartedEvent

from lib_custom.chat_store import ChatMessage, ChatMessageStore

# ContextVars for per-context active_store and current_prefix.
# Unlike threading.local, ContextVars are propagated by contextvars.copy_context(),
# which crewai's event bus uses when dispatching handlers to its thread pool.
_active_store_var: contextvars.ContextVar[ChatMessageStore | None] = (
    contextvars.ContextVar("_active_store", default=None)
)
_current_prefix_var: contextvars.ContextVar[str] = (
    contextvars.ContextVar("_current_prefix", default="")
)


@dataclass
class RuntimeState:
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Thread-shared, UI-readable progress info.
    progress_info: dict[str, str] = field(
        default_factory=lambda: {
            "step": "0",
            "total": "1",
            "label": "准备中...",
            "live": "等待任务启动",
            "last_update": str(time.time()),
        }
    )

    # Thread-shared LLM call info for UI display.
    llm_call_info: dict[str, str | int] = field(
        default_factory=lambda: {
            "status": "idle",
            "call_count": 0,
            "completed_count": 0,
            "failed_count": 0,
            "agent_role": "",
            "model": "",
            "call_started_at": "",
        }
    )

    def set_active_store(self, store: ChatMessageStore | None) -> None:
        """Set active store for the current context (propagated to event handlers)."""
        _active_store_var.set(store)

    def get_active_store(self) -> ChatMessageStore | None:
        """Get active store for the current context."""
        return _active_store_var.get()

    def snapshot_progress(self) -> dict[str, str]:
        with self.lock:
            return dict(self.progress_info)

    def snapshot_llm(self) -> dict[str, str | int]:
        with self.lock:
            return dict(self.llm_call_info)

    def set_progress(self, **updates: str) -> None:
        with self.lock:
            self.progress_info.update(updates)

    def set_llm(self, **updates: str | int) -> None:
        with self.lock:
            self.llm_call_info.update(updates)

    def set_current_prefix(self, prefix: str) -> None:
        """Set current prefix for the current context (propagated to event handlers)."""
        _current_prefix_var.set(prefix)

    def get_current_prefix(self) -> str:
        """Get current prefix for the current context."""
        return _current_prefix_var.get()


STATE = RuntimeState()

_handlers_registered = False


def ensure_event_handlers_registered() -> None:
    """Register crewai event handlers once per Python process."""
    global _handlers_registered
    if _handlers_registered:
        return
    _handlers_registered = True

    @crewai_event_bus.on(AgentExecutionStartedEvent)
    def _on_agent_started(source, event: AgentExecutionStartedEvent):
        store = STATE.get_active_store()
        if store is None:
            return
        prefix = STATE.get_current_prefix()
        task_id = str(getattr(event.task, "id", "") or "")
        key = f"{prefix}:{task_id}" if task_id else None
        STATE.set_progress(
            live=f"{event.agent.role} 开始发言",
            last_update=str(time.time()),
        )
        # Create/refresh the message container so UI shows something immediately.
        if key:
            store.upsert(
                key=key,
                role=event.agent.role,
                content="*正在生成...*",
                msg_type="stream",
            )
        else:
            store.add(ChatMessage(role=event.agent.role, content=f"*{event.agent.role} 开始发言...*", msg_type="started"))

    @crewai_event_bus.on(AgentExecutionCompletedEvent)
    def _on_agent_completed(source, event: AgentExecutionCompletedEvent):
        store = STATE.get_active_store()
        if store is None:
            return
        prefix = STATE.get_current_prefix()
        task_id = str(getattr(event.task, "id", "") or "")
        key = f"{prefix}:{task_id}" if task_id else None
        STATE.set_progress(
            live=f"{event.agent.role} 发言完成",
            last_update=str(time.time()),
        )
        # Ensure final text is visible even if no stream chunks arrived.
        if key:
            store.upsert(key=key, role=event.agent.role, content=event.output, msg_type="completed")
        else:
            store.add(ChatMessage(role=event.agent.role, content=event.output, msg_type="completed"))

    @crewai_event_bus.on(TaskStartedEvent)
    def _on_task_started(source, event: TaskStartedEvent):
        store = STATE.get_active_store()
        if store is None:
            return
        desc = ""
        if event.task and hasattr(event.task, "description"):
            desc = event.task.description[:80]
        STATE.set_progress(
            live=(f"任务开始: {desc}" if desc else "任务开始"),
            last_update=str(time.time()),
        )
        store.add(ChatMessage(role="system", content=f"📋 任务开始: {desc}...", msg_type="task_started"))

    @crewai_event_bus.on(TaskCompletedEvent)
    def _on_task_completed(source, event: TaskCompletedEvent):
        store = STATE.get_active_store()
        if store is None:
            return
        STATE.set_progress(
            live="任务完成",
            last_update=str(time.time()),
        )
        store.add(ChatMessage(role="system", content="✅ 任务完成", msg_type="task_completed"))

    @crewai_event_bus.on(LLMCallStartedEvent)
    def _on_llm_call_started(source, event: LLMCallStartedEvent):
        now = str(time.time())
        with STATE.lock:
            STATE.llm_call_info["status"] = "sending"
            STATE.llm_call_info["call_count"] = int(STATE.llm_call_info.get("call_count", 0)) + 1
            STATE.llm_call_info["agent_role"] = getattr(event, "agent_role", "") or ""
            STATE.llm_call_info["model"] = getattr(event, "model", "") or ""
            STATE.llm_call_info["call_started_at"] = now
            agent_role = STATE.llm_call_info.get("agent_role") or "LLM"
            STATE.progress_info["live"] = f"📡 发送请求... ({agent_role})"
            STATE.progress_info["last_update"] = now

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def _on_llm_call_completed(source, event: LLMCallCompletedEvent):
        now_f = time.time()
        now = str(now_f)
        with STATE.lock:
            started = float(STATE.llm_call_info.get("call_started_at", 0) or 0)
            duration = f" ({now_f - started:.1f}s)" if started > 0 else ""
            STATE.llm_call_info["status"] = "completed"
            STATE.llm_call_info["completed_count"] = int(STATE.llm_call_info.get("completed_count", 0)) + 1
            STATE.progress_info["live"] = f"✅ 收到响应{duration}"
            STATE.progress_info["last_update"] = now

    @crewai_event_bus.on(LLMCallFailedEvent)
    def _on_llm_call_failed(source, event: LLMCallFailedEvent):
        now = str(time.time())
        with STATE.lock:
            STATE.llm_call_info["status"] = "failed"
            STATE.llm_call_info["failed_count"] = int(STATE.llm_call_info.get("failed_count", 0)) + 1
            STATE.progress_info["live"] = f"❌ 请求失败: {str(event.error)[:60]}"
            STATE.progress_info["last_update"] = now

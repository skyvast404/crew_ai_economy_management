"""Thread-safe message store shared between background thread and Streamlit UI."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str
    content: str
    msg_type: str
    # Stable key for incremental updates (streaming). If None, message is append-only.
    key: str | None = None
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

    def upsert(self, key: str, role: str, content: str, msg_type: str) -> None:
        """Insert or update a message by key (used for streaming partial output)."""
        with self._lock:
            for i in range(len(self._messages) - 1, -1, -1):
                if self._messages[i].key == key:
                    self._messages[i] = ChatMessage(
                        key=key,
                        role=role,
                        content=content,
                        msg_type=msg_type,
                        timestamp=self._messages[i].timestamp,
                    )
                    return
            self._messages.append(ChatMessage(key=key, role=role, content=content, msg_type=msg_type))

    def finalize_streaming(self) -> None:
        """Convert any in-progress streaming messages to completed."""
        with self._lock:
            updated: list[ChatMessage] = []
            for m in self._messages:
                if m.msg_type == "stream":
                    updated.append(ChatMessage(key=m.key, role=m.role, content=m.content, msg_type="completed", timestamp=m.timestamp))
                else:
                    updated.append(m)
            self._messages = updated

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

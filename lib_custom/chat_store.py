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


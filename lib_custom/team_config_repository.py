"""Repository for team configuration persistence (JSON file)."""

from __future__ import annotations

import json
from pathlib import Path
import threading

from lib_custom.personality_types import TeamConfig


_DEFAULT_PATH = "team_config.json"


class TeamConfigRepository:
    """Thread-safe persistence layer for TeamConfig."""

    def __init__(self, config_path: str = _DEFAULT_PATH) -> None:
        self._path = Path(config_path)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def save_team_config(self, config: TeamConfig) -> None:
        """Persist *config* to JSON file (atomic write)."""
        with self._lock:
            self._atomic_write(config)

    def load_team_config(self) -> TeamConfig | None:
        """Load from JSON. Returns ``None`` when no file exists."""
        with self._lock:
            if not self._path.exists():
                return None
            try:
                with open(self._path) as fh:
                    data = json.load(fh)
                return TeamConfig(**data)
            except Exception as exc:
                raise ValueError(f"Failed to load team config: {exc}") from exc

    def delete_config(self) -> None:
        """Remove the config file if it exists."""
        with self._lock:
            if self._path.exists():
                self._path.unlink()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _atomic_write(self, config: TeamConfig) -> None:
        tmp = self._path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as fh:
                json.dump(config.model_dump(), fh, indent=2, ensure_ascii=False)
            tmp.replace(self._path)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            raise ValueError(f"Failed to save team config: {exc}") from exc

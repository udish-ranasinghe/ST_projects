"""
Session memory: per-session key-value store for entity extraction results,
verification state, and tool observation history.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

_store: Dict[str, "SessionMemory"] = {}
_lock = threading.Lock()

SESSION_TTL_SECONDS = 3600  # 1 hour


class SessionMemory:
    """Stores per-session state for the agent."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.created_at: float = time.time()
        self.last_active: float = time.time()
        self._data: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core key-value interface
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.last_active = time.time()

    def update(self, mapping: Dict[str, Any]) -> None:
        self._data.update(mapping)
        self.last_active = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def verified(self) -> bool:
        return bool(self._data.get("verified", False))

    @verified.setter
    def verified(self, value: bool) -> None:
        self._data["verified"] = value

    @property
    def last_order_id(self) -> Optional[str]:
        return self._data.get("last_order_id")

    @last_order_id.setter
    def last_order_id(self, value: str) -> None:
        self._data["last_order_id"] = value

    def record_tool_result(self, tool: str, result: Dict[str, Any]) -> None:
        history = self._data.setdefault("tool_history", [])
        history.append({"tool": tool, "result": result})
        self.last_active = time.time()


# ------------------------------------------------------------------
# Session store helpers
# ------------------------------------------------------------------

def get_or_create(session_id: str) -> SessionMemory:
    with _lock:
        _evict_expired()
        if session_id not in _store:
            _store[session_id] = SessionMemory(session_id)
        return _store[session_id]


def clear_session(session_id: str) -> None:
    with _lock:
        _store.pop(session_id, None)


def _evict_expired() -> None:
    now = time.time()
    expired = [sid for sid, mem in _store.items() if now - mem.last_active > SESSION_TTL_SECONDS]
    for sid in expired:
        del _store[sid]

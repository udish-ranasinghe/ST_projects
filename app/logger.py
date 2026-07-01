"""
Structured per-turn JSON logger.
Writes one JSON line per agent turn to a log file and also emits
a standard Python log record.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

_LOG_PATH = os.getenv("LOG_FILE", "logs/agent_turns.jsonl")

logger = logging.getLogger(__name__)


def _ensure_dir() -> None:
    log_dir = os.path.dirname(_LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)


def log_turn(
    session_id: str,
    user_message: str,
    plan: Dict[str, Any],
    observations: list,
    response: str,
    memory_snapshot: Dict[str, Any],
) -> None:
    """Append a structured JSON record to the log file."""
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "session_id": session_id,
        "user_message": user_message,
        "planner_goal": plan.get("goal"),
        "selected_tool": plan.get("tool"),
        "tool_args": plan.get("args"),
        "observations_summary": [
            {"tool": o.get("tool"), "ok": o.get("result", {}).get("ok")}
            for o in observations
        ],
        "response_preview": response[:300],
        "memory_keys": list(memory_snapshot.keys()),
    }

    try:
        _ensure_dir()
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.warning("Could not write turn log: %s", exc)

    logger.info(
        "turn | session=%s tool=%s ok=%s",
        session_id,
        plan.get("tool"),
        any(o.get("result", {}).get("ok") for o in observations),
    )

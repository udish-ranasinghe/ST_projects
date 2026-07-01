"""
Planner: wraps the LLM model to produce a structured plan dict.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.models import llm

logger = logging.getLogger(__name__)

ALLOWED_TOOLS: List[str] = [
    "track_order_tool",
    "refund_policy_tool",
    "update_account_tool",
    "billing_tool",
    "product_issue_tool",
    "handoff_tool",
]

_SAFE_FALLBACK = {
    "goal": "safe_fallback",
    "tool": "handoff_tool",
    "args": {},
    "done": False,
    "final_response": "",
}


def plan_step(
    user_message: str,
    entities: Dict[str, Any],
    memory_dict: Dict[str, Any],
    observations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ask the LLM (or mock) to produce one planning step.

    Returns a dict: {goal, tool, args, done, final_response}
    The returned tool is guaranteed to be in ALLOWED_TOOLS.
    """
    try:
        plan = llm.generate_plan(
            user_message=user_message,
            entities=entities,
            memory=memory_dict,
            observations=observations,
            tool_names=ALLOWED_TOOLS,
        )
    except Exception as exc:
        logger.error("Planner error: %s", exc)
        return dict(_SAFE_FALLBACK)

    # Allowlist enforcement — never execute an unlisted tool
    if plan.get("tool") not in ALLOWED_TOOLS:
        logger.warning("Planner returned unlisted tool '%s'; using handoff.", plan.get("tool"))
        plan["tool"] = "handoff_tool"

    return plan

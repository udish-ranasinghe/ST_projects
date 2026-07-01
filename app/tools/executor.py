"""
Tool executor: maps tool names to implementations and runs them.
Only tools in TOOL_REGISTRY may be invoked (explicit allowlist).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from app.tools import (
    billing,
    handoff,
    product_issue,
    refund_policy,
    track_order,
    update_account,
)

logger = logging.getLogger(__name__)

# Explicit allowlist of tool_name -> callable
TOOL_REGISTRY: Dict[str, Any] = {
    "track_order_tool": track_order.run,
    "refund_policy_tool": refund_policy.run,
    "update_account_tool": update_account.run,
    "billing_tool": billing.run,
    "product_issue_tool": product_issue.run,
    "handoff_tool": handoff.run,
}


def execute(tool_name: str, args: Dict[str, Any], memory_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool by name.

    Raises ValueError if the tool is not in the allowlist.
    Always returns a dict with at least {"ok": bool}.
    """
    if tool_name not in TOOL_REGISTRY:
        logger.error("Blocked attempt to call unlisted tool: %s", tool_name)
        raise ValueError(f"Tool '{tool_name}' is not in the allowed tool registry.")

    try:
        result = TOOL_REGISTRY[tool_name](args, memory_dict)
    except Exception as exc:
        logger.exception("Tool '%s' raised an exception: %s", tool_name, exc)
        result = {"ok": False, "error": "tool_exception", "message": str(exc)}

    return result

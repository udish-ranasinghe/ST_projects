"""
Agent loop: plan → act → observe → respond (max N steps per turn).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Tuple

from app.agent import memory as memory_store
from app.agent.entities import extract_entities
from app.agent.planner import plan_step
from app.tools.executor import execute
from app import logger as turn_logger

logger = logging.getLogger(__name__)

MAX_STEPS: int = int(os.getenv("AGENT_MAX_STEPS", "3"))

# Phrases that immediately trigger escalation before planning
_ESCALATION_PHRASES = [
    "legal", "sue", "court", "lawyer", "fraud",
    "threatening", "abuse", "police",
]


def _check_guardrails(message: str) -> bool:
    """Return True if the message should be immediately escalated."""
    lower = message.lower()
    return any(phrase in lower for phrase in _ESCALATION_PHRASES)


def _compose_response(tool: str, obs: Dict[str, Any], memory_dict: Dict[str, Any]) -> str:
    """Convert a tool observation into a user-friendly response string."""
    if tool == "track_order_tool":
        if not obs.get("ok"):
            return obs.get("message", "I could not retrieve your order details.")
        return (
            f"📦 **Order #{obs['order_id']}**\n"
            f"Status: **{obs['status']}**\n"
            f"Carrier: {obs['carrier']}\n"
            f"Estimated delivery: {obs['eta']}\n"
            f"Destination: {obs.get('destination', 'N/A')}"
        )

    if tool == "refund_policy_tool":
        if not obs.get("ok"):
            return obs.get("message", "I was unable to retrieve the refund policy.")
        return obs.get("answer", "Please contact a human agent for refund details.")

    if tool == "update_account_tool":
        if not obs.get("ok"):
            return obs.get("message", "I was unable to update your account.")
        field = obs.get("updated_field", "field")
        value = obs.get("new_value", "")
        return f"✅ Your **{field}** has been updated to `{value}` successfully."

    if tool == "billing_tool":
        return (
            f"🧾 {obs.get('message', 'Your billing case has been created.')}\n"
            f"Case ID: **{obs.get('case_id', 'N/A')}**"
        )

    if tool == "product_issue_tool":
        return (
            f"📋 {obs.get('message', 'Your product issue ticket has been created.')}\n"
            f"Ticket ID: **{obs.get('ticket_id', 'N/A')}**"
        )

    if tool == "handoff_tool":
        return (
            f"🙋 {obs.get('message', 'You are being escalated to a human agent.')}\n"
            f"Ticket ID: **{obs.get('ticket_id', 'N/A')}**"
        )

    return "I was unable to complete your request. Let me connect you with a human agent."


def run_turn(user_message: str, session_id: str) -> Dict[str, Any]:
    """
    Main entry point: run one agent turn and return a structured result dict.

    Returns:
        {
            "response": str,
            "plan": dict,
            "observations": list,
            "memory": dict,
            "session_id": str,
        }
    """
    # Input size guardrail
    max_chars: int = int(os.getenv("MAX_MSG_CHARS", "1000"))
    if len(user_message) > max_chars:
        return {
            "response": f"⚠️ Your message exceeds the {max_chars}-character limit. Please shorten it.",
            "plan": {},
            "observations": [],
            "memory": {},
            "session_id": session_id,
        }

    mem = memory_store.get_or_create(session_id)

    # Immediate escalation guardrail
    if _check_guardrails(user_message):
        from app.tools import handoff
        obs = handoff.run({}, mem.to_dict())
        response = _compose_response("handoff_tool", obs, mem.to_dict())
        plan = {"goal": "guardrail_escalation", "tool": "handoff_tool", "args": {}}
        turn_logger.log_turn(session_id, user_message, plan, [{"tool": "handoff_tool", "result": obs}], response, mem.to_dict())
        return {
            "response": response,
            "plan": plan,
            "observations": [{"tool": "handoff_tool", "result": obs}],
            "memory": mem.to_dict(),
            "session_id": session_id,
        }

    # Extract entities and update memory
    entities = extract_entities(user_message)
    if entities.get("verified"):
        mem.verified = True
    if entities.get("order_id"):
        mem.last_order_id = entities["order_id"]

    observations: List[Dict[str, Any]] = []
    last_plan: Dict[str, Any] = {}
    response: str = ""

    for step in range(MAX_STEPS):
        last_plan = plan_step(
            user_message=user_message,
            entities=entities,
            memory_dict=mem.to_dict(),
            observations=observations,
        )

        # Honor planner's "done" signal
        if last_plan.get("done") and last_plan.get("final_response"):
            response = last_plan["final_response"]
            break

        tool_name: str = last_plan.get("tool", "handoff_tool")
        args: Dict[str, Any] = last_plan.get("args", {})

        # Merge extracted entities into args for robustness
        for k, v in entities.items():
            args.setdefault(k, v)

        # Pass user_message to refund_policy_tool if no explicit question
        if tool_name == "refund_policy_tool":
            args.setdefault("question", user_message)

        obs = execute(tool_name, args, mem.to_dict())
        observations.append({"tool": tool_name, "args": args, "result": obs})
        mem.record_tool_result(tool_name, obs)

        response = _compose_response(tool_name, obs, mem.to_dict())

        # Single-step resolution: exit after first meaningful tool call
        # (extend here for multi-step if needed)
        break

    if not response:
        # Fallback if loop exhausted without a response
        from app.tools import handoff
        obs = handoff.run({}, mem.to_dict())
        observations.append({"tool": "handoff_tool", "args": {}, "result": obs})
        response = _compose_response("handoff_tool", obs, mem.to_dict())
        last_plan = {"goal": "max_steps_reached", "tool": "handoff_tool", "args": {}}

    turn_logger.log_turn(session_id, user_message, last_plan, observations, response, mem.to_dict())

    return {
        "response": response,
        "plan": last_plan,
        "observations": observations,
        "memory": mem.to_dict(),
        "session_id": session_id,
    }

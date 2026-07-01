"""
LLM model wrapper with Qwen2.5 Instruct support and deterministic mock fallback.
"""
from __future__ import annotations

import json
import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_generator = None
_mock_mode: bool = False


def _load_model() -> bool:
    """Attempt to load Qwen2.5 Instruct via transformers. Returns True on success."""
    global _generator, _mock_mode
    model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        dtype = "float16" if torch.cuda.is_available() else "float32"
        import torch as _torch
        torch_dtype = _torch.float16 if torch.cuda.is_available() else _torch.float32

        _generator = pipeline(
            task="text-generation",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=True,
        )
        logger.info("Model loaded: %s on device %d", model_id, device)
        _mock_mode = False
        return True
    except Exception as exc:
        logger.warning("Could not load model '%s': %s — falling back to mock mode.", model_id, exc)
        _mock_mode = True
        return False


def _ensure_model() -> None:
    global _generator, _mock_mode
    if _generator is None and not _mock_mode:
        _load_model()


# ---------------------------------------------------------------------------
# Mock deterministic planner responses
# ---------------------------------------------------------------------------
_MOCK_PLAN_RULES: list[tuple[list[str], dict]] = [
    # Product issues checked BEFORE order tracking to avoid "damaged/missing item" matching "order"/"arrived"
    (["damaged", "broken", "defective", "faulty", "incorrect item", "wrong item", "wrong product",
      "missing item", "missing product", "not working", "received the wrong"],
     {"goal": "product issue", "tool": "product_issue_tool", "args": {}, "done": False, "final_response": ""}),
    (["track", "parcel", "delivery", "shipping", "where is", "dispatch",
      "where's my order", "status of my order", "order status"],
     {"goal": "track order", "tool": "track_order_tool", "args": {}, "done": False, "final_response": ""}),
    (["refund", "return", "money back", "send back", "exchange", "cancel"],
     {"goal": "refund/return policy", "tool": "refund_policy_tool", "args": {}, "done": False, "final_response": ""}),
    (["update", "change", "email", "phone", "address", "profile"],
     {"goal": "update account", "tool": "update_account_tool", "args": {}, "done": False, "final_response": ""}),
    (["billing", "charged", "invoice", "payment", "twice", "overcharged", "subscription"],
     {"goal": "billing issue", "tool": "billing_tool", "args": {}, "done": False, "final_response": ""}),
    # General "missing" / "wrong" AFTER specific product-issue phrases to catch remaining cases
    (["missing", "wrong", "incorrect"],
     {"goal": "product issue", "tool": "product_issue_tool", "args": {}, "done": False, "final_response": ""}),
    # Bare "order" (without other product-issue context) → tracking
    (["order", "arrived", "status"],
     {"goal": "track order", "tool": "track_order_tool", "args": {}, "done": False, "final_response": ""}),
    (["human", "agent", "person", "operator", "supervisor", "manager", "escalate"],
     {"goal": "human handoff", "tool": "handoff_tool", "args": {}, "done": False, "final_response": ""}),
]


def mock_plan(user_message: str, entities: dict, memory: dict, observations: list) -> dict:
    """Deterministic keyword-based planner for demo/test mode."""
    msg = user_message.lower()
    for keywords, template in _MOCK_PLAN_RULES:
        if any(kw in msg for kw in keywords):
            plan = dict(template)
            plan["args"] = dict(entities)
            return plan
    # Default: handoff
    return {
        "goal": "fallback handoff",
        "tool": "handoff_tool",
        "args": {},
        "done": False,
        "final_response": "",
    }


def mock_text(system: str, user: str) -> str:
    """Simple canned text responses for common scenarios."""
    u = user.lower()
    if "policy" in u or "return" in u or "refund" in u:
        return (
            "Based on our policy, returns are accepted within 30 days of delivery "
            "for unused items in original packaging. Refunds are processed within 5–7 business days "
            "after inspection. Final-sale items are not eligible."
        )
    return "I have noted your request. A support team member will follow up shortly."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_plan(
    user_message: str,
    entities: dict,
    memory: dict,
    observations: list,
    tool_names: list[str],
) -> dict:
    """
    Call the LLM planner and return a structured plan dict.
    Falls back to mock_plan if the model is unavailable or output is unparseable.
    """
    _ensure_model()

    if _mock_mode or _generator is None:
        return mock_plan(user_message, entities, memory, observations)

    prompt = (
        "You are a customer-service planning agent. "
        f"Choose exactly one tool from: {tool_names}\n\n"
        "Return a JSON object with keys: goal, tool, args, done, final_response\n\n"
        "Rules:\n"
        "- If user requests a human agent, anger, or legal threats → handoff_tool\n"
        "- Order tracking/delivery/parcel questions → track_order_tool\n"
        "- Refund/return questions → refund_policy_tool\n"
        "- Account email/phone/address update → update_account_tool (requires verification)\n"
        "- Billing/payment/invoice issues → billing_tool\n"
        "- Damaged/wrong/missing product → product_issue_tool\n"
        "- When uncertain, prefer handoff_tool\n"
        "- Set done=true and provide final_response only when you can answer directly without a tool\n\n"
        f"user_message: {user_message}\n"
        f"extracted_entities: {json.dumps(entities)}\n"
        f"session_memory: {json.dumps(memory)}\n"
        f"previous_observations: {json.dumps(observations)}\n"
    )

    messages = [
        {"role": "system", "content": "Return only valid JSON. No markdown fences. No extra text."},
        {"role": "user", "content": prompt},
    ]
    try:
        out = _generator(messages, max_new_tokens=256, do_sample=False, return_full_text=False)
        txt = out[0]["generated_text"]
        if isinstance(txt, list):
            txt = txt[-1]["content"]
        txt = str(txt).strip()
        start = txt.find("{")
        end = txt.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(txt[start:end])
    except Exception as exc:
        logger.warning("LLM plan parse failed: %s", exc)

    return mock_plan(user_message, entities, memory, observations)


def generate_text(system_instruction: str, user_content: str, max_new_tokens: int = 160) -> str:
    """
    Generate free-form text (used for policy Q&A).
    Falls back to mock_text if the model is unavailable.
    """
    _ensure_model()

    if _mock_mode or _generator is None:
        return mock_text(system_instruction, user_content)

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content},
    ]
    try:
        out = _generator(messages, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
        txt = out[0]["generated_text"]
        if isinstance(txt, list):
            return txt[-1]["content"].strip()
        return str(txt).strip()
    except Exception as exc:
        logger.warning("LLM text generation failed: %s", exc)
        return mock_text(system_instruction, user_content)


def is_mock_mode() -> bool:
    _ensure_model()
    return _mock_mode


def set_mock_mode(enabled: bool = True) -> None:
    """Force mock mode — useful for testing."""
    global _mock_mode
    _mock_mode = enabled

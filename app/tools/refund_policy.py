"""
Refund / return policy tool.
Answers the customer's question grounded in the policy YAML text.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml

_POLICY_TEXT: str = ""


def _load_policy() -> str:
    global _POLICY_TEXT
    if not _POLICY_TEXT:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "policies.yaml")
        with open(config_path) as f:
            data = yaml.safe_load(f)
        _POLICY_TEXT = data.get("policies", {}).get("return_and_refund", "")
    return _POLICY_TEXT


def run(args: Dict[str, Any], memory_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Args expected:
        question (str): the customer's refund/return question
    """
    from app.models import llm  # imported here to allow mock injection in tests

    policy = _load_policy()
    question = args.get("question") or args.get("user_message") or "Tell me about the return policy."

    answer = llm.generate_text(
        system_instruction=(
            "You are a customer support assistant. "
            "Answer the question using ONLY the provided policy text. "
            "Be concise and accurate. "
            "If the policy does not address the question, say you will connect the customer to a human agent."
        ),
        user_content=f"Policy:\n{policy}\n\nCustomer question:\n{question}",
    )

    return {"ok": True, "answer": answer}

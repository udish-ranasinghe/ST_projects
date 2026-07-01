"""
Billing issue tool.
Creates a billing support case for manual review.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict


def run(args: Dict[str, Any], memory_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Args accepted (all optional):
        order_id  (str): related order, if known
        issue     (str): brief description of the billing issue
    """
    case_id = f"BILL-{uuid.uuid4().hex[:8].upper()}"
    order_id = args.get("order_id") or memory_dict.get("last_order_id")

    return {
        "ok": True,
        "case_id": case_id,
        "order_id": order_id,
        "message": (
            f"A billing support case has been created (Case ID: {case_id}). "
            "Our billing team will review this and contact you within 2 business days."
        ),
    }

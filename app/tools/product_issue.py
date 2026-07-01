"""
Product issue tool.
Creates a product issue ticket for damaged, wrong, or missing items.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict


def run(args: Dict[str, Any], memory_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Args accepted (all optional):
        order_id    (str): related order
        issue_type  (str): 'damaged' | 'wrong' | 'missing'
        description (str): free-text description
    """
    ticket_id = f"PROD-{uuid.uuid4().hex[:8].upper()}"
    order_id = args.get("order_id") or memory_dict.get("last_order_id")
    issue_type = args.get("issue_type", "unspecified")

    return {
        "ok": True,
        "ticket_id": ticket_id,
        "order_id": order_id,
        "issue_type": issue_type,
        "message": (
            f"A product issue ticket has been raised (Ticket ID: {ticket_id}). "
            "Our support team will contact you within 1 business day to arrange a replacement or refund."
        ),
    }

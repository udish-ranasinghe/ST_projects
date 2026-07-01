"""
Order tracking tool.
Looks up order status from the mock orders loaded from skills.json.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

_ORDERS: Optional[Dict[str, Any]] = None


def _load_orders() -> Dict[str, Any]:
    global _ORDERS
    if _ORDERS is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "skills.json")
        with open(config_path) as f:
            data = json.load(f)
        _ORDERS = data.get("mock_orders", {})
    return _ORDERS


def run(args: Dict[str, Any], memory_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Args expected:
        order_id (str): the order to look up

    Falls back to memory_dict["last_order_id"] if not in args.
    """
    orders = _load_orders()
    order_id = str(args.get("order_id") or memory_dict.get("last_order_id") or "").strip()

    if not order_id:
        return {
            "ok": False,
            "error": "missing_order_id",
            "message": "Please provide your order ID so I can track it.",
        }

    if order_id not in orders:
        return {
            "ok": False,
            "error": "order_not_found",
            "order_id": order_id,
            "message": f"Order #{order_id} was not found. Please double-check the order number.",
        }

    order = orders[order_id]
    return {
        "ok": True,
        "order_id": order_id,
        "status": order["status"],
        "eta": order["eta"],
        "carrier": order["carrier"],
        "destination": order.get("destination", ""),
    }

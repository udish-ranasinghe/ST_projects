"""
Account update tool.
Allows updating email, phone, or address — gated behind identity verification.
"""
from __future__ import annotations

from typing import Any, Dict

ALLOWED_FIELDS = {"email", "phone", "address"}


def run(args: Dict[str, Any], memory_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Args expected:
        field    (str): one of 'email', 'phone', 'address'
        new_value (str): the replacement value
        verified (bool, optional): explicit verification flag

    Verification is accepted from args OR from memory_dict["verified"].
    """
    field: str = (args.get("field") or "").strip().lower()
    new_value: str = str(args.get("new_value") or "").strip()
    verified: bool = bool(args.get("verified", False) or memory_dict.get("verified", False))

    if not field:
        return {
            "ok": False,
            "error": "missing_field",
            "message": "Please specify what you'd like to update: email, phone, or address.",
        }

    if field not in ALLOWED_FIELDS:
        return {
            "ok": False,
            "error": "invalid_field",
            "message": f"'{field}' is not a supported account field. Choose from: email, phone, address.",
        }

    if not new_value:
        return {
            "ok": False,
            "error": "missing_new_value",
            "message": f"Please provide the new {field} you'd like to use.",
        }

    if not verified:
        return {
            "ok": False,
            "error": "not_verified",
            "message": (
                "For security, account changes require identity verification. "
                "Please complete the OTP verification step first, then reply 'verified'."
            ),
        }

    # Persist the update in session memory
    memory_dict[f"profile_{field}"] = new_value

    return {
        "ok": True,
        "updated_field": field,
        "new_value": new_value,
        "message": f"Your {field} has been updated successfully.",
    }

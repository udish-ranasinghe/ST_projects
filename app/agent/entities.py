"""
Entity extractor: pulls order_id, email, phone, address, field, new_value,
and verification signals from raw user text.
"""
from __future__ import annotations

import re
from typing import Dict, Any


def extract_entities(message: str) -> Dict[str, Any]:
    """Return a dict of recognised entities found in *message*."""
    entities: Dict[str, Any] = {}
    msg_lower = message.lower()

    # Order ID — 5-10 digit number, optionally prefixed with # or 'order'
    # Use bounded quantifiers ({0,5}) to prevent polynomial ReDoS.
    order_match = re.search(r"(?:order\s{0,5}#?\s{0,5}|#)(\d{5,10})", message, re.IGNORECASE)
    if not order_match:
        # Bare 5-10 digit number
        order_match = re.search(r"\b(\d{5,10})\b", message)
    if order_match:
        entities["order_id"] = order_match.group(1)

    # Account field & new value
    if re.search(r"\bemail\b", msg_lower):
        entities["field"] = "email"
        # Bound local-part and domain lengths to prevent ReDoS
        email_match = re.search(r"[\w.+-]{1,64}@[\w-]{1,63}\.[a-zA-Z]{2,10}", message)
        if email_match:
            entities["new_value"] = email_match.group(0)
    elif re.search(r"\bphone\b|\bmobile\b|\bnumber\b", msg_lower):
        entities["field"] = "phone"
        # Match digits/separators with bounded length to prevent ReDoS
        phone_match = re.search(r"(\+?\d[\d\s\-().]{1,20}\d)", message)
        if phone_match:
            entities["new_value"] = phone_match.group(1).strip()
    elif re.search(r"\baddress\b", msg_lower):
        entities["field"] = "address"
        # Capture everything after "to" or after "address" keyword
        addr_match = re.search(
            r"(?:address\s+to|to\s+address|change\s+(?:my\s+)?address\s+to|address:?)\s+(.+)",
            message,
            re.IGNORECASE,
        )
        if addr_match:
            entities["new_value"] = addr_match.group(1).strip()

    # Verification signals
    if re.search(r"\bverified\b|\botp\s*(?:ok|confirmed|done)\b|\bcode\s+confirmed\b", msg_lower):
        entities["verified"] = True

    return entities

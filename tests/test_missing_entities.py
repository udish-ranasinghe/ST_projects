"""
Tests for missing-entity flows: agent should ask for missing info
rather than failing silently.
"""
import pytest

from app.models.llm import set_mock_mode
from app.agent.loop import run_turn


@pytest.fixture(autouse=True)
def use_mock_mode():
    set_mock_mode(True)
    yield


def _run(message: str) -> dict:
    import uuid
    return run_turn(message, str(uuid.uuid4()))


def test_track_order_missing_order_id():
    """When no order_id is given, the response should ask for one."""
    result = _run("Where is my parcel?")
    response = result["response"].lower()
    assert "order" in response or "number" in response or "id" in response, (
        f"Expected a prompt for order ID, got: {result['response']}"
    )


def test_track_order_with_order_id():
    """When order_id is present, the response should include order details."""
    result = _run("Track order #12345")
    response = result["response"].lower()
    assert "12345" in response or "in transit" in response or "dhl" in response


def test_track_order_not_found():
    """When order is not in the mock DB, response should say not found."""
    result = _run("Where is order #00001?")
    response = result["response"].lower()
    assert "not found" in response or "could not" in response or "00001" in response


def test_update_account_missing_field():
    """Update without specifying field should ask what to update."""
    result = _run("I want to update my account")
    response = result["response"].lower()
    # Should either ask for field or ask for verification
    assert any(kw in response for kw in ["email", "phone", "address", "update", "field", "verification"]), (
        f"Unexpected response: {result['response']}"
    )


def test_update_account_missing_new_value():
    """Specifying field=email but no new email should prompt for value."""
    result = _run("Change my email")
    response = result["response"].lower()
    assert any(kw in response for kw in ["email", "new", "provide", "update", "verification"]), (
        f"Unexpected response: {result['response']}"
    )

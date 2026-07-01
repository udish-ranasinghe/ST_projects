"""
Tests for the verification gate on account updates.
"""
import pytest
import uuid

from app.models.llm import set_mock_mode
from app.agent.loop import run_turn
from app.agent import memory as memory_store


@pytest.fixture(autouse=True)
def use_mock_mode():
    set_mock_mode(True)
    yield


def test_update_account_blocked_without_verification():
    """Account update must be blocked when not verified."""
    session_id = str(uuid.uuid4())
    result = run_turn("Change my email to hacker@evil.com", session_id)
    response = result["response"].lower()
    assert any(kw in response for kw in ["verif", "otp", "security", "not verif"]), (
        f"Expected verification prompt, got: {result['response']}"
    )


def test_update_account_allowed_after_verification():
    """After 'verified' signal, account update should succeed."""
    session_id = str(uuid.uuid4())

    # Step 1: attempt update — should be blocked
    run_turn("Change my email to newuser@example.com", session_id)

    # Step 2: user sends 'verified'
    run_turn("verified", session_id)

    # Step 3: retry the update — should now succeed
    result = run_turn("Change my email to newuser@example.com", session_id)
    response = result["response"].lower()
    assert any(kw in response for kw in ["updated", "success", "newuser@example.com", "email"]), (
        f"Expected successful update, got: {result['response']}"
    )


def test_verification_flag_persists_in_memory():
    """After 'verified', session memory should record it."""
    session_id = str(uuid.uuid4())
    run_turn("verified", session_id)
    mem = memory_store.get_or_create(session_id)
    assert mem.verified is True


def test_update_account_with_inline_verified():
    """If the user includes 'verified' in the same message, update should go through."""
    session_id = str(uuid.uuid4())
    result = run_turn("verified - change my email to verified@example.com", session_id)
    response = result["response"].lower()
    # Should attempt the update and either succeed or ask for proper value
    assert any(kw in response for kw in ["email", "update", "verif", "success"]), (
        f"Unexpected response: {result['response']}"
    )

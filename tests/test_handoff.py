"""
Tests for human handoff escalation behaviour.
"""
import pytest
import uuid

from app.models.llm import set_mock_mode
from app.agent.loop import run_turn


@pytest.fixture(autouse=True)
def use_mock_mode():
    set_mock_mode(True)
    yield


def _run(message: str) -> dict:
    return run_turn(message, str(uuid.uuid4()))


@pytest.mark.parametrize(
    "message",
    [
        "I want to speak to a human",
        "Connect me to an agent",
        "I want to talk to a person",
        "Give me a human",
        "I need to speak to someone",
    ],
)
def test_explicit_handoff_request(message: str):
    result = _run(message)
    response = result["response"].lower()
    assert any(kw in response for kw in ["ticket", "escalat", "human", "agent"]), (
        f"Expected handoff response for '{message}', got: {result['response']}"
    )


@pytest.mark.parametrize(
    "message",
    [
        "I will sue you",
        "This is fraud",
        "I am calling the police",
    ],
)
def test_guardrail_escalation(message: str):
    """Legal/threat keywords should trigger immediate handoff."""
    result = _run(message)
    response = result["response"].lower()
    assert any(kw in response for kw in ["ticket", "escalat", "human", "agent"]), (
        f"Expected guardrail escalation for '{message}', got: {result['response']}"
    )


def test_handoff_response_contains_ticket_id():
    result = _run("I want a human agent")
    response = result["response"]
    # Ticket ID format CS-XXXXXXXX
    import re
    assert re.search(r"CS-[A-F0-9]{8}", response), (
        f"Expected ticket ID in response, got: {response}"
    )


def test_handoff_tool_directly():
    """Test the handoff tool in isolation."""
    from app.tools.handoff import run
    obs = run({}, {})
    assert obs["ok"] is True
    assert "ticket_id" in obs
    assert obs["ticket_id"].startswith("CS-")

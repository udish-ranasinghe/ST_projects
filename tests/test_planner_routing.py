"""
Tests for planner routing: each representative prompt should be
mapped to the expected tool in mock mode.
"""
import pytest

from app.models.llm import set_mock_mode
from app.agent.planner import plan_step
from app.agent.entities import extract_entities


@pytest.fixture(autouse=True)
def use_mock_mode():
    set_mock_mode(True)
    yield
    set_mock_mode(True)  # keep mock after test


@pytest.mark.parametrize(
    "message,expected_tool",
    [
        ("Where is my order #12345?", "track_order_tool"),
        ("Track my parcel 55555", "track_order_tool"),
        ("What is the status of my delivery?", "track_order_tool"),
        ("I want a refund", "refund_policy_tool"),
        ("Can I return my shoes?", "refund_policy_tool"),
        ("What is your return policy?", "refund_policy_tool"),
        ("Change my email to test@example.com", "update_account_tool"),
        ("Update my phone number", "update_account_tool"),
        ("Update my address", "update_account_tool"),
        ("I was charged twice", "billing_tool"),
        ("Invoice issue", "billing_tool"),
        ("Payment problem", "billing_tool"),
        ("The item arrived damaged", "product_issue_tool"),
        ("I received the wrong product", "product_issue_tool"),
        ("Item is missing from my order", "product_issue_tool"),
        ("I want to speak to a human", "handoff_tool"),
        ("Connect me to an agent", "handoff_tool"),
        ("I want to talk to a person", "handoff_tool"),
    ],
)
def test_planner_routes_correctly(message: str, expected_tool: str):
    entities = extract_entities(message)
    plan = plan_step(
        user_message=message,
        entities=entities,
        memory_dict={},
        observations=[],
    )
    assert plan["tool"] == expected_tool, (
        f"Message: '{message}'\n"
        f"Expected: {expected_tool}\n"
        f"Got: {plan['tool']}"
    )


def test_planner_returns_allowed_tool_only():
    """Planner must never return a tool outside the allowlist."""
    from app.agent.planner import ALLOWED_TOOLS

    test_messages = [
        "Hello",
        "xyz-unknown-request",
        "some random text with no keywords",
    ]
    for msg in test_messages:
        entities = extract_entities(msg)
        plan = plan_step(msg, entities, {}, [])
        assert plan["tool"] in ALLOWED_TOOLS, f"Got unlisted tool: {plan['tool']}"


def test_planner_escalation_on_legal_phrase():
    """Guardrail: legal/threatening messages should reach handoff_tool."""
    from app.agent.loop import run_turn
    import uuid

    result = run_turn("I will sue you if this is not resolved", str(uuid.uuid4()))
    assert "handoff" in result["response"].lower() or "ticket" in result["response"].lower()

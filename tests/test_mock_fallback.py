"""
Tests for mock fallback planner/text behavior and health reporting.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import llm
from app.models.llm import mock_plan, mock_text, set_mock_mode


@pytest.fixture(autouse=True)
def use_mock_mode():
    set_mock_mode(True)
    yield
    set_mock_mode(True)


def test_mock_plan_returns_capabilities_for_greetings():
    plan = mock_plan("Hello, what can you do?", {}, {}, [])
    assert plan["done"] is True
    assert "order tracking" in plan["final_response"].lower()
    assert "billing" in plan["final_response"].lower()


def test_mock_plan_prioritizes_specific_intents_over_greetings():
    plan = mock_plan("Hey, I need a refund for my order", {}, {}, [])
    assert plan["tool"] == "refund_policy_tool"
    assert plan["done"] is False


@pytest.mark.parametrize(
    "message,expected_text",
    [
        ("I need help with billing", "billing"),
        ("Can you track my order?", "order"),
        ("Help me update my account email", "account updates"),
        ("What is your refund policy?", "returns are accepted within 30 days"),
    ],
)
def test_mock_text_keyword_coverage(message: str, expected_text: str):
    response = mock_text("", message).lower()
    assert expected_text in response


def test_unknown_prompt_returns_clarification():
    plan = mock_plan("Can you tell me a fun fact?", {}, {}, [])
    assert plan["done"] is True
    assert "i can help with" in plan["final_response"].lower()
    assert "could you share" in plan["final_response"].lower()


def test_health_reports_mock_mode():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0", "mock_mode": True}


def test_mock_mode_false_falls_back_deterministically_when_model_load_fails(monkeypatch):
    calls = {"count": 0}

    def fake_load() -> bool:
        calls["count"] += 1
        llm._mock_mode = True
        llm._model_load_attempted = True
        return False

    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setattr(llm, "_generator", None)
    monkeypatch.setattr(llm, "_mock_mode", False)
    monkeypatch.setattr(llm, "_mock_mode_forced", None)
    monkeypatch.setattr(llm, "_model_load_attempted", False)
    monkeypatch.setattr(llm, "_load_model", fake_load)

    llm._ensure_model()
    assert llm.is_mock_mode() is True
    assert calls["count"] == 1

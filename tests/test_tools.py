"""
Tests for individual tool implementations.
"""
import pytest
from app.models.llm import set_mock_mode


@pytest.fixture(autouse=True)
def use_mock_mode():
    set_mock_mode(True)
    yield


# ---------------------------------------------------------------------------
# track_order
# ---------------------------------------------------------------------------
class TestTrackOrder:
    def test_found(self):
        from app.tools.track_order import run
        obs = run({"order_id": "12345"}, {})
        assert obs["ok"] is True
        assert obs["order_id"] == "12345"
        assert obs["status"] == "In Transit"
        assert obs["carrier"] == "DHL"

    def test_not_found(self):
        from app.tools.track_order import run
        obs = run({"order_id": "00000"}, {})
        assert obs["ok"] is False
        assert obs["error"] == "order_not_found"

    def test_missing_id_falls_back_to_memory(self):
        from app.tools.track_order import run
        obs = run({}, {"last_order_id": "99999"})
        assert obs["ok"] is True
        assert obs["status"] == "Delivered"

    def test_missing_id_no_memory(self):
        from app.tools.track_order import run
        obs = run({}, {})
        assert obs["ok"] is False
        assert obs["error"] == "missing_order_id"


# ---------------------------------------------------------------------------
# update_account
# ---------------------------------------------------------------------------
class TestUpdateAccount:
    def test_blocked_without_verification(self):
        from app.tools.update_account import run
        obs = run({"field": "email", "new_value": "test@test.com"}, {})
        assert obs["ok"] is False
        assert obs["error"] == "not_verified"

    def test_allowed_with_memory_verification(self):
        from app.tools.update_account import run
        memory = {"verified": True}
        obs = run({"field": "email", "new_value": "new@test.com"}, memory)
        assert obs["ok"] is True
        assert obs["updated_field"] == "email"

    def test_allowed_with_arg_verification(self):
        from app.tools.update_account import run
        obs = run({"field": "phone", "new_value": "+447911123456", "verified": True}, {})
        assert obs["ok"] is True

    def test_missing_field(self):
        from app.tools.update_account import run
        obs = run({"verified": True, "new_value": "something"}, {"verified": True})
        assert obs["ok"] is False
        assert obs["error"] == "missing_field"

    def test_missing_new_value(self):
        from app.tools.update_account import run
        obs = run({"field": "email", "verified": True}, {"verified": True})
        assert obs["ok"] is False
        assert obs["error"] == "missing_new_value"

    def test_invalid_field(self):
        from app.tools.update_account import run
        obs = run({"field": "password", "new_value": "abc", "verified": True}, {})
        assert obs["ok"] is False
        assert obs["error"] == "invalid_field"

    def test_updates_memory(self):
        from app.tools.update_account import run
        memory = {"verified": True}
        run({"field": "email", "new_value": "stored@test.com"}, memory)
        assert memory.get("profile_email") == "stored@test.com"


# ---------------------------------------------------------------------------
# billing
# ---------------------------------------------------------------------------
class TestBilling:
    def test_creates_case(self):
        from app.tools.billing import run
        obs = run({}, {})
        assert obs["ok"] is True
        assert "case_id" in obs
        assert obs["case_id"].startswith("BILL-")

    def test_uses_memory_order_id(self):
        from app.tools.billing import run
        obs = run({}, {"last_order_id": "12345"})
        assert obs["order_id"] == "12345"


# ---------------------------------------------------------------------------
# product_issue
# ---------------------------------------------------------------------------
class TestProductIssue:
    def test_creates_ticket(self):
        from app.tools.product_issue import run
        obs = run({}, {})
        assert obs["ok"] is True
        assert "ticket_id" in obs
        assert obs["ticket_id"].startswith("PROD-")


# ---------------------------------------------------------------------------
# refund_policy
# ---------------------------------------------------------------------------
class TestRefundPolicy:
    def test_returns_answer(self):
        from app.tools.refund_policy import run
        obs = run({"question": "How many days do I have to return?"}, {})
        assert obs["ok"] is True
        assert len(obs["answer"]) > 0


# ---------------------------------------------------------------------------
# handoff
# ---------------------------------------------------------------------------
class TestHandoff:
    def test_creates_ticket(self):
        from app.tools.handoff import run
        obs = run({}, {})
        assert obs["ok"] is True
        assert obs["ticket_id"].startswith("CS-")


# ---------------------------------------------------------------------------
# executor allowlist
# ---------------------------------------------------------------------------
class TestExecutor:
    def test_unknown_tool_raises(self):
        from app.tools.executor import execute
        with pytest.raises(ValueError, match="not in the allowed tool registry"):
            execute("evil_tool", {}, {})

    def test_known_tool_runs(self):
        from app.tools.executor import execute
        obs = execute("handoff_tool", {}, {})
        assert obs["ok"] is True

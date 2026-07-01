"""pytest configuration — adds project root to sys.path and enables mock mode."""
import sys
import os

# Ensure project root is on sys.path so `app` is importable
sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    """Force mock mode for all tests (no GPU / model download needed)."""
    os.environ.setdefault("MOCK_MODE", "true")
    # Import and set mock mode early
    try:
        from app.models.llm import set_mock_mode
        set_mock_mode(True)
    except ImportError:
        pass

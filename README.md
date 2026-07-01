# 🤖 Customer Service AI Agent

A production-quality agentic customer service bot built with **FastAPI**, a **plan → act → observe → respond** agent loop, and a **Gradio** chat UI.

The agent uses **Qwen2.5-1.5B-Instruct** (via Hugging Face Transformers) for planning and policy Q&A, with a **deterministic mock mode** for demos and tests that requires no GPU or model download.

---

## Architecture

```
User message
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                        │
│                                                              │
│  Input validation ──► Guardrail check                        │
│          │                    │ (legal/threat → immediate    │
│          ▼                    │  handoff)                    │
│  Entity extraction            │                              │
│   (order_id, email,           │                              │
│    phone, address,            │                              │
│    verified flag)             │                              │
│          │                    │                              │
│          ▼                    │                              │
│  ┌─────────────────────────────────────────┐                 │
│  │         Agent Loop (max 3 steps)        │                 │
│  │                                         │                 │
│  │  ┌──────────┐  ┌──────────┐  ┌───────┐ │                 │
│  │  │ Planner  │→ │Executor  │→ │Observe│ │                 │
│  │  │ (LLM /   │  │(allowlist│  │       │ │                 │
│  │  │  mock)   │  │ checked) │  │       │ │                 │
│  │  └──────────┘  └──────────┘  └───────┘ │                 │
│  │           plan → act → observe          │                 │
│  └─────────────────────────────────────────┘                 │
│          │                                                   │
│          ▼                                                   │
│  Response composer ──► Session memory update                 │
│          │                                                   │
│          ▼                                                   │
│  Structured JSON turn log                                    │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
Gradio Chat UI (calls /chat endpoint)
```

### Key components

| Path | Purpose |
|---|---|
| `app/main.py` | FastAPI app, rate limiting, `/chat` & `/health` routes |
| `app/agent/loop.py` | Agent loop: plan → act → observe → respond |
| `app/agent/planner.py` | Wraps LLM to produce structured plan JSON |
| `app/agent/entities.py` | Regex-based entity extraction |
| `app/agent/memory.py` | Per-session in-memory state store |
| `app/tools/` | One module per tool (track order, refund, billing, …) |
| `app/tools/executor.py` | Explicit tool allowlist enforcement |
| `app/models/llm.py` | Qwen2.5 wrapper + deterministic mock fallback |
| `app/config/skills.json` | Skill definitions, mock order data, guardrail config |
| `app/config/policies.yaml` | Return/refund, billing, account-update policy text |
| `app/ui/chat.py` | Gradio chat interface |
| `app/logger.py` | Structured per-turn JSON logging |
| `tests/` | pytest test suite |

---

## Supported Skills / Tools

| Tool | Trigger keywords | Sensitive |
|---|---|---|
| `track_order_tool` | track, order, parcel, delivery, shipping | No |
| `refund_policy_tool` | refund, return, money back, exchange, cancel | No |
| `update_account_tool` | change email / phone / address | **Yes** (requires verification) |
| `billing_tool` | charged, invoice, payment, billing | No |
| `product_issue_tool` | damaged, broken, wrong, missing item | No |
| `handoff_tool` | human, agent, operator, escalate | No |

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### 1. Clone and install

```bash
git clone https://github.com/udish-ranasinghe/ST_projects.git
cd ST_projects
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (all defaults work for demo mode)
```

### 3. (Optional) Enable real LLM

To use Qwen2.5-1.5B-Instruct instead of mock mode, uncomment the ML lines in `requirements.txt` and reinstall:

```bash
pip install torch transformers accelerate sentencepiece
```

Then set `MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct` in `.env` (this is the default).

---

## Running

### Backend (FastAPI)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# or
python run_backend.py
```

API docs: http://localhost:8000/docs

### UI (Gradio)

In a second terminal:

```bash
python run_ui.py
```

Open http://localhost:7860 in your browser.

### Docker Compose (backend + UI together)

```bash
docker compose up --build
```

---

## Running Tests

```bash
pytest
```

Tests run in **mock mode** automatically (no GPU needed).

Test coverage:
- `test_planner_routing.py` — intent routing to expected tool for 18 prompts
- `test_missing_entities.py` — missing order ID / field / value flows
- `test_verification_gate.py` — account update blocked/allowed by verification state
- `test_handoff.py` — explicit handoff requests + guardrail escalation
- `test_tools.py` — unit tests for every tool and the executor allowlist
- `test_entities.py` — entity extraction correctness

---

## Sample Prompts

```
Where is my order #12345?
My parcel hasn't arrived yet
I want a refund for my shoes
What is your return policy?
I was charged twice for my order
The item arrived damaged
Change my email to new@example.com
verified
Change my email to new@example.com
I want to speak to a human agent
```

---

## Logs

Structured JSON turn logs are written to `logs/agent_turns.jsonl`.

Each record includes:

```json
{
  "timestamp": "2026-07-01T12:00:00+00:00",
  "session_id": "abc-123",
  "user_message": "Where is my order #12345?",
  "planner_goal": "track order",
  "selected_tool": "track_order_tool",
  "tool_args": {"order_id": "12345"},
  "observations_summary": [{"tool": "track_order_tool", "ok": true}],
  "response_preview": "📦 Order #12345 ...",
  "memory_keys": ["last_order_id"]
}
```

---

## Security & Guardrails

- **Tool allowlist**: only the 6 registered tools can be called; arbitrary tool names are rejected with `ValueError`
- **Verification gate**: account updates require the user to pass OTP verification first
- **Input size limit**: messages over 1000 characters are rejected
- **Rate limiting**: 20 requests/minute per IP (configurable)
- **Escalation triggers**: legal, fraud, threatening, or abusive language triggers immediate human handoff
- **No secrets in code**: use `.env` for all configuration (see `.env.example`)

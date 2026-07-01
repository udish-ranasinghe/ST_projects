"""
Gradio chat interface for the Customer Service AI Agent.

Run directly:
    python -m app.ui.chat
or via:
    python run_ui.py
"""
from __future__ import annotations

import json
import os
import uuid

import gradio as gr
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SHOW_DEBUG = os.getenv("SHOW_DEBUG", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Chat function
# ---------------------------------------------------------------------------

def chat_fn(message: str, history: list, session_state: dict) -> tuple[str, list, dict]:
    if not message or not message.strip():
        return "", history, session_state

    # Ensure a persistent session ID exists per UI session
    if "session_id" not in session_state:
        session_state["session_id"] = str(uuid.uuid4())

    session_id = session_state["session_id"]

    try:
        resp = requests.post(
            f"{BACKEND_URL}/chat",
            json={"message": message, "session_id": session_id, "debug": SHOW_DEBUG},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        response_text = data.get("response", "Sorry, something went wrong.")

        if SHOW_DEBUG and data.get("debug"):
            dbg = data["debug"]
            response_text += (
                "\n\n---\n"
                f"🔍 **Plan:** `{json.dumps(dbg.get('plan', {}))}`\n\n"
                f"📡 **Observations:** `{json.dumps(dbg.get('observations', []))}`\n\n"
                f"🧠 **Memory:** `{json.dumps(dbg.get('memory', {}))}`"
            )

    except requests.exceptions.ConnectionError:
        response_text = (
            "⚠️ Cannot reach the backend. "
            "Make sure the FastAPI server is running (`uvicorn app.main:app --reload`)."
        )
    except Exception as exc:
        response_text = f"⚠️ Error: {exc}"

    history.append([message, response_text])
    return "", history, session_state


def reset_fn(session_state: dict) -> tuple[list, dict]:
    """Clear chat history and create a new session."""
    if "session_id" in session_state:
        try:
            requests.delete(f"{BACKEND_URL}/session/{session_state['session_id']}", timeout=5)
        except Exception:
            pass
    session_state["session_id"] = str(uuid.uuid4())
    return [], session_state


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

EXAMPLE_PROMPTS = [
    "Where is my order #12345?",
    "My parcel hasn't arrived yet",
    "I want a refund for my shoes",
    "What is your return policy?",
    "I was charged twice",
    "The item arrived damaged",
    "Change my email to user@example.com",
    "verified",
    "I want to speak to a human agent",
]

CSS = """
.gradio-container {
    max-width: 820px !important;
    margin: auto !important;
}
footer { display: none !important; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Customer Service AI Agent") as demo:
        gr.Markdown(
            """
            # 🤖 Customer Service AI Agent
            **Agentic support bot** — plan · act · observe · respond

            Ask about orders, returns, billing, account updates, or product issues.
            """
        )

        session_state = gr.State({})

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=480,
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Type your message here…",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    reset_btn = gr.Button("🔄 New Session", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### Quick examples")
                for prompt in EXAMPLE_PROMPTS:
                    gr.Button(prompt, variant="secondary", size="sm").click(
                        lambda p=prompt: p,
                        outputs=msg_box,
                    )

        # Wire up events
        send_btn.click(
            fn=chat_fn,
            inputs=[msg_box, chatbot, session_state],
            outputs=[msg_box, chatbot, session_state],
        )
        msg_box.submit(
            fn=chat_fn,
            inputs=[msg_box, chatbot, session_state],
            outputs=[msg_box, chatbot, session_state],
        )
        reset_btn.click(
            fn=reset_fn,
            inputs=[session_state],
            outputs=[chatbot, session_state],
        )

        gr.Markdown(
            "_Tip: Use `SHOW_DEBUG=true` env var to reveal planner decisions and observations._"
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(
        server_name=os.getenv("UI_HOST", "0.0.0.0"),
        server_port=int(os.getenv("UI_PORT", "7860")),
        debug=False,
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        css=CSS,
    )


if __name__ == "__main__":
    main()

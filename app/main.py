"""
FastAPI entry point for the Customer Service AI Agent.
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from app.agent.loop import run_turn
from app.agent import memory as memory_store
from app.models.llm import is_mock_mode

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter (in-process, per IP)
# ---------------------------------------------------------------------------
_rate_counts: Dict[str, list] = defaultdict(list)
RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))


def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    window = 60
    _rate_counts[client_ip] = [t for t in _rate_counts[client_ip] if now - t < window]
    if len(_rate_counts[client_ip]) >= RATE_LIMIT_PER_MINUTE:
        return False
    _rate_counts[client_ip].append(now)
    return True


@asynccontextmanager
async def lifespan(_: FastAPI):
    mock_mode = is_mock_mode()
    if mock_mode:
        logger.warning("Application started in mock mode.")
        logger.info("To enable the real LLM, install the optional model dependencies. Set MOCK_MODE=false to force that path when dependencies are present.")
    else:
        logger.info("Application started with the real LLM backend.")
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Customer Service AI Agent",
    description="Agentic customer service bot: plan → act → observe → respond",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="Customer message")
    session_id: str = Field(..., min_length=1, max_length=64, description="Session identifier")
    debug: bool = Field(default=False, description="Include debug info in response")

    @validator("message")
    def strip_message(cls, v: str) -> str:
        return v.strip()


class ChatResponse(BaseModel):
    response: str
    session_id: str
    debug: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    mock_mode: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version="1.0.0", mock_mode=is_mock_mode())


@app.post("/chat", response_model=ChatResponse, tags=["agent"])
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Run one agent turn and return the response."""
    client_ip = request.client.host if request.client else "unknown"

    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait before sending more messages.",
        )

    result = run_turn(
        user_message=body.message,
        session_id=body.session_id,
    )

    debug_info = None
    if body.debug:
        debug_info = {
            "plan": result.get("plan"),
            "observations": result.get("observations"),
            "memory": result.get("memory"),
        }

    return ChatResponse(
        response=result["response"],
        session_id=result["session_id"],
        debug=debug_info,
    )


@app.delete("/session/{session_id}", tags=["ops"])
def clear_session(session_id: str) -> JSONResponse:
    """Clear a session's memory (e.g., for testing or user log-out)."""
    memory_store.clear_session(session_id)
    return JSONResponse({"cleared": True, "session_id": session_id})

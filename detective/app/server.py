"""
Detective FastAPI server — A2A endpoints + health checks.

Provides:
- GET  /.well-known/agent.json   → AgentCard (A2A discovery)
- POST /tasks/send               → A2A task handler (skill: score_lead)
- POST /score                    → Direct HTTP scoring endpoint
- GET  /health                   → Liveness check

On startup:
- Loads the pre-configured ICP from icp_config.json
- Starts a Redis subscriber background task for lead_ingested events
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

# Ensure detective root is importable
_detective_root = str(Path(__file__).resolve().parent.parent)
if _detective_root not in sys.path:
    sys.path.insert(0, _detective_root)

from .config import settings
from .scorer import load_icp_from_config, score_single_lead
from .event_emitter import DetectiveEventEmitter
from .subscriber import start_detective_subscriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Detective Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# A2A Pydantic models (same schema as Worker)
# ---------------------------------------------------------------------------

class DataPart(BaseModel):
    type: Literal["data"] = "data"
    data: Dict[str, Any]


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class A2AMessage(BaseModel):
    role: str = "user"
    parts: List[Union[DataPart, TextPart]]


class A2ATask(BaseModel):
    id: str
    message: A2AMessage
    model_config = ConfigDict(extra="allow")


class TaskStatus(BaseModel):
    state: Literal["submitted", "working", "completed", "failed"]
    message: Optional[str] = None


class ArtifactPart(BaseModel):
    type: str
    data: Any


class Artifact(BaseModel):
    parts: List[ArtifactPart]


class TaskResult(BaseModel):
    id: str
    status: TaskStatus
    artifacts: List[Artifact] = []


# ---------------------------------------------------------------------------
# AgentCard
# ---------------------------------------------------------------------------
DETECTIVE_AGENT_CARD: Dict[str, Any] = {
    "name": "Detective Agent",
    "description": "ICP-based lead scoring, company ranking, and persona selection for AgenticOutbound.",
    "url": f"http://detective:{settings.PORT}",
    "version": "1.0.0",
    "skills": [
        {
            "id": "score_lead",
            "name": "Score Lead",
            "description": "Accepts a lead_ingested payload and returns a scored lead with ICP match, similarity, and selected persona.",
            "inputModes": ["application/json"],
            "outputModes": ["application/json"],
        },
    ],
    "authentication": {"schemes": ["ApiKey"]},
}

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_icp_attributes = None
_icp_text = ""
_emitter: Optional[DetectiveEventEmitter] = None


# ---------------------------------------------------------------------------
# Direct scoring endpoint
# ---------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    """Direct HTTP scoring — accepts Inject's lead_ingested payload."""
    payload: Dict[str, Any]
    correlation_id: str = ""


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup() -> None:
    global _icp_attributes, _icp_text, _emitter

    # Load ICP config
    _icp_attributes = load_icp_from_config(settings.ICP_CONFIG_PATH)
    try:
        cfg_path = Path(settings.ICP_CONFIG_PATH)
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                _icp_text = cfg.get("icp_text", "")
    except Exception:
        pass

    logger.info(
        "ICP loaded: %d industries, %d target roles",
        len(_icp_attributes.industry),
        len(_icp_attributes.target_roles),
    )

    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.WORKER_URL}/v1/config", timeout=5.0)
            if resp.status_code == 200:
                worker_config = resp.json()
                settings.update_from_worker(worker_config)
                logger.info("Dynamic configuration synced from Worker module")
    except Exception as exc:
        logger.warning(f"Could not fetch dynamic configuration on startup: {exc}")

    # Event emitter
    _emitter = DetectiveEventEmitter(
        redis_url=settings.REDIS_URL,
        worker_url=settings.WORKER_URL,
    )

    # Start Redis subscriber as background task
    app.state.subscriber_task = asyncio.create_task(start_detective_subscriber())
    logger.info("Detective server started on port %d", settings.PORT)
    logger.info("A2A active — /.well-known/agent.json, /tasks/send")


@app.on_event("shutdown")
async def shutdown() -> None:
    if hasattr(app.state, "subscriber_task"):
        app.state.subscriber_task.cancel()
        try:
            await app.state.subscriber_task
        except asyncio.CancelledError:
            pass
    logger.info("Detective server shut down")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "detective"}


@app.get("/.well-known/agent.json")
async def get_agent_card() -> JSONResponse:
    return JSONResponse(content=DETECTIVE_AGENT_CARD)


@app.post("/score")
async def score_lead_direct(request: ScoreRequest) -> JSONResponse:
    """Score a single lead via direct HTTP POST."""
    try:
        scored = await score_single_lead(
            payload=request.payload,
            icp_attributes=_icp_attributes,
            icp_text=_icp_text,
            groq_api_key=settings.GROQ_API_KEY,
        )

        # Emit lead_scored event
        if _emitter and request.correlation_id:
            await _emitter.emit_lead_scored(
                correlation_id=request.correlation_id,
                scored_result=scored,
            )

        return JSONResponse(content=scored)
    except Exception as exc:
        logger.error("Scoring failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )


def _failed(task_id: str, message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=TaskResult(
            id=task_id,
            status=TaskStatus(state="failed", message=message),
        ).model_dump(),
    )


@app.post("/tasks/send")
async def tasks_send(task: A2ATask) -> JSONResponse:
    """A2A task handler — routes to the score_lead skill."""
    if not task.message.parts:
        return _failed(task.id, "message.parts must not be empty")

    first_part = task.message.parts[0]
    if not isinstance(first_part, DataPart):
        return _failed(task.id, "First part must be a DataPart")

    data = first_part.data
    skill = data.get("skill")

    if skill != "score_lead":
        return _failed(task.id, f"Unknown skill: '{skill}'")

    envelope = data.get("envelope")
    if not isinstance(envelope, dict):
        return _failed(task.id, "Missing or invalid 'envelope' in data")

    payload = envelope.get("payload", {})
    correlation_id = envelope.get("correlation_id", "")

    try:
        scored = await score_single_lead(
            payload=payload,
            icp_attributes=_icp_attributes,
            icp_text=_icp_text,
            groq_api_key=settings.GROQ_API_KEY,
        )

        # Emit event
        if _emitter and correlation_id:
            await _emitter.emit_lead_scored(
                correlation_id=correlation_id,
                scored_result=scored,
            )

        return JSONResponse(
            content=TaskResult(
                id=task.id,
                status=TaskStatus(state="completed"),
                artifacts=[
                    Artifact(parts=[ArtifactPart(type="data", data=scored)])
                ],
            ).model_dump()
        )
    except Exception as exc:
        logger.error("A2A score_lead failed: %s", exc)
        return _failed(task.id, f"Scoring error: {exc}", status_code=500)

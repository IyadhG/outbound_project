"""
Redis pub/sub subscriber for Detective module.

Subscribes to the 'lead_ingested' channel. For each valid EventEnvelope it
scores the lead, emits a 'lead_scored' event, and optionally forwards
qualified leads to the Writer service.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Set

import redis.asyncio as aioredis

from .config import settings
from .scorer import score_single_lead, load_icp_from_config
from .event_emitter import DetectiveEventEmitter

logger = logging.getLogger(__name__)

CHANNEL = "lead_ingested"

_REQUIRED_FIELDS: Set[str] = {
    "event_id",
    "correlation_id",
    "module",
    "event_type",
    "timestamp",
    "payload",
    "metadata",
}


def _validate_envelope(data: Dict[str, Any]) -> bool:
    """Return True if all required EventEnvelope fields are present."""
    return _REQUIRED_FIELDS.issubset(data.keys())


async def start_detective_subscriber() -> None:
    """
    Connect to Redis, subscribe to 'lead_ingested', score each lead,
    and emit 'lead_scored' events.

    Runs until cancelled (asyncio.CancelledError).
    """
    client: aioredis.Redis | None = None
    pubsub: aioredis.client.PubSub | None = None

    # Pre-load ICP once at subscriber start
    icp_attributes = load_icp_from_config(settings.ICP_CONFIG_PATH)
    icp_text = ""
    try:
        import json as _json
        from pathlib import Path

        cfg_path = Path(settings.ICP_CONFIG_PATH)
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = _json.load(f)
                icp_text = cfg.get("icp_text", "")
    except Exception:
        pass

    emitter = DetectiveEventEmitter(
        redis_url=settings.REDIS_URL,
        worker_url=settings.WORKER_URL,
    )

    # Lazy import to avoid circular deps
    writer_client = None
    if settings.AUTO_FORWARD_TO_WRITER:
        try:
            from .writer_client import WriterClient

            writer_client = WriterClient(writer_url=settings.WRITER_URL)
        except ImportError:
            logger.warning("WriterClient not available — will not auto-forward to Writer")

    try:
        client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        pubsub = client.pubsub()
        await pubsub.subscribe(CHANNEL, "config_updated")
        logger.info("Detective subscriber started — listening on '%s' and 'config_updated'", CHANNEL)

        async for message in pubsub.listen():
            if message["type"] != "message":
                continue

            raw = message.get("data", "")
            channel_name = message.get("channel")
            
            try:
                parsed_json = json.loads(raw)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Skipping malformed JSON: %s", exc)
                continue

            # Handle config hot-reload
            if channel_name == "config_updated":
                settings.update_from_worker(parsed_json)
                logger.info("Dynamic configuration hot-reloaded from Worker.")
                continue

            # Otherwise, assume lead_ingested envelope
            envelope = parsed_json
            if not isinstance(envelope, dict) or not _validate_envelope(envelope):
                logger.warning("Skipping invalid envelope")
                continue

            correlation_id = envelope["correlation_id"]
            payload = envelope["payload"]

            # Skip leads not ready for outreach
            readiness = payload.get("readiness_flags", {})
            if not readiness.get("ready_for_outreach", True):
                logger.info("Skipping lead %s — not ready for outreach", correlation_id)
                continue

            # Score the lead
            try:
                scored = await score_single_lead(
                    payload=payload,
                    icp_attributes=icp_attributes,
                    icp_text=icp_text,
                    groq_api_key=settings.GROQ_API_KEY,
                )
                logger.info(
                    "Scored lead correlation_id=%s final_score=%.3f qualified=%s",
                    correlation_id,
                    scored["final_score"],
                    scored["qualified_for_outreach"],
                )
            except Exception as exc:
                logger.error("Failed to score lead %s: %s", correlation_id, exc)
                continue

            # Emit lead_scored event to Worker
            try:
                await emitter.emit_lead_scored(
                    correlation_id=correlation_id,
                    scored_result=scored,
                )
            except Exception as exc:
                logger.error("Failed to emit lead_scored: %s", exc)

            # Forward qualified leads to Writer
            if scored["qualified_for_outreach"] and writer_client:
                try:
                    writer_response = await writer_client.send_scored_lead(
                        scored_result=scored,
                        envelope=envelope,
                    )
                    logger.info(
                        "Forwarded lead %s to Writer — success=%s",
                        correlation_id,
                        writer_response.get("success") if writer_response else False,
                    )
                except Exception as exc:
                    logger.warning("Writer forwarding failed for %s: %s", correlation_id, exc)

    except asyncio.CancelledError:
        logger.info("Detective subscriber shutting down gracefully")
        raise
    except Exception as exc:
        logger.error("Detective subscriber unexpected error: %s", exc)
        raise
    finally:
        if pubsub is not None:
            try:
                await pubsub.unsubscribe(CHANNEL)
                await pubsub.close()
            except Exception:
                pass
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass
        logger.info("Detective subscriber connection closed")

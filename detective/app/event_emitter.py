"""
Event emitter for Detective module.

Publishes lead_scored EventEnvelopes to Redis (primary) with HTTP POST
to Worker's /v1/events/ingest as fallback.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def build_envelope(
    module: str,
    event_type: str,
    correlation_id: str,
    payload: dict,
    metadata: Optional[dict] = None,
) -> Dict[str, Any]:
    """Build a canonical EventEnvelope."""
    return {
        "event_id": str(uuid.uuid4()),
        "correlation_id": correlation_id,
        "module": module,
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
        "metadata": metadata or {},
    }


class DetectiveEventEmitter:
    """Publishes events to Redis pub/sub with HTTP fallback to Worker."""

    def __init__(self, redis_url: str = "", worker_url: str = ""):
        self._redis = None
        self._worker_url = worker_url or os.environ.get("WORKER_URL", "http://api:8000")

        try:
            import redis.asyncio as aioredis

            url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
            self._redis = aioredis.from_url(url)
        except Exception as exc:
            logger.warning("Redis unavailable (%s); will use HTTP fallback only.", exc)

    async def emit_lead_scored(
        self,
        correlation_id: str,
        scored_result: dict,
    ) -> None:
        """Emit a lead_scored event."""
        envelope = build_envelope(
            module="detective",
            event_type="lead_scored",
            correlation_id=correlation_id,
            payload=scored_result,
        )
        await self._publish(envelope, channel="lead_scored")

    async def _publish(self, envelope: dict, channel: str) -> None:
        """Publish to Redis; fall back to HTTP POST on failure."""
        json_payload = json.dumps(envelope, default=str)

        # Try Redis first
        if self._redis is not None:
            try:
                await self._redis.publish(channel, json_payload)
                logger.info(
                    "Published %s event_id=%s to Redis channel '%s'",
                    envelope["event_type"],
                    envelope["event_id"],
                    channel,
                )
                return
            except Exception as exc:
                logger.warning("Redis publish failed: %s — falling back to HTTP", exc)

        # HTTP fallback
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self._worker_url}/v1/events/ingest",
                    json=envelope,
                )
                response.raise_for_status()
                logger.info(
                    "Published %s event_id=%s via HTTP to Worker",
                    envelope["event_type"],
                    envelope["event_id"],
                )
        except Exception as exc:
            logger.error(
                "Failed to publish %s event via both Redis and HTTP: %s",
                envelope["event_type"],
                exc,
            )

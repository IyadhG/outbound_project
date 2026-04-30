import asyncio
import json
import logging
import os
import uuid

logger = logging.getLogger(__name__)


class EventEmitter:
    def __init__(self, redis_url: str = None):
        self._redis = None
        self._queue = None

        try:
            import redis.asyncio as aioredis
            url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
            self._redis = aioredis.from_url(url)
        except Exception as e:
            logger.warning(
                "Redis unavailable (%s); falling back to in-memory queue.", e
            )
            self._queue = asyncio.Queue()

    async def emit_lead_ingested(self, payload: dict) -> None:
        try:
            envelope = {
                "event_id": str(uuid.uuid4()),
                "correlation_id": payload["correlation_id"],
                "module": "inject",
                "event_type": payload["event_type"],
                "timestamp": payload["timestamp"],
                "payload": payload,
                "metadata": {},
            }
            json_payload = json.dumps(envelope)
            if self._redis is not None:
                await self._redis.publish("lead_ingested", json_payload)
            else:
                logger.warning(
                    "Redis not available; pushing lead_ingested event to in-memory queue."
                )
                await self._queue.put(json_payload)
        except Exception as e:
            logger.warning("Failed to emit lead_ingested event: %s", e)

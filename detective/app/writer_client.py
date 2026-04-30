"""
Writer client — maps Detective's scored output to Writer's GenerateRequest
and calls Writer's /api/generate/simple endpoint.
"""

import logging
import os
from typing import Any, Dict, Optional

import httpx

from .config import settings

logger = logging.getLogger(__name__)


def _pick_channel(persona: dict) -> str:
    """Pick best channel based on available contact info."""
    email = persona.get("email", "")
    if email and email not in ("", "Non trouvé", "Non renseigné"):
        return "email"
    return "linkedin_dm"


def build_generate_request(scored_result: dict, envelope: dict) -> dict:
    """
    Map Detective's scored output + original Inject envelope into
    Writer's GenerateRequest format.
    """
    persona = scored_result.get("selected_persona") or {}
    company = envelope.get("payload", {}).get("company_data", {})
    intent_signals = envelope.get("payload", {}).get("intent_signals", {})
    correlation_id = envelope.get("correlation_id", "")

    # Parse sender value props from comma-separated env var
    value_props_raw = settings.SENDER_VALUE_PROPS
    value_props = [v.strip() for v in value_props_raw.split(",") if v.strip()] if value_props_raw else []

    pain_points_raw = settings.OFFER_PAIN_POINTS
    pain_points = [v.strip() for v in pain_points_raw.split(",") if v.strip()] if pain_points_raw else []

    return {
        # Target (from Detective persona)
        "target_prospect": persona.get("name", "Decision Maker"),
        "target_company": company.get("name", "Unknown Company"),
        "prospect_role": persona.get("job_title", "Decision Maker"),
        # Channel (auto-detected from persona contact info)
        "channel": _pick_channel(persona),
        "intent": "direct_outreach",
        "stage": "first_touch",
        # Style — use Writer defaults
        "personality": {},
        # Sender company (from client config)
        "company_details": {
            "company_name": settings.SENDER_COMPANY_NAME,
            "elevator_pitch": settings.SENDER_ELEVATOR_PITCH or None,
            "value_props": value_props,
        },
        # Offer (from client config)
        "selected_offer": {
            "offer_name": settings.OFFER_NAME,
            "solution_summary": settings.OFFER_SOLUTION_SUMMARY or None,
            "pain_points": pain_points,
            "cta": settings.OFFER_CTA or None,
        },
        # Detective enrichment data (passed through to Writer's Researcher agent)
        "detective_context": {
            "correlation_id": correlation_id,
            "score": scored_result.get("final_score", 0),
            "similarity_score": scored_result.get("similarity_score", 0),
            "intent_signals": intent_signals,
            "company_data": company,
            "selected_persona": persona,
            "filters_passed": scored_result.get("filters_passed", []),
        },
    }


class WriterClient:
    """HTTP client that sends scored leads to Writer for message generation."""

    def __init__(self, writer_url: str = "", timeout: float = 60.0):
        self.writer_url = writer_url or os.environ.get(
            "WRITER_URL", "http://writer:8003"
        )
        self.timeout = timeout

    async def send_scored_lead(
        self,
        scored_result: dict,
        envelope: dict,
    ) -> Optional[Dict[str, Any]]:
        """Send a scored lead to Writer's /api/generate/simple endpoint."""
        request_body = build_generate_request(scored_result, envelope)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.writer_url}/api/generate/simple",
                    json=request_body,
                )
                response.raise_for_status()
                result = response.json()
                logger.info(
                    "Writer response for %s: success=%s score=%s",
                    request_body["target_prospect"],
                    result.get("success"),
                    result.get("score"),
                )
                return result
        except httpx.TimeoutException:
            logger.warning(
                "Writer timeout for %s (%.0fs)",
                request_body["target_prospect"],
                self.timeout,
            )
            return None
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Writer HTTP error %s for %s",
                exc.response.status_code,
                request_body["target_prospect"],
            )
            return None
        except Exception as exc:
            logger.error("Writer client error: %s", exc)
            return None

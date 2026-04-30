import uuid
from datetime import datetime


class DetectiveFormatter:
    """Formats enriched company data into a Detective-ready payload."""

    def format(self, merged_profile: dict, personas: list, intent_signals: dict, processing_log: list = None) -> dict:
        """
        Build and return the full Detective-ready payload.

        Args:
            merged_profile: Enriched company profile dict (from Apollo + Apify merge).
            personas: List of persona dicts discovered for the company.
            intent_signals: Intent signals dict from IntentCollector.collect().
            processing_log: Optional list of processing log entries. Defaults to [].

        Returns:
            Detective-ready payload conforming to the schema in requirement 2.4.
        """
        company_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())

        company_data = {
            "name": merged_profile.get("name", ""),
            "domain": merged_profile.get("domain", ""),
            "industry": merged_profile.get("industry", ""),
            "location": merged_profile.get("location", {}),
            "website_url": merged_profile.get("website_url", ""),
            "linkedin_url": merged_profile.get("linkedin_url", ""),
            "estimated_num_employees": merged_profile.get("estimated_num_employees", ""),
            "annual_revenue": merged_profile.get("annual_revenue", ""),
            "founded_year": merged_profile.get("founded_year", ""),
        }

        raw_score = merged_profile.get("data_quality_score", 0.0)
        data_quality_score = float(raw_score) if raw_score is not None else 0.0

        enrichment_data = {
            "data_quality_score": data_quality_score,
            "confidence_scores": merged_profile.get("confidence_scores", {}),
        }

        formatted_personas = [
            {
                "name": p.get("name", ""),
                "title": p.get("title", ""),
                "email": p.get("email", ""),
                "phone": p.get("phone", ""),
                "linkedin_url": p.get("linkedin_url", ""),
                "enrichment_level": p.get("enrichment_level", ""),
            }
            for p in personas
        ]

        # has_valid_contact: any persona with a real email (not placeholder/empty)
        invalid_emails = {"Non trouvé", "Non renseigné", ""}
        has_valid_contact = any(
            p.get("email", "") not in invalid_emails
            for p in personas
        )

        data_completeness = max(0.0, min(1.0, float(merged_profile.get("data_quality_score", 0.0))))

        readiness_flags = {
            "has_valid_contact": has_valid_contact,
            "data_completeness": data_completeness,
            "ready_for_outreach": has_valid_contact and data_completeness >= 0.5,
        }

        return {
            "company_id": company_id,
            "correlation_id": correlation_id,
            "company_data": company_data,
            "enrichment_data": enrichment_data,
            "personas": formatted_personas,
            "intent_signals": intent_signals,
            "readiness_flags": readiness_flags,
            "event_type": "lead_ingested",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "processing_log": processing_log if processing_log is not None else [],
        }

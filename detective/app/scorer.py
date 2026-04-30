"""
Single-lead scorer — adapts Detective's batch pipeline to score one lead at a time.

Takes an Inject lead_ingested payload and scores it against a pre-loaded ICP.
Delegates to tool wrappers from agent_tools.py rather than instantiating
brain/ and ranking/ classes inline.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the detective root is on sys.path so brain/ and ranking/ imports work
_detective_root = str(Path(__file__).resolve().parent.parent)
if _detective_root not in sys.path:
    sys.path.insert(0, _detective_root)

from brain import ICPAttributes, ICPExtractionAgent, CompanyMatcher, GeoAgent  # noqa: E402
from agent_tools import (  # noqa: E402
    filter_companies as filter_companies_tool,
    rank_companies as rank_companies_tool,
    score_personas as score_personas_tool,
)

logger = logging.getLogger(__name__)


def load_icp_from_config(config_path: str) -> ICPAttributes:
    """Load pre-configured ICP from a JSON file."""
    import json

    path = Path(config_path)
    if not path.exists():
        logger.warning("ICP config file not found at %s — using empty ICP", config_path)
        return ICPAttributes()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ICPAttributes(**data)


def _extract_company_profile(payload: dict) -> Dict[str, Any]:
    """Convert Inject's lead_ingested payload into a company profile dict
    compatible with Detective's CompanyFilter and CompanyRanker."""
    company_data = payload.get("company_data", {})
    location = company_data.get("location", {})
    enrichment = payload.get("enrichment_data", {})

    # Parse employee range from string like "500-1000"
    emp_str = company_data.get("estimated_num_employees", "")
    num_employees = None
    if emp_str:
        try:
            parts = str(emp_str).replace(",", "").split("-")
            # Use the midpoint if it's a range
            if len(parts) == 2:
                num_employees = (int(parts[0].strip()) + int(parts[1].strip())) // 2
            else:
                num_employees = int(parts[0].strip())
        except (ValueError, IndexError):
            pass

    return {
        "name": company_data.get("name", ""),
        "domain": company_data.get("domain", ""),
        "industry": company_data.get("industry", ""),
        "country": location.get("country", ""),
        "city": location.get("city", ""),
        "website_url": company_data.get("website_url", ""),
        "linkedin_url": company_data.get("linkedin_url", ""),
        "estimated_num_employees": num_employees,
        "annual_revenue": company_data.get("annual_revenue", ""),
        "founded_year": company_data.get("founded_year", ""),
        "data_quality_score": enrichment.get("data_quality_score", 0.0),
        # Preserve the full payload for embedding
        "basic_info": company_data,
    }


def _extract_personas(payload: dict) -> List[Dict[str, Any]]:
    """Convert Inject's personas into Detective's persona format."""
    personas = payload.get("personas", [])
    result = []
    for p in personas:
        result.append({
            "full_name": p.get("name", ""),
            "first_name": p.get("name", "").split(" ")[0] if p.get("name") else "",
            "last_name": " ".join(p.get("name", "").split(" ")[1:]) if p.get("name") else "",
            "job_title_role": p.get("title", ""),
            "job_title_level": "",
            "emails": [{"address": p.get("email", "")}] if p.get("email") else [],
            "linkedin_url": p.get("linkedin_url", ""),
            "city": "",
            "country": "",
            "is_likely_to_engage": 0.5,
            "intent_strength": 5,
        })
    return result


async def score_single_lead(
    payload: dict,
    icp_attributes: ICPAttributes,
    icp_text: str = "",
    groq_api_key: str = "",
) -> Dict[str, Any]:
    """
    Score a single lead (from Inject's lead_ingested event) against a pre-loaded ICP.

    Returns a dict with:
        - final_score: float (0-1)
        - icp_match: bool
        - filters_passed: list of filter names the company passed
        - similarity_score: float (0-1)
        - intent_boost: float
        - selected_persona: dict with best persona
        - qualified_for_outreach: bool (based on threshold)
        - company_data: dict (original company data)
    """
    company_profile = _extract_company_profile(payload)
    personas = _extract_personas(payload)
    intent_signals = payload.get("intent_signals", {})
    company_key = company_profile.get("domain", "unknown").replace(".", "_")

    # -------------------------------------------------------------------------
    # Step 1: Filter by ICP criteria (employee count, country)
    # -------------------------------------------------------------------------
    filters_passed = []
    icp_dict = icp_attributes.model_dump()

    companies_dict = {company_key: company_profile}
    filter_result = filter_companies_tool.invoke({"companies": companies_dict, "icp_attributes": icp_dict})
    filtered = filter_result.get("results", {})

    if not filtered:
        logger.info("Lead %s did not pass ICP filters", company_key)
        return {
            "final_score": 0.0,
            "icp_match": False,
            "filters_passed": [],
            "similarity_score": 0.0,
            "intent_boost": 0.0,
            "selected_persona": None,
            "qualified_for_outreach": False,
            "company_data": company_profile,
        }

    filters_passed = ["employee_count", "country"]

    # -------------------------------------------------------------------------
    # Step 2: Compute embedding similarity
    # -------------------------------------------------------------------------
    similarity_score = 0.5  # default if embedding fails
    try:
        effective_icp_text = icp_text or _icp_to_text(icp_attributes)
        rank_result = rank_companies_tool.invoke({"companies": filtered, "icp_text": effective_icp_text})
        rankings = rank_result.get("results", [])
        if rankings:
            similarity_score = rankings[0].get("similarity_score", 0.5)
            filters_passed.append("similarity")
    except Exception as exc:
        logger.warning("Embedding similarity failed: %s — using default 0.5", exc)

    # -------------------------------------------------------------------------
    # Step 3: Intent boost
    # -------------------------------------------------------------------------
    intent_boost = 0.0
    try:
        news_count = len(intent_signals.get("recent_news", []))
        job_count = intent_signals.get("job_postings_count", 0)
        tech_changes = len(intent_signals.get("technology_changes", []))
        if news_count > 0:
            intent_boost += 0.03
        if job_count > 5:
            intent_boost += 0.02
        if tech_changes > 0:
            intent_boost += 0.02
        intent_boost = min(intent_boost, 0.05)
        if intent_boost > 0:
            filters_passed.append("intent")
    except Exception as exc:
        logger.warning("Intent scoring failed: %s", exc)

    final_score = min(similarity_score + intent_boost, 1.0)

    # -------------------------------------------------------------------------
    # Step 4: Rank personas
    # -------------------------------------------------------------------------
    selected_persona = None
    try:
        target_roles = icp_attributes.target_roles if icp_attributes.target_roles else []
        persona_result = score_personas_tool.invoke({
            "company_key": company_key,
            "company_data": company_profile,
            "personas": personas,
            "target_roles": target_roles,
        })

        if "error" not in persona_result:
            selected_persona = persona_result.get("selected_persona") or None
            if selected_persona:
                filters_passed.append("persona_ranked")
    except Exception as exc:
        logger.warning("Persona ranking failed: %s", exc)

    # Fallback: if no persona was ranked, use raw first persona
    if not selected_persona and personas:
        p = personas[0]
        selected_persona = {
            "name": p.get("full_name", "Unknown"),
            "job_title": p.get("job_title_role", "Unknown"),
            "email": p.get("emails", [{}])[0].get("address", "") if p.get("emails") else "",
            "linkedin": p.get("linkedin_url", ""),
            "persona_score": 0.3,
            "is_sales_dept": False,
            "is_ceo": False,
            "is_target": False,
        }

    return {
        "final_score": round(final_score, 4),
        "icp_match": True,
        "filters_passed": filters_passed,
        "similarity_score": round(similarity_score, 4),
        "intent_boost": round(intent_boost, 4),
        "selected_persona": selected_persona,
        "qualified_for_outreach": final_score >= float(
            os.getenv("QUALIFICATION_THRESHOLD", "0.6")
        ),
        "company_data": company_profile,
    }


def _icp_to_text(icp: ICPAttributes) -> str:
    """Convert structured ICP attributes back to a text description for embedding."""
    parts = []
    if icp.industry:
        parts.append(f"Industries: {', '.join(icp.industry)}")
    if icp.company_size.min or icp.company_size.max:
        parts.append(
            f"Company size: {icp.company_size.min or '?'}-{icp.company_size.max or '?'} employees"
        )
    if icp.target_countries:
        parts.append(f"Countries: {', '.join(icp.target_countries)}")
    if icp.target_roles:
        parts.append(f"Target roles: {', '.join(icp.target_roles)}")
    if icp.must_have_traits:
        parts.append(f"Must have: {', '.join(icp.must_have_traits)}")
    return ". ".join(parts)

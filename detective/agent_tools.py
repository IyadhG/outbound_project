"""
Agent Tools — LangChain @tool-decorated wrappers for all brain/ and ranking/ modules.

Each wrapper:
1. Lazily creates a singleton class instance on first call (stored in _instances).
2. Delegates to the underlying class method.
3. Serializes results to JSON-compatible dicts.
4. Catches all exceptions and returns {"error": str(e), "results": [], "count": 0}.

Requirements: 1.3, 6.1, 6.2, 6.3
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton instance registry — populated lazily on first tool call.
# Keys: "icp_agent", "company_matcher", "geo_agent", "company_filter",
#       "company_ranker", "final_scorer", "persona_ranker", "persona_scorer"
# ---------------------------------------------------------------------------
_instances: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# LangChain tool decorator — prefer langchain_core, fall back to langchain
# ---------------------------------------------------------------------------
try:
    from langchain_core.tools import tool
except ImportError:
    from langchain.tools import tool  # type: ignore


# ---------------------------------------------------------------------------
# Helper: lazy Groq client
# ---------------------------------------------------------------------------
def _get_groq_client():
    """Return a cached Groq client, creating it on first call."""
    if "groq_client" not in _instances:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        _instances["groq_client"] = Groq(api_key=api_key)
    return _instances["groq_client"]


# ===========================================================================
# Tool 1: extract_icp
# ===========================================================================

@tool
def extract_icp(icp_text: str) -> dict:
    """Extract structured ICP attributes from natural language text.

    Args:
        icp_text: Raw natural-language description of the Ideal Customer Profile.

    Returns:
        A dict representation of ICPAttributes (via model_dump()), or an error dict.
    """
    try:
        if "icp_agent" not in _instances:
            from brain.icp_agent import ICPExtractionAgent
            groq_client = _get_groq_client()
            _instances["icp_agent"] = ICPExtractionAgent(groq_client=groq_client)

        agent = _instances["icp_agent"]
        icp_attributes = agent.extract_icp_attributes(icp_text)
        return icp_attributes.model_dump()

    except Exception as e:
        logger.error("extract_icp failed: %s", e)
        return {"error": str(e), "results": [], "count": 0}


# ===========================================================================
# Tool 2: match_companies
# ===========================================================================

@tool
def match_companies(industries: list) -> dict:
    """Find companies matching the given industry list from merged_profiles.

    Args:
        industries: List of target industry strings extracted from the ICP.

    Returns:
        {"results": {company_key: profile, ...}, "count": int}
    """
    try:
        if "company_matcher" not in _instances:
            from brain.company_matcher import CompanyMatcher
            groq_client = _get_groq_client()
            _instances["company_matcher"] = CompanyMatcher(groq_client=groq_client)

        matcher = _instances["company_matcher"]
        results = matcher.find_matching_companies(list(industries))
        return {"results": results, "count": len(results)}

    except Exception as e:
        logger.error("match_companies failed: %s", e)
        return {"error": str(e), "results": [], "count": 0}


# ===========================================================================
# Tool 3: geo_filter
# ===========================================================================

@tool
def geo_filter(companies: dict, city: str, country: str, range_km: float) -> dict:
    """Filter companies by proximity to a target city (requires ORS_API_KEY).

    If ORS_API_KEY is absent the input companies are returned unchanged with
    a ``skipped`` flag so the agent can continue without geo-filtering.

    Args:
        companies: Dict of company profiles keyed by company_key.
        city: Target city name.
        country: Target country name.
        range_km: Maximum driving distance in kilometres.

    Returns:
        {"results": {...}, "count": int} or {"results": {...}, "count": int, "skipped": True}
    """
    try:
        ors_api_key = os.environ.get("ORS_API_KEY")
        if not ors_api_key:
            logger.info("ORS_API_KEY absent — geo_filter skipped, returning input unchanged")
            return {"results": companies, "count": len(companies), "skipped": True}

        if "geo_agent" not in _instances:
            from brain.geo_agent import GeoAgent
            _instances["geo_agent"] = GeoAgent(api_key=ors_api_key)

        agent = _instances["geo_agent"]
        filtered = agent.filter_companies_by_proximity(
            companies=companies,
            target_city=city,
            target_country=country,
            range_km=float(range_km),
        )
        return {"results": filtered, "count": len(filtered)}

    except Exception as e:
        logger.error("geo_filter failed: %s", e)
        return {"error": str(e), "results": [], "count": 0}


# ===========================================================================
# Tool 4: filter_companies
# ===========================================================================

@tool
def filter_companies(companies: dict, icp_attributes: dict) -> dict:
    """Filter companies by employee count and country from ICP constraints.

    Args:
        companies: Dict of company profiles keyed by company_key.
        icp_attributes: ICP attributes dict (as returned by extract_icp).

    Returns:
        {"results": {...}, "count": int}
    """
    try:
        # CompanyFilter is stateful (holds icp), so we create a fresh instance
        # each call to respect the provided icp_attributes.
        from ranking.company_filter import CompanyFilter
        company_filter = CompanyFilter(icp_attributes=icp_attributes)
        filtered = company_filter.filter_companies(companies)
        return {"results": filtered, "count": len(filtered)}

    except Exception as e:
        logger.error("filter_companies failed: %s", e)
        return {"error": str(e), "results": [], "count": 0}


# ===========================================================================
# Tool 5: rank_companies
# ===========================================================================

@tool
def rank_companies(companies: dict, icp_text: str) -> dict:
    """Rank companies by semantic similarity to the ICP using Gemini embeddings.

    If GeminiEmbedder raises during initialisation or ranking, all companies
    are returned with similarity_score=0.5 as a graceful fallback.

    Args:
        companies: Dict of company profiles keyed by company_key.
        icp_text: Raw ICP text used to build the embedding query.

    Returns:
        {"results": [...ranked dicts...], "count": int}
    """
    try:
        # Try to build a GeminiEmbedder; fall back gracefully on failure.
        embedder = None
        try:
            from ranking.embedder import GeminiEmbedder
            gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            embedder = GeminiEmbedder(api_key=gemini_api_key)
        except Exception as emb_err:
            logger.warning("GeminiEmbedder initialisation failed (%s) — using similarity_score=0.5 fallback", emb_err)

        if embedder is None:
            # Fallback: return companies as a list with default similarity_score
            fallback = []
            for company_key, company_data in companies.items():
                basic = company_data.get("basic_info", {})
                fallback.append({
                    "company_key": company_key,
                    "company_name": basic.get("name", company_key),
                    "similarity_score": 0.5,
                    "intent_score": 0.0,
                    "final_score": 0.5,
                    "company_data": company_data,
                })
            return {"results": fallback, "count": len(fallback)}

        from ranking.company_ranker import CompanyRanker
        ranker = CompanyRanker(embedder=embedder)

        try:
            ranker.embed_icp(icp_text)
            ranked = ranker.rank_companies(companies)
        except Exception as rank_err:
            logger.warning("rank_companies raised (%s) — using similarity_score=0.5 fallback", rank_err)
            fallback = []
            for company_key, company_data in companies.items():
                basic = company_data.get("basic_info", {})
                fallback.append({
                    "company_key": company_key,
                    "company_name": basic.get("name", company_key),
                    "similarity_score": 0.5,
                    "intent_score": 0.0,
                    "final_score": 0.5,
                    "company_data": company_data,
                })
            return {"results": fallback, "count": len(fallback)}

        return {"results": ranked, "count": len(ranked)}

    except Exception as e:
        logger.error("rank_companies failed: %s", e)
        return {"error": str(e), "results": [], "count": 0}


# ===========================================================================
# Tool 6: collect_intent
# ===========================================================================

@tool
def collect_intent(company_names: list) -> dict:
    """Collect intent signals for a list of companies (optional; skipped if unavailable).

    This tool is a best-effort stub.  Any failure (missing API key, network
    error, etc.) causes it to return a skipped result so the agent can
    continue without intent signals.

    Args:
        company_names: List of company name strings.

    Returns:
        {"results": {...}, "count": int} or {"results": {}, "count": 0, "skipped": True}
    """
    try:
        # Intent collection is optional — if no implementation is available,
        # return a skipped result immediately.
        return {"results": {}, "count": 0, "skipped": True}

    except Exception as e:
        logger.warning("collect_intent failed: %s — skipping", e)
        return {"results": {}, "count": 0, "skipped": True}


# ===========================================================================
# Tool 7: calculate_final_scores
# ===========================================================================

@tool
def calculate_final_scores(ranked_companies: list, intent_results: dict) -> dict:
    """Calculate final scores combining similarity and LLM-evaluated intent boost.

    Args:
        ranked_companies: List of ranked company dicts (as returned by rank_companies).
        intent_results: Intent signals dict keyed by company_key (may be empty).

    Returns:
        {"results": [...], "count": int}
    """
    try:
        if "final_scorer" not in _instances:
            from ranking.final_scorer import FinalScorer
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is not set")
            _instances["final_scorer"] = FinalScorer(api_key=api_key)

        scorer = _instances["final_scorer"]
        results = scorer.calculate_final_scores(
            ranked_companies=list(ranked_companies),
            intent_results=intent_results or {},
        )
        return {"results": results, "count": len(results)}

    except Exception as e:
        logger.error("calculate_final_scores failed: %s", e)
        return {"error": str(e), "results": [], "count": 0}


# ===========================================================================
# Tool 8: score_personas
# ===========================================================================

@tool
def score_personas(
    company_key: str,
    company_data: dict,
    personas: list,
    target_roles: list,
) -> dict:
    """Score and rank personas for a company using rule-based + optional LLM scoring.

    Iterates over *personas*, calls PersonaScorer.score() for each, collects
    results, counts LLM escalations, and returns the highest-scored persona as
    ``selected_persona``.

    Args:
        company_key: Unique identifier for the company.
        company_data: Company profile dict.
        personas: List of raw persona dicts.
        target_roles: List of target role strings from the ICP.

    Returns:
        {"selected_persona": {...}, "all_scored": [...], "llm_escalations": int}
    """
    try:
        # Build PersonaRanker + PersonaScorer lazily.
        # PersonaRanker is keyed by target_roles so we rebuild when they change.
        roles_key = tuple(sorted(target_roles or []))
        ranker_cache_key = f"persona_ranker_{roles_key}"

        if ranker_cache_key not in _instances:
            from ranking.persona_ranker import PersonaRanker
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is not set")
            _instances[ranker_cache_key] = PersonaRanker(
                target_roles=list(target_roles or []),
                api_key=api_key,
            )

        persona_ranker = _instances[ranker_cache_key]

        scorer_cache_key = f"persona_scorer_{roles_key}"
        if scorer_cache_key not in _instances:
            from persona_scorer import PersonaScorer
            _instances[scorer_cache_key] = PersonaScorer(persona_ranker=persona_ranker)

        scorer = _instances[scorer_cache_key]

        all_scored = []
        llm_escalations = 0

        for persona in personas:
            try:
                scored_persona, was_escalated = scorer.score(persona)
                all_scored.append(scored_persona)
                if was_escalated:
                    llm_escalations += 1
            except Exception as persona_err:
                logger.warning(
                    "score_personas: failed to score persona for %s: %s",
                    company_key,
                    persona_err,
                )
                continue

        # Select the highest-scored persona
        selected_persona: dict = {}
        if all_scored:
            all_scored.sort(key=lambda p: p.get("persona_score", 0.0), reverse=True)
            selected_persona = all_scored[0]

        return {
            "selected_persona": selected_persona,
            "all_scored": all_scored,
            "llm_escalations": llm_escalations,
        }

    except Exception as e:
        logger.error("score_personas failed: %s", e)
        return {"error": str(e), "results": [], "count": 0}

"""
Unit tests for the updated app/scorer.py.

Validates:
- score_single_lead signature and return schema are unchanged (Requirement 7.1)
- score_single_lead completes within 10 seconds with mocked tool wrappers (Requirement 7.3)
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure detective root is on path
_detective_root = str(Path(__file__).resolve().parent.parent)
if _detective_root not in sys.path:
    sys.path.insert(0, _detective_root)

from brain.schema import ICPAttributes, Range


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_icp() -> ICPAttributes:
    return ICPAttributes(
        industry=["Software"],
        company_size=Range(min=50, max=500),
        target_countries=["US"],
        target_roles=["VP of Sales"],
    )


def _make_payload() -> dict:
    return {
        "company_data": {
            "name": "Acme Corp",
            "domain": "acme.com",
            "industry": "Software",
            "location": {"country": "US", "city": "New York"},
            "estimated_num_employees": "100-200",
            "website_url": "https://acme.com",
            "linkedin_url": "https://linkedin.com/company/acme",
            "annual_revenue": "10M",
            "founded_year": "2010",
        },
        "enrichment_data": {"data_quality_score": 0.8},
        "personas": [
            {
                "name": "Jane Doe",
                "title": "VP of Sales",
                "email": "jane@acme.com",
                "linkedin_url": "https://linkedin.com/in/janedoe",
            }
        ],
        "intent_signals": {
            "recent_news": ["Acme raises Series B"],
            "job_postings_count": 10,
            "technology_changes": ["Migrated to AWS"],
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Test: return schema is unchanged
# ---------------------------------------------------------------------------

class TestScoreSingleLeadSchema:
    """Verify the return dict always contains all required keys."""

    REQUIRED_KEYS = {
        "final_score",
        "icp_match",
        "filters_passed",
        "similarity_score",
        "intent_boost",
        "selected_persona",
        "qualified_for_outreach",
        "company_data",
    }

    def _mock_tools(self, filtered_result, rank_result, persona_result):
        """Return a context manager that patches all three tool wrappers."""
        import unittest.mock as mock
        return mock.patch.multiple(
            "app.scorer",
            filter_companies_tool=MagicMock(invoke=MagicMock(return_value=filtered_result)),
            rank_companies_tool=MagicMock(invoke=MagicMock(return_value=rank_result)),
            score_personas_tool=MagicMock(invoke=MagicMock(return_value=persona_result)),
        )

    def test_schema_when_company_passes_filters(self):
        """All required keys present when company passes all filters."""
        from app.scorer import score_single_lead

        filtered = {"acme_com": {"name": "Acme Corp"}}
        rank_result = {"results": [{"similarity_score": 0.75}], "count": 1}
        persona_result = {
            "selected_persona": {
                "name": "Jane Doe",
                "job_title": "VP of Sales",
                "persona_score": 0.9,
                "is_sales_dept": False,
                "is_ceo": False,
                "is_target": True,
            },
            "all_scored": [],
            "llm_escalations": 0,
        }

        with self._mock_tools(
            {"results": filtered, "count": 1},
            rank_result,
            persona_result,
        ):
            result = _run(score_single_lead(_make_payload(), _make_icp()))

        assert self.REQUIRED_KEYS == set(result.keys()), (
            f"Missing keys: {self.REQUIRED_KEYS - set(result.keys())}"
        )

    def test_schema_when_company_fails_filters(self):
        """All required keys present even when company is filtered out."""
        from app.scorer import score_single_lead

        with patch("app.scorer.filter_companies_tool") as mock_filter:
            mock_filter.invoke.return_value = {"results": {}, "count": 0}
            result = _run(score_single_lead(_make_payload(), _make_icp()))

        assert self.REQUIRED_KEYS == set(result.keys())
        assert result["icp_match"] is False
        assert result["final_score"] == 0.0

    def test_final_score_is_float_in_range(self):
        """final_score is a float between 0 and 1."""
        from app.scorer import score_single_lead

        filtered = {"acme_com": {"name": "Acme Corp"}}
        rank_result = {"results": [{"similarity_score": 0.8}], "count": 1}
        persona_result = {"selected_persona": {}, "all_scored": [], "llm_escalations": 0}

        with self._mock_tools(
            {"results": filtered, "count": 1},
            rank_result,
            persona_result,
        ):
            result = _run(score_single_lead(_make_payload(), _make_icp()))

        assert isinstance(result["final_score"], float)
        assert 0.0 <= result["final_score"] <= 1.0

    def test_similarity_score_is_float_in_range(self):
        """similarity_score is a float between 0 and 1."""
        from app.scorer import score_single_lead

        filtered = {"acme_com": {"name": "Acme Corp"}}
        rank_result = {"results": [{"similarity_score": 0.65}], "count": 1}
        persona_result = {"selected_persona": {}, "all_scored": [], "llm_escalations": 0}

        with self._mock_tools(
            {"results": filtered, "count": 1},
            rank_result,
            persona_result,
        ):
            result = _run(score_single_lead(_make_payload(), _make_icp()))

        assert isinstance(result["similarity_score"], float)
        assert 0.0 <= result["similarity_score"] <= 1.0

    def test_qualified_for_outreach_is_bool(self):
        """qualified_for_outreach is a boolean."""
        from app.scorer import score_single_lead

        filtered = {"acme_com": {"name": "Acme Corp"}}
        rank_result = {"results": [{"similarity_score": 0.9}], "count": 1}
        persona_result = {"selected_persona": {}, "all_scored": [], "llm_escalations": 0}

        with self._mock_tools(
            {"results": filtered, "count": 1},
            rank_result,
            persona_result,
        ):
            result = _run(score_single_lead(_make_payload(), _make_icp()))

        assert isinstance(result["qualified_for_outreach"], bool)

    def test_filters_passed_is_list(self):
        """filters_passed is always a list."""
        from app.scorer import score_single_lead

        filtered = {"acme_com": {"name": "Acme Corp"}}
        rank_result = {"results": [{"similarity_score": 0.7}], "count": 1}
        persona_result = {"selected_persona": {}, "all_scored": [], "llm_escalations": 0}

        with self._mock_tools(
            {"results": filtered, "count": 1},
            rank_result,
            persona_result,
        ):
            result = _run(score_single_lead(_make_payload(), _make_icp()))

        assert isinstance(result["filters_passed"], list)


# ---------------------------------------------------------------------------
# Test: delegates to tool wrappers
# ---------------------------------------------------------------------------

class TestScoreSingleLeadDelegation:
    """Verify that score_single_lead calls the tool wrappers, not inline classes."""

    def test_filter_companies_tool_is_called(self):
        """filter_companies_tool.invoke is called with companies and icp_attributes."""
        from app.scorer import score_single_lead

        with patch("app.scorer.filter_companies_tool") as mock_filter, \
             patch("app.scorer.rank_companies_tool") as mock_rank, \
             patch("app.scorer.score_personas_tool") as mock_personas:

            mock_filter.invoke.return_value = {"results": {}, "count": 0}
            mock_rank.invoke.return_value = {"results": [], "count": 0}
            mock_personas.invoke.return_value = {"selected_persona": {}, "all_scored": [], "llm_escalations": 0}

            _run(score_single_lead(_make_payload(), _make_icp()))

        mock_filter.invoke.assert_called_once()
        call_kwargs = mock_filter.invoke.call_args[0][0]
        assert "companies" in call_kwargs
        assert "icp_attributes" in call_kwargs

    def test_rank_companies_tool_is_called_when_filter_passes(self):
        """rank_companies_tool.invoke is called when filter returns results."""
        from app.scorer import score_single_lead

        filtered = {"acme_com": {"name": "Acme Corp"}}

        with patch("app.scorer.filter_companies_tool") as mock_filter, \
             patch("app.scorer.rank_companies_tool") as mock_rank, \
             patch("app.scorer.score_personas_tool") as mock_personas:

            mock_filter.invoke.return_value = {"results": filtered, "count": 1}
            mock_rank.invoke.return_value = {"results": [{"similarity_score": 0.7}], "count": 1}
            mock_personas.invoke.return_value = {"selected_persona": {}, "all_scored": [], "llm_escalations": 0}

            _run(score_single_lead(_make_payload(), _make_icp()))

        mock_rank.invoke.assert_called_once()
        call_kwargs = mock_rank.invoke.call_args[0][0]
        assert "companies" in call_kwargs
        assert "icp_text" in call_kwargs

    def test_rank_companies_tool_not_called_when_filter_fails(self):
        """rank_companies_tool.invoke is NOT called when filter returns empty."""
        from app.scorer import score_single_lead

        with patch("app.scorer.filter_companies_tool") as mock_filter, \
             patch("app.scorer.rank_companies_tool") as mock_rank:

            mock_filter.invoke.return_value = {"results": {}, "count": 0}

            _run(score_single_lead(_make_payload(), _make_icp()))

        mock_rank.invoke.assert_not_called()

    def test_score_personas_tool_is_called_when_filter_passes(self):
        """score_personas_tool.invoke is called when filter returns results."""
        from app.scorer import score_single_lead

        filtered = {"acme_com": {"name": "Acme Corp"}}

        with patch("app.scorer.filter_companies_tool") as mock_filter, \
             patch("app.scorer.rank_companies_tool") as mock_rank, \
             patch("app.scorer.score_personas_tool") as mock_personas:

            mock_filter.invoke.return_value = {"results": filtered, "count": 1}
            mock_rank.invoke.return_value = {"results": [{"similarity_score": 0.7}], "count": 1}
            mock_personas.invoke.return_value = {"selected_persona": {}, "all_scored": [], "llm_escalations": 0}

            _run(score_single_lead(_make_payload(), _make_icp()))

        mock_personas.invoke.assert_called_once()
        call_kwargs = mock_personas.invoke.call_args[0][0]
        assert "company_key" in call_kwargs
        assert "company_data" in call_kwargs
        assert "personas" in call_kwargs
        assert "target_roles" in call_kwargs


# ---------------------------------------------------------------------------
# Test: completes within 10 seconds with mocked tools (Requirement 7.3)
# ---------------------------------------------------------------------------

class TestScoreSingleLeadPerformance:
    """Verify score_single_lead completes within 10 seconds with mocked tools."""

    def test_completes_within_10_seconds(self):
        """score_single_lead must complete within 10 seconds when tools are mocked."""
        from app.scorer import score_single_lead

        filtered = {"acme_com": {"name": "Acme Corp"}}
        rank_result = {"results": [{"similarity_score": 0.75}], "count": 1}
        persona_result = {
            "selected_persona": {
                "name": "Jane Doe",
                "persona_score": 0.85,
                "is_sales_dept": False,
                "is_ceo": False,
                "is_target": True,
            },
            "all_scored": [],
            "llm_escalations": 0,
        }

        with patch("app.scorer.filter_companies_tool") as mock_filter, \
             patch("app.scorer.rank_companies_tool") as mock_rank, \
             patch("app.scorer.score_personas_tool") as mock_personas:

            mock_filter.invoke.return_value = {"results": filtered, "count": 1}
            mock_rank.invoke.return_value = rank_result
            mock_personas.invoke.return_value = persona_result

            start = time.monotonic()
            _run(score_single_lead(_make_payload(), _make_icp()))
            elapsed = time.monotonic() - start

        assert elapsed < 10.0, f"score_single_lead took {elapsed:.2f}s (limit: 10s)"


# ---------------------------------------------------------------------------
# Test: fallback persona when score_personas returns empty
# ---------------------------------------------------------------------------

class TestScoreSingleLeadFallback:
    """Verify fallback persona logic when score_personas_tool returns empty."""

    def test_fallback_persona_used_when_score_personas_returns_empty(self):
        """When score_personas returns empty selected_persona, fallback to raw first persona."""
        from app.scorer import score_single_lead

        filtered = {"acme_com": {"name": "Acme Corp"}}

        with patch("app.scorer.filter_companies_tool") as mock_filter, \
             patch("app.scorer.rank_companies_tool") as mock_rank, \
             patch("app.scorer.score_personas_tool") as mock_personas:

            mock_filter.invoke.return_value = {"results": filtered, "count": 1}
            mock_rank.invoke.return_value = {"results": [{"similarity_score": 0.7}], "count": 1}
            # Return empty selected_persona
            mock_personas.invoke.return_value = {"selected_persona": {}, "all_scored": [], "llm_escalations": 0}

            result = _run(score_single_lead(_make_payload(), _make_icp()))

        # Fallback persona should be set from raw personas list
        assert result["selected_persona"] is not None
        assert result["selected_persona"]["persona_score"] == 0.3  # fallback default

    def test_no_persona_when_payload_has_no_personas(self):
        """When payload has no personas and score_personas returns empty, selected_persona is None."""
        from app.scorer import score_single_lead

        payload = _make_payload()
        payload["personas"] = []  # no personas

        filtered = {"acme_com": {"name": "Acme Corp"}}

        with patch("app.scorer.filter_companies_tool") as mock_filter, \
             patch("app.scorer.rank_companies_tool") as mock_rank, \
             patch("app.scorer.score_personas_tool") as mock_personas:

            mock_filter.invoke.return_value = {"results": filtered, "count": 1}
            mock_rank.invoke.return_value = {"results": [{"similarity_score": 0.7}], "count": 1}
            mock_personas.invoke.return_value = {"selected_persona": {}, "all_scored": [], "llm_escalations": 0}

            result = _run(score_single_lead(payload, _make_icp()))

        assert result["selected_persona"] is None

    def test_persona_ranked_added_to_filters_passed(self):
        """'persona_ranked' is added to filters_passed when score_personas returns a persona."""
        from app.scorer import score_single_lead

        filtered = {"acme_com": {"name": "Acme Corp"}}
        persona = {
            "name": "Jane Doe",
            "job_title": "VP of Sales",
            "persona_score": 0.9,
            "is_sales_dept": False,
            "is_ceo": False,
            "is_target": True,
        }

        with patch("app.scorer.filter_companies_tool") as mock_filter, \
             patch("app.scorer.rank_companies_tool") as mock_rank, \
             patch("app.scorer.score_personas_tool") as mock_personas:

            mock_filter.invoke.return_value = {"results": filtered, "count": 1}
            mock_rank.invoke.return_value = {"results": [{"similarity_score": 0.7}], "count": 1}
            mock_personas.invoke.return_value = {"selected_persona": persona, "all_scored": [persona], "llm_escalations": 0}

            result = _run(score_single_lead(_make_payload(), _make_icp()))

        assert "persona_ranked" in result["filters_passed"]

    def test_persona_ranked_not_added_when_score_personas_errors(self):
        """'persona_ranked' is NOT added when score_personas_tool returns an error."""
        from app.scorer import score_single_lead

        filtered = {"acme_com": {"name": "Acme Corp"}}

        with patch("app.scorer.filter_companies_tool") as mock_filter, \
             patch("app.scorer.rank_companies_tool") as mock_rank, \
             patch("app.scorer.score_personas_tool") as mock_personas:

            mock_filter.invoke.return_value = {"results": filtered, "count": 1}
            mock_rank.invoke.return_value = {"results": [{"similarity_score": 0.7}], "count": 1}
            mock_personas.invoke.return_value = {"error": "GROQ_API_KEY not set", "results": [], "count": 0}

            result = _run(score_single_lead(_make_payload(), _make_icp()))

        assert "persona_ranked" not in result["filters_passed"]

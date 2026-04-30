"""
Unit and integration tests for the agentic gate functions.

Tests cover:
  - dqs_calculator.compute_dqs
  - processing_log.make_log_entry
  - main_discovery._merge_ai_result
  - detective_formatter.DetectiveFormatter (backward compat)
  - main_discovery._gate_entity_validation (Gate 1)
  - main_discovery._gate_data_quality (Gate 2)
  - main_discovery._gate_persona_worthiness (Gate 3)
"""

import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch

# ---------------------------------------------------------------------------
# Ensure inject_collect_project root is on sys.path so modules resolve
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ---------------------------------------------------------------------------
# Pre-stub heavy native dependencies so project modules can be imported
# without requiring neo4j, playwright, google-genai, etc. to be installed.
# ---------------------------------------------------------------------------
for _mod in [
    "neo4j", "playwright", "playwright.sync_api",
    "google", "google.genai",
    "fitz", "PIL", "PIL.Image",
    "fake_useragent", "bs4", "dotenv",
    "database_manager", "httpx", "redis",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# ---------------------------------------------------------------------------
# Imports under test (must come after sys.modules patching)
# ---------------------------------------------------------------------------
import asyncio
import json
import os
import tempfile
import pytest
from dqs_calculator import compute_dqs
from processing_log import make_log_entry
from detective_formatter import DetectiveFormatter
import main_discovery
from main_discovery import (
    _gate_entity_validation,
    _merge_ai_result,
    _gate_data_quality,
    _gate_persona_worthiness,
)


# ===========================================================================
# Test Class 1: TestComputeDqs — unit tests for dqs_calculator
# ===========================================================================

class TestComputeDqs:
    def test_empty_dict_returns_zero(self):
        assert compute_dqs({}) == pytest.approx(0.0)

    def test_full_profile_returns_one(self):
        profile = {
            "domain": "acme.com",
            "name": "Acme Corp",
            "industry": "Software",
            "estimated_num_employees": 500,
            "annual_revenue": "$10M",
            "location": {"country": "France"},
            "linkedin_url": "https://linkedin.com/company/acme",
            "website_url": "https://acme.com",
        }
        assert compute_dqs(profile) == pytest.approx(1.0)

    def test_unknown_prefix_domain_contributes_zero(self):
        assert compute_dqs({"domain": "unknown_acme"}) == pytest.approx(0.0)

    def test_non_renseigne_treated_as_empty(self):
        profile = {k: "Non renseigné" for k in [
            "domain", "name", "industry",
            "estimated_num_employees", "annual_revenue", "linkedin_url", "website_url",
        ]}
        profile["location"] = {"country": "Non renseigné"}
        assert compute_dqs(profile) == pytest.approx(0.0)

    def test_none_treated_as_empty(self):
        profile = {k: None for k in [
            "domain", "name", "industry",
            "estimated_num_employees", "annual_revenue", "linkedin_url", "website_url",
        ]}
        profile["location"] = {"country": None}
        assert compute_dqs(profile) == pytest.approx(0.0)

    def test_nested_location_country(self):
        assert compute_dqs({"location": {"country": "France"}}) == pytest.approx(0.10)

    def test_top_level_country_fallback(self):
        assert compute_dqs({"country": "Germany"}) == pytest.approx(0.10)


# ===========================================================================
# Test Class 2: TestMakeLogEntry — unit tests for processing_log
# ===========================================================================

class TestMakeLogEntry:
    def test_base_fields_present(self):
        entry = make_log_entry("data_quality", "proceed_normal", 0.75)
        assert "gate" in entry
        assert "timestamp" in entry
        assert "action" in entry
        assert "dqs_at_gate" in entry

    def test_base_field_values(self):
        entry = make_log_entry("entity_validation", "domain_corrected", 0.0)
        assert entry["gate"] == "entity_validation"
        assert entry["action"] == "domain_corrected"
        assert entry["dqs_at_gate"] == 0.0

    def test_extra_kwargs_merged(self):
        entry = make_log_entry(
            "data_quality", "deep_scrape", 0.3,
            dqs_before=0.3, path_taken="deep_scrape", worker_review_flag=False,
        )
        assert entry["dqs_before"] == 0.3
        assert entry["path_taken"] == "deep_scrape"
        assert entry["worker_review_flag"] is False

    def test_timestamp_is_utc_iso8601(self):
        entry = make_log_entry("test", "test", 0.5)
        ts = entry["timestamp"]
        assert "+00:00" in ts or ts.endswith("Z")


# ===========================================================================
# Test Class 3: TestMergeAiResult — unit tests for _merge_ai_result
# ===========================================================================

class TestMergeAiResult:
    def _write_ai_json(self, tmp_path, ai_data):
        """Write ai_data to a temp JSON file and return the path."""
        p = tmp_path / "ai_result.json"
        p.write_text(json.dumps(ai_data), encoding="utf-8")
        return str(p)

    def test_empty_field_filled_by_ai(self, tmp_path):
        profile = {"name": ""}
        ai_data = {"identity": {"name": {"value": "Acme Corp", "confidence": 0.9, "source": "web"}}}
        path = self._write_ai_json(tmp_path, ai_data)
        _merge_ai_result(profile, path)
        assert profile["name"] == "Acme Corp"

    def test_non_renseigne_field_filled_by_ai(self, tmp_path):
        profile = {"industry": "Non renseigné"}
        ai_data = {"identity": {"industry": {"value": "Software", "confidence": 0.8, "source": "web"}}}
        path = self._write_ai_json(tmp_path, ai_data)
        _merge_ai_result(profile, path)
        assert profile["industry"] == "Software"

    def test_populated_field_not_overwritten(self, tmp_path):
        profile = {"name": "Original Name"}
        ai_data = {"identity": {"name": {"value": "AI Name", "confidence": 0.9, "source": "web"}}}
        path = self._write_ai_json(tmp_path, ai_data)
        _merge_ai_result(profile, path)
        assert profile["name"] == "Original Name"

    def test_ai_none_value_not_merged(self, tmp_path):
        profile = {"name": ""}
        ai_data = {"identity": {"name": {"value": None, "confidence": 0.0, "source": "web"}}}
        path = self._write_ai_json(tmp_path, ai_data)
        _merge_ai_result(profile, path)
        assert profile["name"] == ""

    def test_technologies_list_merged(self, tmp_path):
        profile = {"technologies": []}
        ai_data = {"technologies": [{"name": "React", "category": "Frontend", "confidence": 0.99}]}
        path = self._write_ai_json(tmp_path, ai_data)
        _merge_ai_result(profile, path)
        assert len(profile["technologies"]) == 1
        assert profile["technologies"][0]["name"] == "React"

    def test_technologies_not_overwritten_when_present(self, tmp_path):
        profile = {"technologies": [{"name": "Vue"}]}
        ai_data = {"technologies": [{"name": "React"}]}
        path = self._write_ai_json(tmp_path, ai_data)
        _merge_ai_result(profile, path)
        assert profile["technologies"][0]["name"] == "Vue"

    def test_invalid_json_path_does_not_crash(self):
        profile = {"name": "Test"}
        _merge_ai_result(profile, "/nonexistent/path.json")
        assert profile["name"] == "Test"  # unchanged, no crash


# ===========================================================================
# Test Class 4: TestDetectiveFormatterProcessingLog — backward compat
# ===========================================================================

class TestDetectiveFormatterProcessingLog:
    def _make_profile(self):
        return {"name": "Test", "domain": "test.com", "data_quality_score": 0.5}

    def test_no_processing_log_kwarg_returns_empty_list(self):
        fmt = DetectiveFormatter()
        payload = fmt.format(self._make_profile(), [], {})
        assert payload["processing_log"] == []

    def test_none_processing_log_returns_empty_list(self):
        fmt = DetectiveFormatter()
        payload = fmt.format(self._make_profile(), [], {}, processing_log=None)
        assert payload["processing_log"] == []

    def test_processing_log_list_preserved(self):
        fmt = DetectiveFormatter()
        log = [{"gate": "data_quality", "action": "proceed_normal", "dqs_at_gate": 0.8}]
        payload = fmt.format(self._make_profile(), [], {}, processing_log=log)
        assert payload["processing_log"] == log


# ===========================================================================
# Test Class 5: TestGateEntityValidation — integration tests for Gate 1
# ===========================================================================

class TestGateEntityValidation:
    def test_non_synthetic_domain_skips_gate(self):
        """Gate should return original domain without calling search_news."""
        mock_apify = MagicMock()
        mock_apify.search_news = AsyncMock()
        log = []
        result = asyncio.run(_gate_entity_validation("acme.com", "Acme", mock_apify, log))
        assert result == "acme.com"
        mock_apify.search_news.assert_not_called()
        assert len(log) == 0

    def test_synthetic_domain_with_valid_name_corrects_domain(self):
        """Gate should call search_news and return extracted netloc."""
        mock_apify = MagicMock()
        mock_apify.search_news = AsyncMock(
            return_value=[{"url": "https://acme.com/news/article"}]
        )
        log = []
        result = asyncio.run(_gate_entity_validation("unknown_acme", "Acme Corp", mock_apify, log))
        assert result == "acme.com"
        assert len(log) == 1
        assert log[0]["action"] == "domain_corrected"
        assert log[0]["result"] == "acme.com"

    def test_synthetic_domain_no_results_retains_original(self):
        """Gate should retain synthetic domain when search returns empty."""
        mock_apify = MagicMock()
        mock_apify.search_news = AsyncMock(return_value=[])
        log = []
        result = asyncio.run(_gate_entity_validation("unknown_acme", "Acme Corp", mock_apify, log))
        assert result == "unknown_acme"
        assert len(log) == 1
        assert log[0]["action"] == "correction_failed"

    def test_synthetic_domain_search_exception_retains_original(self):
        """Gate should handle search_news exception gracefully."""
        mock_apify = MagicMock()
        mock_apify.search_news = AsyncMock(side_effect=Exception("API error"))
        log = []
        result = asyncio.run(_gate_entity_validation("unknown_acme", "Acme Corp", mock_apify, log))
        assert result == "unknown_acme"
        assert len(log) == 1
        assert log[0]["action"] == "search_failed"

    def test_unknown_company_name_skips_gate(self):
        """Gate should skip when company_name is 'unknown'."""
        mock_apify = MagicMock()
        mock_apify.search_news = AsyncMock()
        log = []
        result = asyncio.run(_gate_entity_validation("unknown_xyz", "unknown", mock_apify, log))
        assert result == "unknown_xyz"
        mock_apify.search_news.assert_not_called()
        assert len(log) == 0


# ===========================================================================
# Test Class 6: TestGateDataQuality — integration tests for Gate 2
# ===========================================================================

class TestGateDataQuality:
    def _make_low_dqs_profile(self):
        """Profile with DQS < 0.5 (only name present = 0.10)."""
        return {"name": "Acme", "domain": "acme.com"}

    def _make_mid_dqs_profile(self):
        """Profile with DQS in [0.5, 0.75)."""
        return {
            "domain": "acme.com",            # 0.20
            "name": "Acme",                  # 0.10
            "industry": "Tech",              # 0.10
            "estimated_num_employees": 100,  # 0.15
            # total = 0.55
        }

    def _make_high_dqs_profile(self):
        """Profile with DQS >= 0.75."""
        return {
            "domain": "acme.com",            # 0.20
            "name": "Acme",                  # 0.10
            "industry": "Tech",              # 0.10
            "estimated_num_employees": 100,  # 0.15
            "annual_revenue": "$10M",        # 0.15
            "location": {"country": "France"},  # 0.10
            # total = 0.80
        }

    def test_low_dqs_invokes_smart_scraper(self, tmp_path):
        """DQS < 0.5 should call SmartScraperAI."""
        ai_json = tmp_path / "result.json"
        ai_json.write_text(
            json.dumps({
                "identity": {
                    "industry": {"value": "Software", "confidence": 0.9, "source": "web"},
                }
            }),
            encoding="utf-8",
        )

        mock_scraper = MagicMock()
        mock_scraper.scrape_and_save = MagicMock(return_value=str(ai_json))
        log = []
        profile = self._make_low_dqs_profile()
        result = asyncio.run(_gate_data_quality(profile, "acme.com", "France", mock_scraper, log))
        mock_scraper.scrape_and_save.assert_called_once()
        assert len(log) == 1
        assert log[0]["path_taken"] == "deep_scrape"

    def test_low_dqs_synthetic_domain_skips_scraper(self):
        """DQS < 0.5 but synthetic domain should skip SmartScraperAI."""
        mock_scraper = MagicMock()
        log = []
        profile = {"name": "Acme", "domain": "unknown_acme"}
        result = asyncio.run(_gate_data_quality(profile, "unknown_acme", "France", mock_scraper, log))
        mock_scraper.scrape_and_save.assert_not_called()
        assert log[0]["path_taken"] == "proceed_normal"

    def test_mid_dqs_flags_for_review(self):
        """DQS in [0.5, 0.75) should set worker_review_flag=True."""
        mock_scraper = MagicMock()
        log = []
        profile = self._make_mid_dqs_profile()
        asyncio.run(_gate_data_quality(profile, "acme.com", "France", mock_scraper, log))
        mock_scraper.scrape_and_save.assert_not_called()
        assert log[0]["path_taken"] == "flag_for_review"
        assert log[0]["worker_review_flag"] is True

    def test_high_dqs_proceeds_normal(self):
        """DQS >= 0.75 should proceed normally without SmartScraperAI."""
        mock_scraper = MagicMock()
        log = []
        profile = self._make_high_dqs_profile()
        asyncio.run(_gate_data_quality(profile, "acme.com", "France", mock_scraper, log))
        mock_scraper.scrape_and_save.assert_not_called()
        assert log[0]["path_taken"] == "proceed_normal"
        assert log[0]["worker_review_flag"] is False

    def test_scraper_timeout_preserves_original_profile(self):
        """SmartScraperAI timeout should preserve original profile."""
        import asyncio as _asyncio

        log = []
        profile = self._make_low_dqs_profile().copy()
        original_name = profile["name"]

        # Patch asyncio.wait_for to raise TimeoutError
        with patch("main_discovery.asyncio.wait_for", side_effect=_asyncio.TimeoutError()):
            result = asyncio.run(
                _gate_data_quality(profile, "acme.com", "France", MagicMock(), log)
            )

        assert result["name"] == original_name
        assert any(e.get("action") == "scraper_timeout" for e in log)

    def test_scraper_exception_preserves_original_profile(self):
        """SmartScraperAI exception should preserve original profile."""
        log = []
        profile = self._make_low_dqs_profile().copy()
        original_name = profile["name"]

        with patch("main_discovery.asyncio.wait_for", side_effect=RuntimeError("scraper crashed")):
            result = asyncio.run(
                _gate_data_quality(profile, "acme.com", "France", MagicMock(), log)
            )

        assert result["name"] == original_name
        assert any(e.get("action") in ("scraper_empty", "gate_error") for e in log)


# ===========================================================================
# Test Class 7: TestGatePersonaWorthiness — integration tests for Gate 3
# ===========================================================================

class TestGatePersonaWorthiness:
    def test_low_dqs_returns_false(self):
        log = []
        result = _gate_persona_worthiness({}, {}, 0.3, log)
        assert result is False
        assert log[0]["decision"] == "skip_personas"

    def test_high_dqs_no_signals_returns_false(self):
        log = []
        intent = {"job_postings_count": 0, "recent_news": []}
        profile = {"estimated_num_employees": "Non renseigné"}
        result = _gate_persona_worthiness(profile, intent, 0.8, log)
        assert result is False
        assert log[0]["decision"] == "skip_personas"

    def test_high_dqs_with_job_postings_returns_true(self):
        log = []
        intent = {"job_postings_count": 5, "recent_news": []}
        result = _gate_persona_worthiness({}, intent, 0.8, log)
        assert result is True
        assert log[0]["decision"] == "run_personas"

    def test_high_dqs_with_news_returns_true(self):
        log = []
        intent = {"job_postings_count": 0, "recent_news": [{"title": "Acme raises Series B"}]}
        result = _gate_persona_worthiness({}, intent, 0.6, log)
        assert result is True

    def test_high_dqs_with_employee_count_returns_true(self):
        log = []
        intent = {"job_postings_count": 0, "recent_news": []}
        profile = {"estimated_num_employees": 500}
        result = _gate_persona_worthiness(profile, intent, 0.6, log)
        assert result is True

    def test_log_entry_contains_required_fields(self):
        log = []
        intent = {"job_postings_count": 3, "recent_news": [{"title": "news"}]}
        _gate_persona_worthiness({}, intent, 0.7, log)
        entry = log[0]
        assert "gate" in entry
        assert "dqs" in entry
        assert "job_postings_count" in entry
        assert "has_news" in entry
        assert "has_employee_count" in entry
        assert "decision" in entry

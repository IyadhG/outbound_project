"""
Bug Condition Exploration Tests
================================
These tests PASS on UNFIXED code (asserting buggy behavior exists).
They will FAIL after the fix is applied — that is the intended design.

Requirements covered: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

import asyncio
import os
import sys
import time
import importlib
import inspect
import unittest
from unittest.mock import patch, MagicMock, AsyncMock, call

# Ensure inject_collect_project is on the path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ---------------------------------------------------------------------------
# Pre-stub heavy dependencies so main_discovery can be imported without
# requiring neo4j, playwright, google-genai, etc. to be installed.
# We do this at module level so all test classes share the same stubs.
# ---------------------------------------------------------------------------
_neo4j_stub = MagicMock()
_smart_scraper_stub = MagicMock()
_persona_stub = MagicMock()
_requests_stub = MagicMock()

# Stub out modules that have hard C/native dependencies
for _mod_name in [
    "neo4j", "neo4j.GraphDatabase",
    "smart_scraper_ai",
    "persona_search_enrich",
    "playwright", "playwright.sync_api",
    "google", "google.genai",
    "fitz", "PIL", "PIL.Image",
    "fake_useragent",
    "bs4",
    "dotenv",
    "httpx",
    "redis",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# Stub database_manager so Neo4jManager doesn't try to connect
_db_manager_stub = MagicMock()
sys.modules["database_manager"] = _db_manager_stub

# Now import main_discovery — it will use the stubs above
import main_discovery  # noqa: E402  (must come after sys.modules patching)


# ---------------------------------------------------------------------------
# Test 1 — Dead code: agentic_intent/ directory must exist (unfixed)
# Validates: Requirement 1.1
# ---------------------------------------------------------------------------
class TestDeadCodePresent(unittest.TestCase):
    """
    Bug condition: isDeadCodePresent(project)
    The agentic_intent/ folder is dead code that should be removed.
    This test PASSES on unfixed code (directory exists).
    It MUST FAIL after the fix (directory deleted).
    """

    def test_agentic_intent_directory_exists(self):
        """Assert that the agentic_intent/ dead-code directory still exists."""
        agentic_intent_path = os.path.join(PROJECT_DIR, "agentic_intent")
        self.assertTrue(
            os.path.isdir(agentic_intent_path),
            f"Expected dead-code directory to exist at {agentic_intent_path}, but it was not found. "
            "This means the fix has already been applied (directory deleted)."
        )


# ---------------------------------------------------------------------------
# Test 2 — Playwright import: SmartScraperAI must be importable (unfixed)
# Validates: Requirement 1.2
# ---------------------------------------------------------------------------
class TestPlaywrightImport(unittest.TestCase):
    """
    Bug condition: isPlaywrightScraping(enrichmentCall)
    main_discovery.py imports SmartScraperAI (Playwright/Gemini scraper).
    This test PASSES on unfixed code (import succeeds or source contains the symbol).
    It MUST FAIL after the fix (import removed).
    """

    def test_smart_scraper_ai_referenced_in_main_discovery(self):
        """
        Assert that SmartScraperAI is importable from main_discovery OR
        that the source of main_discovery.py contains the string 'SmartScraperAI'.
        """
        main_discovery_path = os.path.join(PROJECT_DIR, "main_discovery.py")
        self.assertTrue(
            os.path.exists(main_discovery_path),
            "main_discovery.py not found"
        )

        with open(main_discovery_path, "r", encoding="utf-8") as f:
            source = f.read()

        # On unfixed code, SmartScraperAI is imported and used in main_discovery.py
        self.assertIn(
            "SmartScraperAI",
            source,
            "Expected 'SmartScraperAI' to appear in main_discovery.py source (Playwright scraper still present). "
            "This means the fix has already been applied (import removed)."
        )


# ---------------------------------------------------------------------------
# Shared mock Apollo response used by tests 3–6
# ---------------------------------------------------------------------------
MOCK_APOLLO_SEARCH_RESULT = [
    {
        "apollo_id": "test-id-001",
        "domain": "testcompany.com",
        "name": "Test Company",
        "industry": "Software",
        "founded_year": "2010",
        "logo_url": "https://testcompany.com/logo.png",
        "website_url": "https://testcompany.com",
        "short_description": "A test company",
        "seo_description": "Test company SEO",
        "alexa_ranking": "50000",
        "annual_revenue": "$10M",
        "total_funding": "$5M",
        "estimated_num_employees": "100",
        "latest_funding_stage": "Series A",
        "linkedin_url": "https://linkedin.com/company/testcompany",
        "twitter_url": "Non renseigné",
        "facebook_url": "Non renseigné",
        "crunchbase_url": "Non renseigné",
        "phone": "Non renseigné",
        "location": {
            "raw_address": "123 Test St, Paris, France",
            "street_address": "123 Test St",
            "city": "Paris",
            "state": "Ile-de-France",
            "postal_code": "75001",
            "country": "France",
        },
        "hierarchy": {
            "num_suborganizations": 0,
            "owned_by_organization_id": "Non renseigné",
        },
    }
]

MOCK_APOLLO_ENRICH_RESULT = MOCK_APOLLO_SEARCH_RESULT[0]


def _run_pipeline_with_mocks(extra_patches=None, delay_fn=None):
    """
    Helper: run discover_and_inject with all external dependencies mocked.
    Returns the list of injected domains.

    extra_patches: dict of {target_string: mock_object} applied on top of defaults.
    delay_fn: if provided, replaces the enrich_organization side_effect to add delay.
    """
    patches = {}

    # Mock ApolloScraper
    mock_scraper_instance = MagicMock()
    mock_scraper_instance.search_companies.return_value = MOCK_APOLLO_SEARCH_RESULT
    if delay_fn:
        mock_scraper_instance.enrich_organization.side_effect = delay_fn
    else:
        mock_scraper_instance.enrich_organization.return_value = MOCK_APOLLO_ENRICH_RESULT
    patches["main_discovery.ApolloScraper"] = MagicMock(return_value=mock_scraper_instance)

    # Mock Neo4jManager
    mock_db_instance = MagicMock()
    mock_db_instance.bulk_import_companies.return_value = None
    mock_db_instance.import_merged_profiles.return_value = None
    mock_db_instance.import_personas.return_value = None
    patches["main_discovery.Neo4jManager"] = MagicMock(return_value=mock_db_instance)

    # Mock SmartScraperAI — make scrape_and_save return None so merged path is skipped
    mock_ai_instance = MagicMock()
    mock_ai_instance.scrape_and_save.return_value = None
    patches["main_discovery.SmartScraperAI"] = MagicMock(return_value=mock_ai_instance)

    # Mock persona search
    patches["main_discovery.search_and_enrich"] = MagicMock(return_value=[])

    if extra_patches:
        patches.update(extra_patches)

    with patch.multiple("main_discovery", **{k.split(".")[-1]: v for k, v in patches.items()}):
        # Re-import to pick up patches
        import main_discovery
        result = main_discovery.discover_and_inject(
            industry="Software",
            location="France",
            limit=1,
        )
    return result, mock_db_instance, mock_ai_instance, mock_scraper_instance


# ---------------------------------------------------------------------------
# Test 3 — Missing intent signals: output must have NO intent_signals key
# Validates: Requirement 1.3
# ---------------------------------------------------------------------------
class TestMissingIntentSignals(unittest.TestCase):
    """
    Bug condition: isMissingIntentSignals(profile)
    The pipeline never collects intent signals.
    This test PASSES on unfixed code (no intent_signals in output).
    It MUST FAIL after the fix (intent_signals present).
    """

    def test_pipeline_output_has_no_intent_signals(self):
        """
        Run the pipeline for one company with mocked Apollo response.
        Assert the returned output has no intent_signals.
        On unfixed code, discover_and_inject returns a list of domain strings (no intent_signals).
        On fixed code, it returns Detective payloads with intent_signals — so this test FAILS.
        """
        mock_scraper = MagicMock()
        mock_scraper.search_companies.return_value = MOCK_APOLLO_SEARCH_RESULT
        mock_scraper.enrich_organization.return_value = MOCK_APOLLO_ENRICH_RESULT

        mock_db = MagicMock()

        mock_apify = MagicMock()
        mock_apify.crawl_website = AsyncMock(return_value={})
        mock_apify.search_news = AsyncMock(return_value=[])

        mock_intent = MagicMock()
        mock_intent.collect = AsyncMock(return_value={"recent_news": [], "job_postings_count": 0, "technology_changes": []})

        mock_event_emitter = MagicMock()
        mock_event_emitter.emit_lead_ingested = AsyncMock(return_value=None)

        with patch.object(main_discovery, "ApolloScraper", return_value=mock_scraper), \
             patch.object(main_discovery, "Neo4jManager", return_value=mock_db), \
             patch.object(main_discovery, "ApifyEnricher", return_value=mock_apify), \
             patch.object(main_discovery, "IntentCollector", return_value=mock_intent), \
             patch.object(main_discovery, "EventEmitter", return_value=mock_event_emitter), \
             patch.object(main_discovery, "search_and_enrich", return_value=[]):
            result = asyncio.run(main_discovery.discover_and_inject(industry="Software", location="France", limit=1))

        # On unfixed code, result is a list of domain strings — no intent_signals anywhere
        # On fixed code, result is a list of Detective payload dicts with intent_signals
        self.assertIsInstance(result, list, "Result should be a list.")
        self.assertGreater(len(result), 0, "No results returned — pipeline may not have run.")

        for item in result:
            # On unfixed code, items are domain strings — asserting no intent_signals
            self.assertNotIn(
                "intent_signals",
                item if isinstance(item, dict) else {},
                f"Found 'intent_signals' in result item — fix has already been applied. "
                f"Item keys: {list(item.keys()) if isinstance(item, dict) else item}"
            )
            # Also assert item is a string (unfixed code returns domain strings)
            self.assertIsInstance(
                item,
                str,
                f"Expected domain string in result, got {type(item)}: {item}. "
                "This suggests the pipeline now returns Detective payloads (fix applied)."
            )


# ---------------------------------------------------------------------------
# Test 4 — Non-Detective output: output must be MISSING Detective schema keys
# Validates: Requirement 1.4
# ---------------------------------------------------------------------------
class TestNonDetectiveOutput(unittest.TestCase):
    """
    Bug condition: isNotDetectiveReady(output)
    The pipeline never produces correlation_id, readiness_flags, event_type, timestamp.
    This test PASSES on unfixed code (keys absent).
    It MUST FAIL after the fix (all keys present).
    """

    DETECTIVE_KEYS = ["correlation_id", "readiness_flags", "event_type", "timestamp"]

    def test_pipeline_output_missing_detective_schema_keys(self):
        """
        Run the pipeline, check the return value.
        On unfixed code, result is a list of domain strings — no Detective keys.
        On fixed code, result is a list of Detective payloads with all required keys — so this test FAILS.
        """
        mock_scraper = MagicMock()
        mock_scraper.search_companies.return_value = MOCK_APOLLO_SEARCH_RESULT
        mock_scraper.enrich_organization.return_value = MOCK_APOLLO_ENRICH_RESULT

        mock_db = MagicMock()

        mock_apify = MagicMock()
        mock_apify.crawl_website = AsyncMock(return_value={})
        mock_apify.search_news = AsyncMock(return_value=[])

        mock_intent = MagicMock()
        mock_intent.collect = AsyncMock(return_value={"recent_news": [], "job_postings_count": 0, "technology_changes": []})

        mock_event_emitter = MagicMock()
        mock_event_emitter.emit_lead_ingested = AsyncMock(return_value=None)

        with patch.object(main_discovery, "ApolloScraper", return_value=mock_scraper), \
             patch.object(main_discovery, "Neo4jManager", return_value=mock_db), \
             patch.object(main_discovery, "ApifyEnricher", return_value=mock_apify), \
             patch.object(main_discovery, "IntentCollector", return_value=mock_intent), \
             patch.object(main_discovery, "EventEmitter", return_value=mock_event_emitter), \
             patch.object(main_discovery, "search_and_enrich", return_value=[]):
            result = asyncio.run(main_discovery.discover_and_inject(industry="Software", location="France", limit=1))

        self.assertIsInstance(result, list, "Result should be a list.")
        self.assertGreater(len(result), 0, "No results returned.")

        for item in result:
            item_dict = item if isinstance(item, dict) else {}
            for key in self.DETECTIVE_KEYS:
                self.assertNotIn(
                    key,
                    item_dict,
                    f"Found Detective key '{key}' in result — fix has already been applied. "
                    f"Item keys: {list(item_dict.keys())}"
                )


# ---------------------------------------------------------------------------
# Test 5 — Sequential processing: 3 companies with 100ms delay must take > 250ms
# Validates: Requirement 1.5
# ---------------------------------------------------------------------------
class TestSequentialProcessing(unittest.TestCase):
    """
    Bug condition: isSequentialProcessing(pipeline)
    The pipeline uses a sequential for-loop, so N companies take N * delay.
    This test PASSES on unfixed code (total time > 250ms for 3 x 100ms companies).
    It MUST FAIL after the fix (async gather reduces wall-clock time).
    """

    DELAY_PER_COMPANY = 0.1   # 100ms
    NUM_COMPANIES = 3
    MIN_EXPECTED_TOTAL = 0.25  # 250ms — proves sequential execution

    def _make_three_companies(self):
        companies = []
        for i in range(self.NUM_COMPANIES):
            c = dict(MOCK_APOLLO_SEARCH_RESULT[0])
            c["domain"] = f"company{i}.com"
            c["name"] = f"Company {i}"
            c["website_url"] = f"https://company{i}.com"
            companies.append(c)
        return companies

    def test_three_companies_take_more_than_250ms(self):
        """
        Mock 3 companies each with a 100ms artificial delay in enrich_organization.
        Assert total wall-clock time > 250ms (proves sequential execution).
        """
        three_companies = self._make_three_companies()

        def slow_enrich(domain=None, target_location=None):
            time.sleep(self.DELAY_PER_COMPANY)
            for c in three_companies:
                if c["domain"] == domain:
                    return c
            return three_companies[0]

        mock_scraper = MagicMock()
        mock_scraper.search_companies.return_value = three_companies
        mock_scraper.enrich_organization.side_effect = slow_enrich

        mock_db = MagicMock()

        mock_apify = MagicMock()
        mock_apify.crawl_website = AsyncMock(return_value={})
        mock_apify.search_news = AsyncMock(return_value=[])

        mock_intent = MagicMock()
        mock_intent.collect = AsyncMock(return_value={"recent_news": [], "job_postings_count": 0, "technology_changes": []})

        mock_event_emitter = MagicMock()
        mock_event_emitter.emit_lead_ingested = AsyncMock(return_value=None)

        with patch.object(main_discovery, "ApolloScraper", return_value=mock_scraper), \
             patch.object(main_discovery, "Neo4jManager", return_value=mock_db), \
             patch.object(main_discovery, "ApifyEnricher", return_value=mock_apify), \
             patch.object(main_discovery, "IntentCollector", return_value=mock_intent), \
             patch.object(main_discovery, "EventEmitter", return_value=mock_event_emitter), \
             patch.object(main_discovery, "search_and_enrich", return_value=[]):
            start = time.monotonic()
            asyncio.run(main_discovery.discover_and_inject(industry="Software", location="France", limit=3))
            elapsed = time.monotonic() - start

        self.assertGreater(
            elapsed,
            self.MIN_EXPECTED_TOTAL,
            f"Total elapsed time {elapsed:.3f}s is NOT > {self.MIN_EXPECTED_TOTAL}s. "
            "This means the pipeline is already running concurrently (fix applied)."
        )


# ---------------------------------------------------------------------------
# Test 6 — No event emission: no Redis publish and events_emitted is empty/absent
# Validates: Requirement 1.6
# ---------------------------------------------------------------------------
class TestNoEventEmission(unittest.TestCase):
    """
    Bug condition: isEventEmissionMissing(ingestResult)
    The pipeline never emits lead_ingested events.
    This test PASSES on unfixed code (no Redis publish, no events_emitted).
    It MUST FAIL after the fix (lead_ingested event emitted).
    """

    def test_no_redis_publish_and_no_events_emitted(self):
        """
        Run the pipeline. Assert:
        1. No Redis publish was called (redis.publish / redis.asyncio publish not invoked).
        2. The return value has no 'events_emitted' key, or it is empty.
        """
        mock_scraper = MagicMock()
        mock_scraper.search_companies.return_value = MOCK_APOLLO_SEARCH_RESULT
        mock_scraper.enrich_organization.return_value = MOCK_APOLLO_ENRICH_RESULT

        mock_db = MagicMock()

        mock_apify = MagicMock()
        mock_apify.crawl_website = AsyncMock(return_value={})
        mock_apify.search_news = AsyncMock(return_value=[])

        mock_intent = MagicMock()
        mock_intent.collect = AsyncMock(return_value={"recent_news": [], "job_postings_count": 0, "technology_changes": []})

        mock_event_emitter = MagicMock()
        mock_event_emitter.emit_lead_ingested = AsyncMock(return_value=None)

        mock_redis = MagicMock()

        with patch.object(main_discovery, "ApolloScraper", return_value=mock_scraper), \
             patch.object(main_discovery, "Neo4jManager", return_value=mock_db), \
             patch.object(main_discovery, "ApifyEnricher", return_value=mock_apify), \
             patch.object(main_discovery, "IntentCollector", return_value=mock_intent), \
             patch.object(main_discovery, "EventEmitter", return_value=mock_event_emitter), \
             patch.object(main_discovery, "search_and_enrich", return_value=[]), \
             patch.dict("sys.modules", {"redis": mock_redis, "redis.asyncio": mock_redis}):
            result = asyncio.run(main_discovery.discover_and_inject(industry="Software", location="France", limit=1))

        # Assert no Redis publish was called
        self.assertFalse(
            mock_redis.publish.called,
            "Redis publish was called — event emission has been implemented (fix applied)."
        )

        # Assert result is a plain list of domains (no events_emitted key)
        # On unfixed code, discover_and_inject returns a list of domain strings
        if isinstance(result, dict):
            events = result.get("events_emitted", [])
            self.assertEqual(
                len(events),
                0,
                f"events_emitted is non-empty ({events}) — fix has already been applied."
            )
        else:
            # result is a list of domain strings — no events_emitted at all
            self.assertIsInstance(
                result,
                list,
                "Expected result to be a list of domain strings on unfixed code."
            )
            # Confirm it does NOT contain any event payload dicts
            for item in result:
                self.assertIsInstance(
                    item,
                    str,
                    f"Expected domain string in result, got {type(item)}: {item}. "
                    "This suggests the pipeline now returns event payloads (fix applied)."
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
Preservation Property Tests
============================
These tests PASS on UNFIXED code — they confirm the baseline behaviors that
must be preserved after the fix is applied.

They MUST ALSO PASS after the fix (proving no regressions).

Requirements covered: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call

# ---------------------------------------------------------------------------
# Ensure inject_collect_project is on the path
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ---------------------------------------------------------------------------
# Pre-stub heavy native dependencies (same pattern as test_bug_condition_exploration.py)
# ---------------------------------------------------------------------------
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
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# Stub database_manager so Neo4jManager doesn't try to connect
sys.modules["database_manager"] = MagicMock()

# Now import main_discovery — it will use the stubs above
import main_discovery  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_apollo_company(domain="alten.ch", name="Alten Switzerland", country="Switzerland"):
    """Return a minimal Apollo-formatted company dict."""
    return {
        "apollo_id": f"apollo-{domain}",
        "domain": domain,
        "name": name,
        "industry": "IT Services",
        "founded_year": "2000",
        "logo_url": f"https://{domain}/logo.png",
        "website_url": f"https://{domain}",
        "short_description": f"{name} description",
        "seo_description": f"{name} SEO",
        "alexa_ranking": "100000",
        "annual_revenue": "$50M",
        "total_funding": "Non renseigné",
        "estimated_num_employees": "500",
        "latest_funding_stage": "Non renseigné",
        "linkedin_url": f"https://linkedin.com/company/{domain.split('.')[0]}",
        "twitter_url": "Non renseigné",
        "facebook_url": "Non renseigné",
        "crunchbase_url": "Non renseigné",
        "phone": "Non renseigné",
        "location": {
            "raw_address": f"1 Main St, Zurich, {country}",
            "street_address": "1 Main St",
            "city": "Zurich",
            "state": "Zurich",
            "postal_code": "8001",
            "country": country,
        },
        "hierarchy": {
            "num_suborganizations": 0,
            "owned_by_organization_id": "Non renseigné",
        },
        "technologies": [],
        "departments": {},
        "funding_events": [],
        "suborganizations": [],
        "keywords": [],
    }


# ---------------------------------------------------------------------------
# Test 1 — Apollo call preservation
# Validates: Requirement 3.1
#
# For all (domain, location) pairs where none of the six bug conditions hold,
# ApolloScraper.enrich_organization is called with the same arguments.
# ---------------------------------------------------------------------------
class TestApolloCallPreservation(unittest.TestCase):
    """
    Preservation property: Apollo enrichment call signature is unchanged.
    For every (domain, location) pair, enrich_organization must be called
    with domain=<domain> and target_location=<location>.

    Validates: Requirement 3.1
    """

    DOMAIN_LOCATION_PAIRS = [
        ("alten.ch", "Switzerland"),
        ("alten.es", "Spain"),
        ("aubay.pt", "Portugal"),
        ("manpower.fr", "France"),
        ("bosch.com", "Germany"),
    ]

    def _run_pipeline_for(self, domain, location):
        """Run the pipeline for a single company and return the mock scraper."""
        import asyncio

        company = _make_apollo_company(domain=domain, country=location)

        mock_scraper = MagicMock()
        mock_scraper.search_companies.return_value = [company]
        mock_scraper.enrich_organization.return_value = company

        mock_db = MagicMock()

        async def _fake_crawl(d):
            return {}

        async def _fake_collect(d, n):
            return {"recent_news": [], "job_postings_count": 0, "technology_changes": []}

        async def _fake_emit(p):
            return None

        mock_apify = MagicMock()
        mock_apify.crawl_website = _fake_crawl
        mock_intent = MagicMock()
        mock_intent.collect = _fake_collect
        mock_detective = MagicMock()
        mock_detective.format.return_value = {}
        mock_emitter = MagicMock()
        mock_emitter.emit_lead_ingested = _fake_emit

        with patch.object(main_discovery, "ApolloScraper", return_value=mock_scraper), \
             patch.object(main_discovery, "Neo4jManager", return_value=mock_db), \
             patch.object(main_discovery, "ApifyEnricher", return_value=mock_apify), \
             patch.object(main_discovery, "IntentCollector", return_value=mock_intent), \
             patch.object(main_discovery, "DetectiveFormatter", return_value=mock_detective), \
             patch.object(main_discovery, "EventEmitter", return_value=mock_emitter), \
             patch.object(main_discovery, "search_and_enrich", return_value=[]):
            asyncio.run(main_discovery.discover_and_inject(
                industry="IT",
                location=location,
                limit=1,
            ))

        return mock_scraper

    def test_enrich_organization_called_with_correct_domain_and_location(self):
        """
        For each (domain, location) pair, enrich_organization must be called
        with keyword arguments domain=<domain> and target_location=<location>.
        """
        for domain, location in self.DOMAIN_LOCATION_PAIRS:
            with self.subTest(domain=domain, location=location):
                mock_scraper = self._run_pipeline_for(domain, location)

                self.assertTrue(
                    mock_scraper.enrich_organization.called,
                    f"enrich_organization was NOT called for domain={domain}, location={location}."
                )

                # Inspect the actual call arguments
                call_args = mock_scraper.enrich_organization.call_args
                kwargs = call_args.kwargs if call_args.kwargs else {}
                args = call_args.args if call_args.args else ()

                # domain can be positional or keyword
                called_domain = kwargs.get("domain") or (args[0] if args else None)
                called_location = kwargs.get("target_location") or (args[1] if len(args) > 1 else None)

                self.assertEqual(
                    called_domain,
                    domain,
                    f"enrich_organization called with domain={called_domain!r}, expected {domain!r}."
                )
                self.assertEqual(
                    called_location,
                    location,
                    f"enrich_organization called with target_location={called_location!r}, expected {location!r}."
                )


# ---------------------------------------------------------------------------
# Test 2 — Neo4j write preservation
# Validates: Requirement 3.3
#
# For all random merged profiles, import_merged_profiles and import_personas
# receive the same data shapes (key sets).
# ---------------------------------------------------------------------------
class TestNeo4jWritePreservation(unittest.TestCase):
    """
    Preservation property: Neo4j write functions receive consistent data shapes.
    bulk_import_companies and import_merged_profiles must be called with dicts
    that contain the expected key sets.

    Validates: Requirement 3.3
    """

    # Minimum keys that must be present in any company dict passed to Neo4j
    REQUIRED_COMPANY_KEYS = {"domain", "name"}

    # Keys that the Apollo scraper always produces (from _format_org_data)
    APOLLO_KEYS = {
        "apollo_id", "domain", "name", "industry", "founded_year",
        "logo_url", "website_url", "short_description", "seo_description",
        "alexa_ranking", "annual_revenue", "total_funding", "estimated_num_employees",
        "latest_funding_stage", "linkedin_url", "twitter_url", "facebook_url",
        "crunchbase_url", "phone", "location", "hierarchy",
    }

    SAMPLE_COMPANIES = [
        _make_apollo_company("alten.ch", "Alten Switzerland", "Switzerland"),
        _make_apollo_company("manpower.fr", "Manpower France", "France"),
        _make_apollo_company("bosch.com", "Bosch", "Germany"),
    ]

    def test_bulk_import_companies_receives_correct_shape(self):
        """
        When import_merged_profiles raises an exception, bulk_import_companies
        must be called with dicts containing at least the required company keys.
        """
        import asyncio

        for company in self.SAMPLE_COMPANIES:
            with self.subTest(domain=company["domain"]):
                captured = []

                mock_scraper = MagicMock()
                mock_scraper.search_companies.return_value = [company]
                mock_scraper.enrich_organization.return_value = company

                mock_db = MagicMock()
                # Force import_merged_profiles to fail so bulk_import_companies is called
                mock_db.import_merged_profiles.side_effect = Exception("forced failure")
                mock_db.bulk_import_companies.side_effect = lambda cs: captured.extend(cs)

                async def _fake_crawl(d):
                    return {}

                async def _fake_collect(d, n):
                    return {"recent_news": [], "job_postings_count": 0, "technology_changes": []}

                async def _fake_emit(p):
                    return None

                mock_apify = MagicMock()
                mock_apify.crawl_website = _fake_crawl
                mock_intent = MagicMock()
                mock_intent.collect = _fake_collect
                mock_detective = MagicMock()
                mock_detective.format.return_value = {}
                mock_emitter = MagicMock()
                mock_emitter.emit_lead_ingested = _fake_emit

                with patch.object(main_discovery, "ApolloScraper", return_value=mock_scraper), \
                     patch.object(main_discovery, "Neo4jManager", return_value=mock_db), \
                     patch.object(main_discovery, "ApifyEnricher", return_value=mock_apify), \
                     patch.object(main_discovery, "IntentCollector", return_value=mock_intent), \
                     patch.object(main_discovery, "DetectiveFormatter", return_value=mock_detective), \
                     patch.object(main_discovery, "EventEmitter", return_value=mock_emitter), \
                     patch.object(main_discovery, "search_and_enrich", return_value=[]):
                    asyncio.run(main_discovery.discover_and_inject(industry="IT", location="France", limit=1))

                self.assertGreater(len(captured), 0, "bulk_import_companies was not called.")

                for written in captured:
                    missing = self.REQUIRED_COMPANY_KEYS - set(written.keys())
                    self.assertEqual(
                        missing,
                        set(),
                        f"bulk_import_companies dict missing required keys: {missing}. "
                        f"Got keys: {set(written.keys())}"
                    )

    def test_import_merged_profiles_receives_correct_shape(self):
        """
        When the pipeline runs normally, import_merged_profiles must be called
        with dicts containing at least the required company keys.
        """
        import asyncio

        company = _make_apollo_company("alten.ch", "Alten Switzerland", "Switzerland")

        captured = []

        async def _fake_crawl(d):
            return {}

        async def _fake_collect(d, n):
            return {"recent_news": [], "job_postings_count": 0, "technology_changes": []}

        async def _fake_emit(p):
            return None

        mock_scraper = MagicMock()
        mock_scraper.search_companies.return_value = [company]
        mock_scraper.enrich_organization.return_value = company

        mock_db = MagicMock()
        mock_db.import_merged_profiles.side_effect = lambda ps: captured.extend(ps)

        mock_apify = MagicMock()
        mock_apify.crawl_website = _fake_crawl
        mock_intent = MagicMock()
        mock_intent.collect = _fake_collect
        mock_detective = MagicMock()
        mock_detective.format.return_value = {}
        mock_emitter = MagicMock()
        mock_emitter.emit_lead_ingested = _fake_emit

        with patch.object(main_discovery, "ApolloScraper", return_value=mock_scraper), \
             patch.object(main_discovery, "Neo4jManager", return_value=mock_db), \
             patch.object(main_discovery, "ApifyEnricher", return_value=mock_apify), \
             patch.object(main_discovery, "IntentCollector", return_value=mock_intent), \
             patch.object(main_discovery, "DetectiveFormatter", return_value=mock_detective), \
             patch.object(main_discovery, "EventEmitter", return_value=mock_emitter), \
             patch.object(main_discovery, "search_and_enrich", return_value=[]):
            asyncio.run(main_discovery.discover_and_inject(industry="IT", location="Switzerland", limit=1))

        self.assertGreater(len(captured), 0, "import_merged_profiles was not called.")

        for written in captured:
            missing = self.REQUIRED_COMPANY_KEYS - set(written.keys())
            self.assertEqual(
                missing,
                set(),
                f"import_merged_profiles dict missing required keys: {missing}. "
                f"Got keys: {set(written.keys())}"
            )


# ---------------------------------------------------------------------------
# Test 3 — Persona cascade preservation
# Validates: Requirement 3.5
#
# For all (domain, location, role) triples, search_and_enrich is called
# with the expected positional/keyword arguments.
# ---------------------------------------------------------------------------
class TestPersonaCascadePreservation(unittest.TestCase):
    """
    Preservation property: search_and_enrich is called with the correct
    domain, location, and role arguments.

    Validates: Requirement 3.5
    """

    TRIPLES = [
        ("alten.ch", "Switzerland", "Sales"),
        ("manpower.fr", "France", "Sales"),
        ("bosch.com", "Germany", "Sales"),
    ]

    def test_search_and_enrich_called_with_correct_args(self):
        """
        For each (domain, location, role) triple, search_and_enrich must be
        called with domain=<domain>, location=<location>, role="Sales".
        """
        import asyncio

        for domain, location, role in self.TRIPLES:
            with self.subTest(domain=domain, location=location):
                company = _make_apollo_company(domain=domain, country=location)

                mock_scraper = MagicMock()
                mock_scraper.search_companies.return_value = [company]
                mock_scraper.enrich_organization.return_value = company

                mock_db = MagicMock()

                async def _fake_crawl(d):
                    return {}

                async def _fake_collect(d, n):
                    return {"recent_news": [], "job_postings_count": 0, "technology_changes": []}

                async def _fake_emit(p):
                    return None

                mock_apify = MagicMock()
                mock_apify.crawl_website = _fake_crawl
                mock_intent = MagicMock()
                mock_intent.collect = _fake_collect
                mock_detective = MagicMock()
                mock_detective.format.return_value = {}
                mock_emitter = MagicMock()
                mock_emitter.emit_lead_ingested = _fake_emit

                mock_search_and_enrich = MagicMock(return_value=[])

                with patch.object(main_discovery, "ApolloScraper", return_value=mock_scraper), \
                     patch.object(main_discovery, "Neo4jManager", return_value=mock_db), \
                     patch.object(main_discovery, "ApifyEnricher", return_value=mock_apify), \
                     patch.object(main_discovery, "IntentCollector", return_value=mock_intent), \
                     patch.object(main_discovery, "DetectiveFormatter", return_value=mock_detective), \
                     patch.object(main_discovery, "EventEmitter", return_value=mock_emitter), \
                     patch.object(main_discovery, "search_and_enrich", mock_search_and_enrich):
                    asyncio.run(main_discovery.discover_and_inject(
                        industry="IT",
                        location=location,
                        limit=1,
                    ))

                self.assertTrue(
                    mock_search_and_enrich.called,
                    f"search_and_enrich was NOT called for domain={domain}."
                )

                call_args = mock_search_and_enrich.call_args
                kwargs = call_args.kwargs if call_args.kwargs else {}
                args = call_args.args if call_args.args else ()

                called_domain = kwargs.get("domain") or (args[0] if args else None)
                called_location = kwargs.get("location") or (args[1] if len(args) > 1 else None)
                called_role = kwargs.get("role") or (args[2] if len(args) > 2 else None)

                self.assertEqual(
                    called_domain,
                    domain,
                    f"search_and_enrich called with domain={called_domain!r}, expected {domain!r}."
                )
                self.assertEqual(
                    called_role,
                    role,
                    f"search_and_enrich called with role={called_role!r}, expected {role!r}."
                )
                # location may be derived from company data; just assert it is a non-empty string
                self.assertIsNotNone(
                    called_location,
                    "search_and_enrich called with location=None."
                )
                self.assertIsInstance(
                    called_location,
                    str,
                    f"search_and_enrich location argument is not a string: {called_location!r}."
                )


# ---------------------------------------------------------------------------
# Test 4 — Anti-collision synthetic domain
# Validates: Requirement 3.6
#
# For all company dicts with null/missing domain, the unknown_<name>
# synthetic domain is applied before any Neo4j write.
# ---------------------------------------------------------------------------
class TestAntiCollisionSyntheticDomain(unittest.TestCase):
    """
    Preservation property: when domain and apollo_id are both missing/invalid,
    a synthetic domain starting with 'unknown_' is applied before any Neo4j write.

    Validates: Requirement 3.6
    """

    NULL_DOMAIN_CASES = [
        {"domain": None, "apollo_id": None, "name": "Acme Corp"},
        {"domain": "Non renseigné", "apollo_id": "Non renseigné", "name": "Beta Ltd"},
        {"domain": "", "apollo_id": "", "name": "Gamma Inc"},
    ]

    def _build_company_with_null_domain(self, overrides):
        """Build a full company dict with a null/missing domain."""
        base = _make_apollo_company("placeholder.com", "Placeholder", "France")
        base.update(overrides)
        return base

    def test_synthetic_domain_applied_before_neo4j_write(self):
        """
        For each null-domain company, bulk_import_companies must be called
        with a domain starting with 'unknown_'.
        """
        import asyncio

        for overrides in self.NULL_DOMAIN_CASES:
            with self.subTest(overrides=overrides):
                company = self._build_company_with_null_domain(overrides)
                captured = []

                mock_scraper = MagicMock()
                mock_scraper.search_companies.return_value = [company]
                # enrich_organization returns None for invalid domain (matches real code)
                mock_scraper.enrich_organization.return_value = None

                mock_db = MagicMock()
                mock_db.bulk_import_companies.side_effect = lambda cs: captured.extend(cs)
                mock_db.import_merged_profiles.side_effect = lambda ps: captured.extend(ps)

                async def _fake_crawl(d):
                    return {}

                async def _fake_collect(d, n):
                    return {"recent_news": [], "job_postings_count": 0, "technology_changes": []}

                async def _fake_emit(p):
                    return None

                mock_apify = MagicMock()
                mock_apify.crawl_website = _fake_crawl
                mock_intent = MagicMock()
                mock_intent.collect = _fake_collect
                mock_detective = MagicMock()
                mock_detective.format.return_value = {}
                mock_emitter = MagicMock()
                mock_emitter.emit_lead_ingested = _fake_emit

                with patch.object(main_discovery, "ApolloScraper", return_value=mock_scraper), \
                     patch.object(main_discovery, "Neo4jManager", return_value=mock_db), \
                     patch.object(main_discovery, "ApifyEnricher", return_value=mock_apify), \
                     patch.object(main_discovery, "IntentCollector", return_value=mock_intent), \
                     patch.object(main_discovery, "DetectiveFormatter", return_value=mock_detective), \
                     patch.object(main_discovery, "EventEmitter", return_value=mock_emitter), \
                     patch.object(main_discovery, "search_and_enrich", return_value=[]):
                    asyncio.run(main_discovery.discover_and_inject(industry="IT", location="France", limit=1))

                self.assertGreater(
                    len(captured),
                    0,
                    f"No Neo4j write was captured for overrides={overrides}."
                )

                for written in captured:
                    written_domain = written.get("domain", "")
                    self.assertTrue(
                        str(written_domain).startswith("unknown_"),
                        f"Expected synthetic domain starting with 'unknown_', "
                        f"got {written_domain!r} for overrides={overrides}."
                    )


# ---------------------------------------------------------------------------
# Test 5 — Subsidiary resolution
# Validates: Requirement 3.2
#
# For all Apollo responses with HQ outside target location, recursive
# enrich_organization call is made on the local branch domain.
# ---------------------------------------------------------------------------
class TestSubsidiaryResolution(unittest.TestCase):
    """
    Preservation property: when Apollo returns a company whose HQ is outside
    the target location, enrich_organization recurses on the local branch domain
    found in suborganizations.

    Validates: Requirement 3.2
    """

    def test_recursive_enrich_on_local_branch_domain(self):
        """
        Provide an Apollo response where location.country != target_location
        and suborganizations contains a local branch.
        Assert enrich_organization is called recursively with the local branch domain.
        """
        # The initial search returns a company with HQ in France
        hq_company = _make_apollo_company("alten.fr", "Alten France HQ", "France")
        hq_company["location"]["country"] = "France"

        # The local Swiss branch domain
        local_branch_domain = "alten.ch"
        local_branch_company = _make_apollo_company(local_branch_domain, "Alten Switzerland", "Switzerland")

        # We test enrich_organization directly on ApolloScraper since that is where
        # the subsidiary resolution logic lives (apollo_scraper.py).
        # We import ApolloScraper directly (it only needs requests, which we stub).
        import requests as _requests_mod
        with patch.object(_requests_mod, "get") as mock_get:
            # First call: returns HQ org with suborganizations containing Swiss branch
            hq_org_response = {
                "organization": {
                    "id": "apollo-alten-fr",
                    "primary_domain": "alten.fr",
                    "name": "Alten France HQ",
                    "country": "France",
                    "suborganizations": [
                        {
                            "id": "apollo-alten-ch",
                            "name": "Alten Switzerland",
                            "primary_domain": local_branch_domain,
                        }
                    ],
                    # Minimal fields to avoid KeyError in _format_org_data
                    "industry": "IT",
                    "founded_year": None,
                    "logo_url": None,
                    "website_url": None,
                    "short_description": None,
                    "seo_description": None,
                    "alexa_ranking": None,
                    "annual_revenue": None,
                    "total_funding_printed": None,
                    "estimated_num_employees": None,
                    "latest_funding_stage": None,
                    "linkedin_url": None,
                    "twitter_url": None,
                    "facebook_url": None,
                    "crunchbase_url": None,
                    "phone": None,
                    "raw_address": None,
                    "street_address": None,
                    "city": None,
                    "state": None,
                    "postal_code": None,
                    "num_suborganizations": 1,
                    "owned_by_organization_id": None,
                    "current_technologies": [],
                    "departmental_head_count": {},
                    "funding_events": [],
                    "keywords": [],
                }
            }

            # Second call (recursive): returns the Swiss branch
            branch_org_response = {
                "organization": {
                    "id": "apollo-alten-ch",
                    "primary_domain": local_branch_domain,
                    "name": "Alten Switzerland",
                    "country": "Switzerland",
                    "suborganizations": [],
                    "industry": "IT",
                    "founded_year": None,
                    "logo_url": None,
                    "website_url": None,
                    "short_description": None,
                    "seo_description": None,
                    "alexa_ranking": None,
                    "annual_revenue": None,
                    "total_funding_printed": None,
                    "estimated_num_employees": None,
                    "latest_funding_stage": None,
                    "linkedin_url": None,
                    "twitter_url": None,
                    "facebook_url": None,
                    "crunchbase_url": None,
                    "phone": None,
                    "raw_address": None,
                    "street_address": None,
                    "city": None,
                    "state": None,
                    "postal_code": None,
                    "num_suborganizations": 0,
                    "owned_by_organization_id": None,
                    "current_technologies": [],
                    "departmental_head_count": {},
                    "funding_events": [],
                    "keywords": [],
                }
            }

            # requests.get returns different responses on successive calls
            first_response = MagicMock()
            first_response.status_code = 200
            first_response.json.return_value = hq_org_response

            second_response = MagicMock()
            second_response.status_code = 200
            second_response.json.return_value = branch_org_response

            mock_get.side_effect = [first_response, second_response]

            # Import ApolloScraper directly (not through main_discovery stub)
            # We need the real implementation
            import importlib
            import apollo_scraper as _apollo_mod
            scraper = _apollo_mod.ApolloScraper(api_key="test-key")

            result = scraper.enrich_organization(
                domain="alten.fr",
                target_location="Switzerland",
            )

        # The result should be the Swiss branch (recursive call succeeded)
        self.assertIsNotNone(
            result,
            "enrich_organization returned None — subsidiary resolution did not recurse."
        )
        self.assertEqual(
            result.get("domain"),
            local_branch_domain,
            f"Expected domain={local_branch_domain!r} from recursive call, "
            f"got {result.get('domain')!r}."
        )

        # Assert requests.get was called twice (initial + recursive)
        self.assertEqual(
            mock_get.call_count,
            2,
            f"Expected 2 HTTP GET calls (initial + recursive), got {mock_get.call_count}."
        )

        # Assert the second call used the local branch domain
        second_call_params = mock_get.call_args_list[1]
        # params is passed as keyword arg 'params'
        called_params = second_call_params.kwargs.get("params") or {}
        self.assertEqual(
            called_params.get("domain"),
            local_branch_domain,
            f"Second enrich_organization call used domain={called_params.get('domain')!r}, "
            f"expected {local_branch_domain!r}."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

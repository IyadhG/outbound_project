"""
Property-Based Tests — Inject Agentic Redesign
================================================
These tests verify universal correctness properties using Hypothesis.
Each test runs a minimum of 100 iterations over randomly generated inputs.

Properties covered:
  1. DQS range invariant
  2. DQS idempotence
  3. DQS weighted sum correctness
  7. Gate log entries contain required base fields
  9. SmartScraperAI merge only fills empty fields and extracts .value
  12. Persona worthiness gate decision logic
  13. DetectiveFormatter includes processing_log in payload
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Ensure inject_collect_project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ---------------------------------------------------------------------------
# Pre-stub heavy native dependencies
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
# Imports under test
# ---------------------------------------------------------------------------
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dqs_calculator import compute_dqs
from processing_log import make_log_entry
from detective_formatter import DetectiveFormatter
from main_discovery import _merge_ai_result, _gate_persona_worthiness


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Strategy for values that count as "empty" in the pipeline
_empty_values = st.one_of(st.none(), st.just(""), st.just("Non renseigné"))

# Strategy for values that count as "present" (non-empty, non-None, non-"Non renseigné")
_present_text = st.text(min_size=1).filter(lambda s: s not in ("", "Non renseigné"))

# Strategy for a valid (non-synthetic) domain
_valid_domain = st.from_regex(r"[a-z]{2,10}\.[a-z]{2,5}", fullmatch=True)

# Strategy for a synthetic domain
_synthetic_domain = st.text(min_size=1).map(lambda s: f"unknown_{s}")

# Strategy for DQS float in [0.0, 1.0]
_dqs_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ===========================================================================
# Property 1: DQS range invariant
# ===========================================================================

# Feature: inject-agentic-redesign, Property 1: For any dict, compute_dqs returns a float in [0.0, 1.0]
@given(
    profile=st.dictionaries(
        st.text(),
        st.one_of(
            st.none(),
            st.text(),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
        ),
    )
)
@settings(max_examples=200)
def test_property_1_dqs_range_invariant(profile):
    """
    # Feature: inject-agentic-redesign, Property 1: DQS range invariant
    For any dict (including empty dicts, dicts with arbitrary keys, and dicts
    with all fields populated), compute_dqs SHALL return a float in [0.0, 1.0].
    Validates: Requirements 6.4
    """
    result = compute_dqs(profile)
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert 0.0 <= result <= 1.0, f"DQS {result} out of range [0.0, 1.0]"


# ===========================================================================
# Property 2: DQS idempotence
# ===========================================================================

@given(
    profile=st.dictionaries(
        st.text(),
        st.one_of(st.none(), st.text(), st.integers()),
    )
)
@settings(max_examples=200)
def test_property_2_dqs_idempotence(profile):
    """
    # Feature: inject-agentic-redesign, Property 2: DQS idempotence
    For any merged profile dict, calling compute_dqs twice on the same
    unmodified dict SHALL produce the same result.
    Validates: Requirements 6.5
    """
    result1 = compute_dqs(profile)
    result2 = compute_dqs(profile)
    assert result1 == result2, f"compute_dqs not idempotent: {result1} != {result2}"


# ===========================================================================
# Property 3: DQS weighted sum correctness
# ===========================================================================

@given(
    has_domain=st.booleans(),
    has_name=st.booleans(),
    has_industry=st.booleans(),
    has_employees=st.booleans(),
    has_revenue=st.booleans(),
    has_country=st.booleans(),
    has_linkedin=st.booleans(),
    has_website=st.booleans(),
)
@settings(max_examples=200)
def test_property_3_dqs_weighted_sum_correctness(
    has_domain, has_name, has_industry, has_employees,
    has_revenue, has_country, has_linkedin, has_website,
):
    """
    # Feature: inject-agentic-redesign, Property 3: DQS weighted sum correctness
    For any merged profile dict, the value returned by compute_dqs SHALL equal
    the sum of the weights of all signals whose corresponding field is non-empty
    and non-synthetic.
    Validates: Requirements 6.1
    """
    profile = {}
    expected = 0.0

    if has_domain:
        profile["domain"] = "acme.com"
        expected += 0.20
    if has_name:
        profile["name"] = "Acme Corp"
        expected += 0.10
    if has_industry:
        profile["industry"] = "Software"
        expected += 0.10
    if has_employees:
        profile["estimated_num_employees"] = 100
        expected += 0.15
    if has_revenue:
        profile["annual_revenue"] = "$10M"
        expected += 0.15
    if has_country:
        profile["location"] = {"country": "France"}
        expected += 0.10
    if has_linkedin:
        profile["linkedin_url"] = "https://linkedin.com/company/acme"
        expected += 0.10
    if has_website:
        profile["website_url"] = "https://acme.com"
        expected += 0.10

    result = compute_dqs(profile)
    assert abs(result - expected) < 1e-9, (
        f"Expected DQS {expected:.2f}, got {result:.2f} for profile keys: {list(profile.keys())}"
    )


# ===========================================================================
# Property 7: Gate log entries contain required base fields
# ===========================================================================

@given(
    gate=st.text(min_size=1),
    action=st.text(min_size=1),
    dqs_at_gate=_dqs_float,
    extra=st.dictionaries(
        st.text(min_size=1).filter(lambda s: s not in ("gate", "timestamp", "action", "dqs_at_gate")),
        st.one_of(st.text(), st.integers(), st.booleans(), st.floats(allow_nan=False)),
    ),
)
@settings(max_examples=200)
def test_property_7_log_entry_base_fields(gate, action, dqs_at_gate, extra):
    """
    # Feature: inject-agentic-redesign, Property 7: Gate log entries contain required base fields
    For any invocation of make_log_entry, the returned dict SHALL contain all
    four base fields: gate (string), timestamp (ISO 8601 string), action (string),
    and dqs_at_gate (float). Gate-specific extra kwargs SHALL also be present.
    Validates: Requirements 1.4, 2.7, 3.5, 4.2
    """
    entry = make_log_entry(gate, action, dqs_at_gate, **extra)

    # Base fields must be present
    assert "gate" in entry, "Missing 'gate' field"
    assert "timestamp" in entry, "Missing 'timestamp' field"
    assert "action" in entry, "Missing 'action' field"
    assert "dqs_at_gate" in entry, "Missing 'dqs_at_gate' field"

    # Base field values
    assert entry["gate"] == gate
    assert entry["action"] == action
    assert entry["dqs_at_gate"] == dqs_at_gate

    # Timestamp is ISO 8601 UTC
    ts = entry["timestamp"]
    assert isinstance(ts, str), f"timestamp is not a string: {ts!r}"
    assert "+00:00" in ts or ts.endswith("Z"), f"timestamp not UTC: {ts!r}"

    # Extra kwargs present as top-level keys
    for k, v in extra.items():
        assert k in entry, f"Extra kwarg '{k}' missing from log entry"
        assert entry[k] == v, f"Extra kwarg '{k}' value mismatch: {entry[k]!r} != {v!r}"


# ===========================================================================
# Property 9: SmartScraperAI merge only fills empty fields and extracts .value
# ===========================================================================

@given(
    populated_name=_present_text,
    populated_industry=_present_text,
    ai_name=_present_text,
    ai_industry=_present_text,
)
@settings(max_examples=200)
def test_property_9_merge_preserves_populated_fields(
    populated_name, populated_industry, ai_name, ai_industry
):
    """
    # Feature: inject-agentic-redesign, Property 9: SmartScraperAI merge only fills empty fields
    For any merged profile with some fields already populated and any Apollo Mirror
    JSON result, after calling _merge_ai_result, every field in the merged profile
    that was non-empty before the merge SHALL retain its original value.
    Validates: Requirements 2.4, 5.3
    """
    profile = {
        "name": populated_name,
        "industry": populated_industry,
    }
    ai_data = {
        "identity": {
            "name": {"value": ai_name, "confidence": 0.9, "source": "web"},
            "industry": {"value": ai_industry, "confidence": 0.8, "source": "web"},
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(ai_data, f)
        tmp_path = f.name

    try:
        _merge_ai_result(profile, tmp_path)
    finally:
        os.unlink(tmp_path)

    # Populated fields must not be overwritten
    assert profile["name"] == populated_name, (
        f"Populated 'name' was overwritten: {profile['name']!r} != {populated_name!r}"
    )
    assert profile["industry"] == populated_industry, (
        f"Populated 'industry' was overwritten: {profile['industry']!r} != {populated_industry!r}"
    )


# ===========================================================================
# Property 12: Persona worthiness gate decision logic
# ===========================================================================

@given(
    dqs=_dqs_float,
    job_postings_count=st.integers(min_value=0, max_value=100),
    has_news_items=st.booleans(),
    employee_count=st.one_of(
        st.just(None),
        st.just(""),
        st.just("Non renseigné"),
        st.just(0),
        st.just("0"),
        st.integers(min_value=1, max_value=10000),
        st.text(min_size=1).filter(lambda s: s not in ("", "Non renseigné", "0")),
    ),
)
@settings(max_examples=300)
def test_property_12_persona_worthiness_gate_logic(
    dqs, job_postings_count, has_news_items, employee_count
):
    """
    # Feature: inject-agentic-redesign, Property 12: Persona worthiness gate decision logic
    - The gate SHALL return False when dqs < 0.5.
    - The gate SHALL return False when dqs >= 0.5 AND all signals absent/zero.
    - The gate SHALL return True when dqs >= 0.5 AND at least one signal present.
    Validates: Requirements 3.1, 3.2, 3.3
    """
    profile = {}
    if employee_count is not None:
        profile["estimated_num_employees"] = employee_count

    intent = {
        "job_postings_count": job_postings_count,
        "recent_news": [{"title": "news"}] if has_news_items else [],
    }

    log = []
    result = _gate_persona_worthiness(profile, intent, dqs, log)

    # Compute expected result
    has_employee_count = employee_count not in (None, "", "Non renseigné", 0, "0")
    has_signal = job_postings_count > 0 or has_news_items or has_employee_count
    expected = dqs >= 0.5 and has_signal

    assert result == expected, (
        f"Gate returned {result}, expected {expected} "
        f"(dqs={dqs:.2f}, jobs={job_postings_count}, news={has_news_items}, "
        f"employees={employee_count!r})"
    )

    # Log entry must always be appended
    assert len(log) == 1, f"Expected 1 log entry, got {len(log)}"
    assert log[0]["gate"] == "persona_worthiness"


# ===========================================================================
# Property 13: DetectiveFormatter includes processing_log in payload
# ===========================================================================

@given(
    log_entries=st.lists(
        st.fixed_dictionaries({
            "gate": st.sampled_from(["entity_validation", "data_quality", "persona_worthiness"]),
            "action": st.text(min_size=1),
            "dqs_at_gate": _dqs_float,
            "timestamp": st.just("2026-01-01T00:00:00+00:00"),
        }),
        max_size=5,
    )
)
@settings(max_examples=200)
def test_property_13_detective_formatter_processing_log_round_trip(log_entries):
    """
    # Feature: inject-agentic-redesign, Property 13: DetectiveFormatter includes processing_log in payload
    For any list passed as processing_log to DetectiveFormatter.format(), the
    returned dict SHALL contain a top-level key "processing_log" whose value is
    equal to the input list. When processing_log=None (or omitted), the returned
    dict SHALL contain "processing_log": [].
    Validates: Requirements 4.3
    """
    fmt = DetectiveFormatter()
    profile = {"name": "Test", "domain": "test.com", "data_quality_score": 0.5}

    # With explicit list
    payload = fmt.format(profile, [], {}, processing_log=log_entries)
    assert "processing_log" in payload, "payload missing 'processing_log' key"
    assert payload["processing_log"] == log_entries, (
        f"processing_log mismatch: {payload['processing_log']!r} != {log_entries!r}"
    )

    # With None → empty list
    payload_none = fmt.format(profile, [], {}, processing_log=None)
    assert payload_none["processing_log"] == [], (
        f"Expected [] for processing_log=None, got {payload_none['processing_log']!r}"
    )

    # Without kwarg → empty list
    payload_omitted = fmt.format(profile, [], {})
    assert payload_omitted["processing_log"] == [], (
        f"Expected [] when processing_log omitted, got {payload_omitted['processing_log']!r}"
    )

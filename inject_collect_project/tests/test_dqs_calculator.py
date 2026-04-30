"""
Unit tests for dqs_calculator.compute_dqs.
"""
import pytest
from inject_collect_project.dqs_calculator import compute_dqs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FULL_PROFILE = {
    "domain": "acme.com",
    "name": "Acme Corp",
    "industry": "Software",
    "estimated_num_employees": 500,
    "annual_revenue": 10_000_000,
    "location": {"country": "France"},
    "linkedin_url": "https://linkedin.com/company/acme",
    "website_url": "https://acme.com",
}


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_full_profile_returns_1():
    assert compute_dqs(FULL_PROFILE) == pytest.approx(1.0)


def test_empty_profile_returns_0():
    assert compute_dqs({}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Individual signal weights
# ---------------------------------------------------------------------------

def test_domain_weight():
    assert compute_dqs({"domain": "acme.com"}) == pytest.approx(0.20)


def test_name_weight():
    assert compute_dqs({"name": "Acme"}) == pytest.approx(0.10)


def test_industry_weight():
    assert compute_dqs({"industry": "Tech"}) == pytest.approx(0.10)


def test_employees_weight():
    assert compute_dqs({"estimated_num_employees": 100}) == pytest.approx(0.15)


def test_revenue_weight():
    assert compute_dqs({"annual_revenue": 5_000_000}) == pytest.approx(0.15)


def test_location_country_nested_weight():
    assert compute_dqs({"location": {"country": "France"}}) == pytest.approx(0.10)


def test_location_country_top_level_weight():
    assert compute_dqs({"country": "Germany"}) == pytest.approx(0.10)


def test_linkedin_url_weight():
    assert compute_dqs({"linkedin_url": "https://linkedin.com/company/x"}) == pytest.approx(0.10)


def test_website_url_weight():
    assert compute_dqs({"website_url": "https://example.com"}) == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Empty value definitions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("empty_val", [None, "", "Non renseigné"])
def test_none_empty_string_non_renseigne_count_as_absent(empty_val):
    profile = {
        "domain": empty_val,
        "name": empty_val,
        "industry": empty_val,
        "estimated_num_employees": empty_val,
        "annual_revenue": empty_val,
        "location": {"country": empty_val},
        "country": empty_val,
        "linkedin_url": empty_val,
        "website_url": empty_val,
    }
    assert compute_dqs(profile) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Synthetic domain detection
# ---------------------------------------------------------------------------

def test_unknown_prefix_domain_contributes_zero():
    assert compute_dqs({"domain": "unknown_abc123"}) == pytest.approx(0.0)


def test_unknown_prefix_domain_exact_unknown_contributes_zero():
    assert compute_dqs({"domain": "unknown_"}) == pytest.approx(0.0)


def test_real_domain_contributes():
    assert compute_dqs({"domain": "real-company.com"}) == pytest.approx(0.20)


def test_domain_containing_unknown_but_not_prefix_contributes():
    # "myunknown_domain.com" does NOT start with "unknown_"
    assert compute_dqs({"domain": "myunknown_domain.com"}) == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Location country fallback logic
# ---------------------------------------------------------------------------

def test_nested_location_country_takes_precedence():
    profile = {"location": {"country": "France"}, "country": "Germany"}
    assert compute_dqs(profile) == pytest.approx(0.10)


def test_top_level_country_used_when_nested_absent():
    profile = {"location": {"country": None}, "country": "Germany"}
    assert compute_dqs(profile) == pytest.approx(0.10)


def test_top_level_country_used_when_no_location_key():
    profile = {"country": "Spain"}
    assert compute_dqs(profile) == pytest.approx(0.10)


def test_both_country_fields_absent_contributes_zero():
    profile = {"location": {"country": ""}, "country": "Non renseigné"}
    assert compute_dqs(profile) == pytest.approx(0.0)


def test_location_key_missing_entirely_no_crash():
    # profile has no "location" key at all
    assert compute_dqs({"country": "France"}) == pytest.approx(0.10)


def test_location_none_value_no_crash():
    # profile["location"] is None — should not raise
    assert compute_dqs({"location": None, "country": "France"}) == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------

def test_return_value_never_exceeds_1():
    assert compute_dqs(FULL_PROFILE) <= 1.0


def test_return_value_never_below_0():
    assert compute_dqs({}) >= 0.0


# ---------------------------------------------------------------------------
# Partial profiles
# ---------------------------------------------------------------------------

def test_partial_profile_dqs_below_threshold():
    profile = {"domain": "acme.com", "name": "Acme"}
    assert compute_dqs(profile) == pytest.approx(0.30)


def test_profile_at_0_5_threshold():
    profile = {
        "domain": "acme.com",       # 0.20
        "name": "Acme",             # 0.10
        "estimated_num_employees": 100,  # 0.15
        "annual_revenue": 1_000_000,     # 0.15 → total 0.60
    }
    assert compute_dqs(profile) == pytest.approx(0.60)

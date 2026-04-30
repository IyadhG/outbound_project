"""
Pure DQS (Data Quality Score) computation module.

No I/O, no state, no class — just a single pure function.
"""


def compute_dqs(profile: dict) -> float:
    """
    Compute the Data Quality Score for a merged company profile.

    Returns a float in [0.0, 1.0] as a weighted sum of 8 presence signals.
    "Non renseigné", None, and "" all count as absent (0 contribution).

    Weights:
        non-synthetic domain:       0.20
        non-empty name:             0.10
        non-empty industry:         0.10
        non-empty employee count:   0.15
        non-empty annual revenue:   0.15
        non-empty location.country: 0.10
        non-empty linkedin_url:     0.10
        non-empty website_url:      0.10
    """
    _ABSENT = {None, "", "Non renseigné"}

    def _present(value) -> bool:
        """Return True if value is considered non-empty."""
        return value not in _ABSENT

    score = 0.0

    # --- non-synthetic domain (0.20) ---
    domain = profile.get("domain")
    if _present(domain) and not domain.startswith("unknown_"):
        score += 0.20

    # --- non-empty name (0.10) ---
    if _present(profile.get("name")):
        score += 0.10

    # --- non-empty industry (0.10) ---
    if _present(profile.get("industry")):
        score += 0.10

    # --- non-empty estimated_num_employees (0.15) ---
    if _present(profile.get("estimated_num_employees")):
        score += 0.15

    # --- non-empty annual_revenue (0.15) ---
    if _present(profile.get("annual_revenue")):
        score += 0.15

    # --- non-empty location country (0.10) ---
    # Check nested location.country first, then top-level country
    location_country = (profile.get("location") or {}).get("country")
    top_level_country = profile.get("country")
    if _present(location_country) or _present(top_level_country):
        score += 0.10

    # --- non-empty linkedin_url (0.10) ---
    if _present(profile.get("linkedin_url")):
        score += 0.10

    # --- non-empty website_url (0.10) ---
    if _present(profile.get("website_url")):
        score += 0.10

    # Clamp to [0.0, 1.0] (defensive — weights already sum to 1.0)
    return max(0.0, min(1.0, score))

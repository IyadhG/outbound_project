"""
Unit tests for PersonaScorer (detective/persona_scorer.py).

Covers:
- Rule-based scoring is called for all personas (Req 5.1)
- LLM escalation triggers when score < threshold AND title is ambiguous (Req 5.2)
- LLM score is used when |llm - rule| > 0.1 (Req 5.3)
- Rule score is retained when |llm - rule| <= 0.1 (Req 5.3)
- LLM exception falls back to rule score (Req 5.4)
"""

import pytest
from unittest.mock import MagicMock, patch

import sys
import os

# Ensure the detective package root is on the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from persona_scorer import PersonaScorer


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_ranker(rule_score: float = 0.5, llm_scores: dict | None = None):
    """Return a mock PersonaRanker with configurable return values."""
    ranker = MagicMock()
    ranker.score_persona.return_value = {
        "name": "Test User",
        "job_title": "Analyst",
        "persona_score": rule_score,
    }
    if llm_scores is None:
        llm_scores = {"seniority_score": 0.5, "position_score": 0.5, "target_match_score": 0.5}
    ranker.analyze_persona_with_llm.return_value = llm_scores
    return ranker


def _persona(title_role: str = "analyst", title_level: str = "") -> dict:
    """Return a minimal persona dict."""
    return {
        "full_name": "Test User",
        "job_title_role": title_role,
        "job_title_level": title_level,
        "is_likely_to_engage": 0.5,
        "intent_strength": 5,
    }


# ---------------------------------------------------------------------------
# Test: rule-based scoring is always called (Req 5.1)
# ---------------------------------------------------------------------------

class TestRuleBasedScoringAlwaysCalled:
    def test_score_persona_called_for_high_score_persona(self):
        """score_persona is called even when the rule score is above threshold."""
        ranker = _make_ranker(rule_score=0.8)
        scorer = PersonaScorer(ranker, llm_threshold=0.4)
        persona = _persona("director")

        result, escalated = scorer.score(persona)

        ranker.score_persona.assert_called_once_with(persona)
        assert result["persona_score"] == 0.8
        assert escalated is False

    def test_score_persona_called_for_low_score_persona(self):
        """score_persona is called when the rule score is below threshold."""
        ranker = _make_ranker(rule_score=0.2)
        scorer = PersonaScorer(ranker, llm_threshold=0.4)
        persona = _persona("analyst")  # no seniority keyword

        scorer.score(persona)

        ranker.score_persona.assert_called_once_with(persona)

    def test_returns_rule_result_dict(self):
        """The returned dict is (at minimum) the rule_result dict."""
        ranker = _make_ranker(rule_score=0.7)
        scorer = PersonaScorer(ranker, llm_threshold=0.4)

        result, _ = scorer.score(_persona("manager"))

        assert result["name"] == "Test User"
        assert "persona_score" in result


# ---------------------------------------------------------------------------
# Test: LLM escalation triggers correctly (Req 5.2)
# ---------------------------------------------------------------------------

class TestLLMEscalationTrigger:
    def test_no_escalation_when_score_above_threshold(self):
        """LLM is NOT called when rule_score >= llm_threshold."""
        ranker = _make_ranker(rule_score=0.5)
        scorer = PersonaScorer(ranker, llm_threshold=0.4)

        _, escalated = scorer.score(_persona("analyst"))

        ranker.analyze_persona_with_llm.assert_not_called()
        assert escalated is False

    def test_no_escalation_when_title_has_seniority_keyword(self):
        """LLM is NOT called when title contains a seniority keyword, even if score is low."""
        for keyword in ["ceo", "director", "vp", "head of", "manager", "senior"]:
            ranker = _make_ranker(rule_score=0.1)
            scorer = PersonaScorer(ranker, llm_threshold=0.4)

            _, escalated = scorer.score(_persona(keyword))

            ranker.analyze_persona_with_llm.assert_not_called(), (
                f"LLM should not be called for title containing '{keyword}'"
            )
            assert escalated is False

    def test_escalation_when_score_below_threshold_and_ambiguous_title(self):
        """LLM IS called when rule_score < threshold AND title has no seniority keyword."""
        ranker = _make_ranker(rule_score=0.2)
        scorer = PersonaScorer(ranker, llm_threshold=0.4)

        scorer.score(_persona("analyst"))

        ranker.analyze_persona_with_llm.assert_called_once()

    def test_escalation_uses_persona_passed_to_score(self):
        """analyze_persona_with_llm receives the same persona dict."""
        ranker = _make_ranker(rule_score=0.1)
        scorer = PersonaScorer(ranker, llm_threshold=0.4)
        persona = _persona("coordinator")

        scorer.score(persona)

        ranker.analyze_persona_with_llm.assert_called_once_with(persona)

    def test_no_escalation_at_exact_threshold(self):
        """LLM is NOT called when rule_score == llm_threshold (boundary: not strictly less)."""
        ranker = _make_ranker(rule_score=0.4)
        scorer = PersonaScorer(ranker, llm_threshold=0.4)

        _, escalated = scorer.score(_persona("analyst"))

        ranker.analyze_persona_with_llm.assert_not_called()
        assert escalated is False


# ---------------------------------------------------------------------------
# Test: LLM score selection logic (Req 5.3)
# ---------------------------------------------------------------------------

class TestLLMScoreSelection:
    def _llm_scores_for_composite(self, composite: float) -> dict:
        """Return LLM sub-scores that produce the given composite (equal weights)."""
        return {
            "seniority_score": composite,
            "position_score": composite,
            "target_match_score": composite,
        }

    def test_llm_score_used_when_difference_greater_than_0_1(self):
        """When |llm - rule| > 0.1, the returned persona_score equals the LLM composite."""
        rule_score = 0.2
        llm_composite = 0.7  # difference = 0.5 > 0.1
        ranker = _make_ranker(
            rule_score=rule_score,
            llm_scores=self._llm_scores_for_composite(llm_composite),
        )
        scorer = PersonaScorer(ranker, llm_threshold=0.4)

        result, escalated = scorer.score(_persona("coordinator"))

        assert escalated is True
        assert abs(result["persona_score"] - llm_composite) < 1e-3

    def test_rule_score_retained_when_difference_at_most_0_1(self):
        """When |llm - rule| <= 0.1, the returned persona_score equals the rule score."""
        rule_score = 0.3
        llm_composite = 0.35  # difference = 0.05 <= 0.1
        ranker = _make_ranker(
            rule_score=rule_score,
            llm_scores=self._llm_scores_for_composite(llm_composite),
        )
        scorer = PersonaScorer(ranker, llm_threshold=0.4)

        result, escalated = scorer.score(_persona("coordinator"))

        assert escalated is False
        assert result["persona_score"] == rule_score

    def test_rule_score_retained_when_difference_exactly_0_1(self):
        """Boundary: |llm - rule| == 0.1 → keep rule score (not strictly greater)."""
        rule_score = 0.2
        llm_composite = 0.3  # difference = 0.1 exactly
        ranker = _make_ranker(
            rule_score=rule_score,
            llm_scores=self._llm_scores_for_composite(llm_composite),
        )
        scorer = PersonaScorer(ranker, llm_threshold=0.4)

        result, escalated = scorer.score(_persona("coordinator"))

        assert escalated is False
        assert result["persona_score"] == rule_score

    def test_llm_score_capped_at_1(self):
        """LLM composite > 1.0 is capped to 1.0 in the returned persona_score."""
        rule_score = 0.1
        # sub-scores > 1 are unusual but we guard against them
        ranker = _make_ranker(
            rule_score=rule_score,
            llm_scores={"seniority_score": 1.5, "position_score": 1.5, "target_match_score": 1.5},
        )
        scorer = PersonaScorer(ranker, llm_threshold=0.4)

        result, escalated = scorer.score(_persona("coordinator"))

        assert escalated is True
        assert result["persona_score"] <= 1.0


# ---------------------------------------------------------------------------
# Test: LLM exception fallback (Req 5.4)
# ---------------------------------------------------------------------------

class TestLLMExceptionFallback:
    def test_returns_rule_score_on_llm_exception(self):
        """When analyze_persona_with_llm raises, score() returns the rule-based result."""
        ranker = _make_ranker(rule_score=0.2)
        ranker.analyze_persona_with_llm.side_effect = RuntimeError("API timeout")
        scorer = PersonaScorer(ranker, llm_threshold=0.4)

        result, escalated = scorer.score(_persona("coordinator"))

        assert escalated is False
        assert result["persona_score"] == 0.2

    def test_does_not_raise_on_llm_exception(self):
        """score() must not propagate exceptions from analyze_persona_with_llm."""
        ranker = _make_ranker(rule_score=0.1)
        ranker.analyze_persona_with_llm.side_effect = Exception("Unexpected error")
        scorer = PersonaScorer(ranker, llm_threshold=0.4)

        # Should not raise
        result, escalated = scorer.score(_persona("coordinator"))
        assert result is not None

    def test_logs_warning_on_llm_exception(self, caplog):
        """A WARNING is logged when analyze_persona_with_llm raises."""
        import logging
        ranker = _make_ranker(rule_score=0.1)
        ranker.analyze_persona_with_llm.side_effect = ValueError("Bad response")
        scorer = PersonaScorer(ranker, llm_threshold=0.4)

        with caplog.at_level(logging.WARNING):
            scorer.score(_persona("coordinator"))

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) > 0, "Expected at least one WARNING log record"

    def test_rule_result_dict_returned_unchanged_on_exception(self):
        """The exact rule_result dict is returned when LLM raises."""
        expected_result = {
            "name": "Jane Doe",
            "job_title": "Coordinator",
            "persona_score": 0.15,
        }
        ranker = MagicMock()
        ranker.score_persona.return_value = expected_result
        ranker.analyze_persona_with_llm.side_effect = ConnectionError("Network error")
        scorer = PersonaScorer(ranker, llm_threshold=0.4)

        result, escalated = scorer.score(_persona("coordinator"))

        assert result is expected_result
        assert escalated is False


# ---------------------------------------------------------------------------
# Test: SENIORITY_KEYWORDS class attribute
# ---------------------------------------------------------------------------

class TestSeniorityKeywords:
    def test_seniority_keywords_is_frozenset(self):
        assert isinstance(PersonaScorer.SENIORITY_KEYWORDS, frozenset)

    def test_seniority_keywords_contains_expected_terms(self):
        expected = {"ceo", "cto", "vp", "director", "head of", "chief", "manager"}
        assert expected.issubset(PersonaScorer.SENIORITY_KEYWORDS)

    def test_seniority_keywords_are_lowercase(self):
        for kw in PersonaScorer.SENIORITY_KEYWORDS:
            assert kw == kw.lower(), f"Keyword '{kw}' is not lowercase"


# ---------------------------------------------------------------------------
# Test: default llm_threshold
# ---------------------------------------------------------------------------

class TestDefaultThreshold:
    def test_default_threshold_is_0_4(self):
        ranker = _make_ranker()
        scorer = PersonaScorer(ranker)
        assert scorer.llm_threshold == 0.4

    def test_custom_threshold_respected(self):
        ranker = _make_ranker(rule_score=0.5)
        scorer = PersonaScorer(ranker, llm_threshold=0.6)

        # rule_score=0.5 < threshold=0.6 → should escalate for ambiguous title
        scorer.score(_persona("analyst"))
        ranker.analyze_persona_with_llm.assert_called_once()

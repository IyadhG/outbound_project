"""
PersonaScorer — hybrid persona scoring component.

Wraps PersonaRanker to provide rule-based scoring with optional LLM escalation
for ambiguous personas (low rule score + no recognized seniority keyword).

Requirements: 5.1, 5.2, 5.3, 5.4
"""

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from detective.ranking.persona_ranker import PersonaRanker

logger = logging.getLogger(__name__)


class PersonaScorer:
    """
    Hybrid persona scorer: rule-based first, LLM escalation when needed.

    Escalation logic (Requirement 5.2):
    - If rule_score < llm_threshold AND the persona's job title contains no
      recognized seniority keyword → invoke analyze_persona_with_llm().
    - If |llm_score - rule_score| > 0.1 → use llm_score (Requirement 5.3).
    - Otherwise → keep rule_score (Requirement 5.3).
    - On any exception from analyze_persona_with_llm() → fall back to
      rule_score and log WARNING (Requirement 5.4).
    """

    # Recognized seniority keywords used to decide whether LLM escalation is
    # needed.  A title that contains any of these is considered unambiguous and
    # will NOT be escalated even when the rule-based score is low.
    SENIORITY_KEYWORDS: frozenset = frozenset([
        "ceo",
        "cto",
        "cfo",
        "coo",
        "cmo",
        "vp",
        "vice president",
        "director",
        "head of",
        "chief",
        "manager",
        "lead",
        "senior",
        "founder",
        "co-founder",
        "owner",
        "president",
    ])

    def __init__(
        self,
        persona_ranker: "PersonaRanker",
        llm_threshold: float = 0.4,
    ) -> None:
        """
        Initialise the scorer.

        Args:
            persona_ranker: An initialised PersonaRanker instance.  Its
                ``score_persona`` and ``analyze_persona_with_llm`` methods are
                called internally.
            llm_threshold: Rule-based score below which LLM escalation is
                considered (default 0.4, configurable via
                DETECTIVE_PERSONA_LLM_THRESHOLD env var).
        """
        self.persona_ranker = persona_ranker
        self.llm_threshold = llm_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_title(self, persona: dict) -> str:
        """Return a lower-cased composite job title from the persona dict."""
        role = persona.get("job_title_role", "") or ""
        level = persona.get("job_title_level", "") or ""
        # Also accept a pre-combined 'job_title' key (used in some test fixtures)
        combined = persona.get("job_title", "") or ""
        title = f"{level} {role} {combined}".strip().lower()
        return title

    def _has_seniority_keyword(self, title: str) -> bool:
        """Return True if *title* contains at least one seniority keyword.

        Multi-word keywords (e.g. "head of", "vice president", "co-founder")
        are matched as substrings.  Single-word keywords are matched at word
        boundaries to avoid false positives (e.g. "coo" inside "coordinator").
        """
        for keyword in self.SENIORITY_KEYWORDS:
            if " " in keyword or "-" in keyword:
                # Multi-word / hyphenated: substring match is fine
                if keyword in title:
                    return True
            else:
                # Single word: require word boundary to avoid partial matches
                if re.search(r"\b" + re.escape(keyword) + r"\b", title):
                    return True
        return False

    def _llm_composite_score(self, llm_result: dict) -> float:
        """
        Derive a single composite score from analyze_persona_with_llm output.

        analyze_persona_with_llm returns:
            {seniority_score, position_score, target_match_score}

        We combine them with equal weights (1/3 each) to produce a value in
        [0, 1].  This mirrors the spirit of the rule-based scorer without
        duplicating its exact formula.
        """
        seniority = float(llm_result.get("seniority_score", 0.5))
        position = float(llm_result.get("position_score", 0.5))
        target = float(llm_result.get("target_match_score", 0.0))
        return round((seniority + position + target) / 3.0, 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, persona: dict) -> tuple:
        """
        Score a persona using rule-based scoring with optional LLM escalation.

        Args:
            persona: Raw persona dict (as stored in the personas JSON files).

        Returns:
            A 2-tuple ``(scored_persona_dict, was_llm_escalated)`` where:
            - ``scored_persona_dict`` is the dict returned by
              ``PersonaRanker.score_persona()`` (possibly with an updated
              ``persona_score`` when LLM escalation changed the score).
            - ``was_llm_escalated`` is ``True`` when
              ``analyze_persona_with_llm()`` was successfully invoked and its
              result was used to override the rule-based score.

        Requirements: 5.1, 5.2, 5.3, 5.4
        """
        # --- Step 1: Rule-based scoring (Requirement 5.1) ---
        rule_result: dict = self.persona_ranker.score_persona(persona)
        rule_score: float = float(rule_result.get("persona_score", 0.0))

        # --- Step 2: Decide whether to escalate to LLM (Requirement 5.2) ---
        title = self._get_title(persona)
        should_escalate = (
            rule_score < self.llm_threshold
            and not self._has_seniority_keyword(title)
        )

        if not should_escalate:
            return rule_result, False

        # --- Step 3: LLM escalation ---
        try:
            llm_result = self.persona_ranker.analyze_persona_with_llm(persona)
            llm_score = self._llm_composite_score(llm_result)

            # Requirement 5.3: use LLM score only when the difference is > 0.1
            if abs(llm_score - rule_score) > 0.1:
                scored_persona = dict(rule_result)
                scored_persona["persona_score"] = round(min(llm_score, 1.0), 3)
                scored_persona["llm_score"] = llm_score
                scored_persona["llm_escalated"] = True
                return scored_persona, True
            else:
                # Difference is small — keep rule-based result
                return rule_result, False

        except Exception as exc:  # Requirement 5.4
            logger.warning(
                "analyze_persona_with_llm() raised an exception for persona "
                "'%s'; falling back to rule-based score. Error: %s",
                persona.get("full_name", "<unknown>"),
                exc,
            )
            return rule_result, False

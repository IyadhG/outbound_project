"""
Detective Agent — ReAct-based agentic pipeline.

This file defines the core data types (ScratchpadEntry, AgentResult) and the
DetectiveAgent class. The __init__ implementation covers startup validation,
environment variable reading, and agent graph creation (Task 4). The run()
implementation is added in Task 5.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, TypedDict

logger = logging.getLogger(__name__)


class ScratchpadEntry(TypedDict):
    """A single entry in the agent's reasoning trace."""

    step: int
    type: Literal["thought", "tool_call", "observation", "error"]
    content: str
    timestamp: str  # ISO 8601 UTC, e.g. "2024-01-15T12:34:56.789Z"


class AgentResult(TypedDict):
    """Structured result returned by DetectiveAgent.run()."""

    final_rankings: List[Dict[str, Any]]
    persona_results: List[Dict[str, Any]]
    agent_scratchpad: List[ScratchpadEntry]
    total_iterations: int
    halt_reason: str  # "goal_achieved" | "max_iterations_reached"
    extracted_icp: Dict[str, Any]  # ICPAttributes.model_dump()
    errors: List[str]


class DetectiveAgent:
    """
    ReAct agent that wraps all brain/ and ranking/ tools.

    Uses langgraph.prebuilt.create_react_agent with Groq function calling.
    Full implementation is added in Tasks 4 (.__init__) and 5 (.run).
    """

    def __init__(
        self,
        groq_api_key: str,
        gemini_api_key: str,
        ors_api_key: str | None = None,
        llm_model: str = "llama-3.1-8b-instant",
        max_iterations: int = 15,
        qualification_threshold: float = 0.6,
        persona_llm_threshold: float = 0.4,
    ) -> None:
        """
        Initialise the DetectiveAgent with startup validation.

        Environment variables provide defaults; constructor params override them
        when the caller passes a non-default value.

        Args:
            groq_api_key: Groq API key (required, non-empty).
            gemini_api_key: Gemini API key (required, non-empty).
            ors_api_key: OpenRouteService API key (optional; geo-filtering
                disabled when absent).
            llm_model: Groq model name. Overrides DETECTIVE_LLM_MODEL env var
                when a non-default value is supplied.
            max_iterations: Maximum ReAct loop cycles. Overrides
                DETECTIVE_MAX_ITERATIONS env var when a non-default value is
                supplied.
            qualification_threshold: Minimum final_score for a qualified lead.
                Overrides QUALIFICATION_THRESHOLD env var when a non-default
                value is supplied.
            persona_llm_threshold: Rule-based score below which LLM escalation
                is considered. Overrides DETECTIVE_PERSONA_LLM_THRESHOLD env
                var when a non-default value is supplied.

        Raises:
            ValueError: If groq_api_key or gemini_api_key is empty/missing.
        """
        # ------------------------------------------------------------------
        # 1. Validate required API keys before doing anything else.
        # ------------------------------------------------------------------
        missing = []
        if not groq_api_key:
            missing.append("GROQ_API_KEY")
        if not gemini_api_key:
            missing.append("GEMINI_API_KEY")
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        # ------------------------------------------------------------------
        # 2. Read environment variables as defaults; constructor params
        #    override when the caller passes a non-default value.
        # ------------------------------------------------------------------
        self.max_iterations: int = (
            max_iterations
            if max_iterations != 15
            else int(os.environ.get("DETECTIVE_MAX_ITERATIONS", 15))
        )

        self.llm_model: str = (
            llm_model
            if llm_model != "llama-3.1-8b-instant"
            else os.environ.get("DETECTIVE_LLM_MODEL", "llama-3.1-8b-instant")
        )

        self.qualification_threshold: float = (
            qualification_threshold
            if qualification_threshold != 0.6
            else float(os.environ.get("QUALIFICATION_THRESHOLD", 0.6))
        )

        self.persona_llm_threshold: float = (
            persona_llm_threshold
            if persona_llm_threshold != 0.4
            else float(os.environ.get("DETECTIVE_PERSONA_LLM_THRESHOLD", 0.4))
        )

        # ------------------------------------------------------------------
        # 3. Store API keys as instance attributes.
        # ------------------------------------------------------------------
        self.groq_api_key: str = groq_api_key
        self.gemini_api_key: str = gemini_api_key
        self.ors_api_key: str | None = ors_api_key

        # ------------------------------------------------------------------
        # 4. Log INFO if ORS key is absent (geo-filtering disabled).
        # ------------------------------------------------------------------
        if not ors_api_key:
            logger.info(
                "ORS_API_KEY is absent — geo-filtering is disabled for this agent instance."
            )

        # ------------------------------------------------------------------
        # 5. Initialise per-run retry state (reset at the start of each run).
        # ------------------------------------------------------------------
        self._retry_counts: Dict[str, int] = {}

        # ------------------------------------------------------------------
        # 6. Propagate API keys to environment so tool wrappers can read them.
        # ------------------------------------------------------------------
        os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ["GEMINI_API_KEY"] = gemini_api_key
        if ors_api_key:
            os.environ["ORS_API_KEY"] = ors_api_key

        # ------------------------------------------------------------------
        # 7. Build the tool list and create the ReAct agent graph.
        # ------------------------------------------------------------------
        from agent_tools import (
            calculate_final_scores,
            collect_intent,
            extract_icp,
            filter_companies,
            geo_filter,
            match_companies,
            rank_companies,
            score_personas,
        )
        from langchain_groq import ChatGroq
        from langgraph.prebuilt import create_react_agent

        tools = [
            extract_icp,
            match_companies,
            geo_filter,
            filter_companies,
            rank_companies,
            collect_intent,
            calculate_final_scores,
            score_personas,
        ]

        llm = ChatGroq(
            api_key=groq_api_key,
            model=self.llm_model,
        )

        self._graph = create_react_agent(llm, tools=tools)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _now_iso(self) -> str:
        """Return the current UTC time as an ISO 8601 string with Z suffix."""
        return datetime.utcnow().isoformat() + "Z"

    def _make_entry(
        self,
        step: int,
        entry_type: Literal["thought", "tool_call", "observation", "error"],
        content: str,
    ) -> "ScratchpadEntry":
        """Build a ScratchpadEntry dict."""
        return ScratchpadEntry(
            step=step,
            type=entry_type,
            content=content,
            timestamp=self._now_iso(),
        )

    def _should_retry(self, tool_name: str, result: dict) -> bool:
        """Return True if the tool should be retried (empty result, under cap)."""
        count = self._retry_counts.get(tool_name, 0)
        is_empty = (
            result.get("count", 0) == 0
            or not result.get("results")
            or "error" in result
        )
        return is_empty and count < 3

    def _record_retry(self, tool_name: str) -> None:
        """Increment the retry counter for a tool."""
        self._retry_counts[tool_name] = self._retry_counts.get(tool_name, 0) + 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        icp_text: str,
        desired_lead_count: int = 10,
        output_name: str = "agent_run",
    ) -> "AgentResult":
        """
        Execute the ReAct loop to find qualified leads.

        Args:
            icp_text: Natural-language description of the Ideal Customer Profile.
            desired_lead_count: Stop when this many qualified leads are found.
            output_name: Label for this run (used in logging / output filenames).

        Returns:
            AgentResult with all fields populated.
        """
        # ------------------------------------------------------------------
        # 1. Reset per-run state.
        # ------------------------------------------------------------------
        self._retry_counts = {}
        agent_scratchpad: List[ScratchpadEntry] = []
        errors: List[str] = []
        step_counter = 0

        def _next_step() -> int:
            nonlocal step_counter
            step_counter += 1
            return step_counter

        # ------------------------------------------------------------------
        # 2. Build the system prompt.
        # ------------------------------------------------------------------
        system_prompt = f"""You are the Detective Agent — an expert B2B lead-generation assistant.

Your goal is to find {desired_lead_count} qualified leads (companies + personas) that match the
Ideal Customer Profile (ICP) provided by the user.

## Workflow

Follow this sequence of tool calls. Adapt based on what each tool returns:

1. **extract_icp** — Parse the ICP text into structured attributes (industries, company size,
   geography, target roles). Always start here.

2. **match_companies** — Use the extracted industries to find matching companies.
   If the result is empty, broaden the industry list with semantically related terms and retry.

3. **geo_filter** (optional) — If the ICP specifies a target city, filter companies by proximity.
   Skip this step if no city is mentioned.

4. **filter_companies** — Apply employee-count and country filters from the ICP.
   If the result is empty, relax the most restrictive filter by 20% and retry.
   If still empty after relaxing size, relax the country filter to the continent level.

5. **rank_companies** — Rank the filtered companies by semantic similarity to the ICP.
   If fewer than {desired_lead_count} results are returned, retry with the unfiltered matched set.

6. **collect_intent** (optional) — Collect intent signals for the top companies.
   If this tool fails or is unavailable, skip it and proceed without intent signals.

7. **calculate_final_scores** — Combine similarity scores and intent signals into final scores.

8. **score_personas** — For each top-ranked company, score its personas against the ICP target roles.

## Termination

Stop as soon as you have accumulated {desired_lead_count} companies with final_score ≥ {self.qualification_threshold}.
Do not call more tools than necessary.

## Retry Rules

- Never retry any single tool more than 3 times.
- When a tool returns empty results, record the observation and reason about a corrective action
  before retrying with broadened criteria.
- If all retries are exhausted and results remain empty, move on to the next applicable tool.

## Output

After scoring personas, summarise the qualified leads found and stop.
"""

        user_message = (
            f"Find the top {desired_lead_count} qualified leads for the following ICP:\n\n{icp_text}"
        )

        # ------------------------------------------------------------------
        # 3. Invoke the LangGraph ReAct agent.
        # ------------------------------------------------------------------
        final_rankings: List[Dict[str, Any]] = []
        persona_results: List[Dict[str, Any]] = []
        extracted_icp: Dict[str, Any] = {}
        halt_reason: str = "max_iterations_reached"
        total_iterations: int = 0

        try:
            from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

            result = self._graph.invoke(
                {
                    "messages": [
                        ("system", system_prompt),
                        ("human", user_message),
                    ]
                },
                config={"recursion_limit": self.max_iterations * 2 + 10},
            )

            messages = result.get("messages", [])

            # ------------------------------------------------------------------
            # 4. Walk the message list and populate the scratchpad.
            # ------------------------------------------------------------------
            iteration_count = 0
            tool_call_id_to_name: Dict[str, str] = {}

            for msg in messages:
                # Skip the initial human message
                if isinstance(msg, HumanMessage):
                    continue

                if isinstance(msg, AIMessage):
                    tool_calls = getattr(msg, "tool_calls", None) or []

                    if tool_calls:
                        # One scratchpad entry per tool call
                        for tc in tool_calls:
                            tool_name = tc.get("name", "unknown_tool")
                            tool_args = tc.get("args", {})
                            tc_id = tc.get("id", "")
                            if tc_id:
                                tool_call_id_to_name[tc_id] = tool_name

                            try:
                                args_str = json.dumps(tool_args)
                            except (TypeError, ValueError):
                                args_str = str(tool_args)

                            agent_scratchpad.append(
                                self._make_entry(
                                    _next_step(),
                                    "tool_call",
                                    f"Tool: {tool_name} | Args: {args_str}",
                                )
                            )
                        iteration_count += 1
                    else:
                        # Pure reasoning / thought
                        content = msg.content or ""
                        if content:
                            agent_scratchpad.append(
                                self._make_entry(
                                    _next_step(),
                                    "thought",
                                    str(content),
                                )
                            )

                elif isinstance(msg, ToolMessage):
                    tool_name = tool_call_id_to_name.get(
                        getattr(msg, "tool_call_id", ""), "unknown_tool"
                    )
                    raw_content = msg.content or ""

                    # Parse the tool result to build a summary and check retry.
                    try:
                        tool_result = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                    except (json.JSONDecodeError, TypeError):
                        tool_result = {}

                    count = tool_result.get("count", 0) if isinstance(tool_result, dict) else 0
                    has_error = isinstance(tool_result, dict) and "error" in tool_result
                    retry_triggered = self._should_retry(tool_name, tool_result) if isinstance(tool_result, dict) else False

                    if retry_triggered:
                        self._record_retry(tool_name)

                    summary_parts = [f"Tool: {tool_name}"]
                    if has_error:
                        summary_parts.append(f"Error: {tool_result.get('error', 'unknown')}")
                    else:
                        summary_parts.append(f"Result count: {count}")
                    if retry_triggered:
                        summary_parts.append("retry_triggered=True")

                    agent_scratchpad.append(
                        self._make_entry(
                            _next_step(),
                            "observation",
                            " | ".join(summary_parts),
                        )
                    )

                    # ----------------------------------------------------------
                    # 5. Extract structured results from specific tool outputs.
                    # ----------------------------------------------------------
                    if isinstance(tool_result, dict):
                        if tool_name == "extract_icp" and not has_error:
                            # Store the extracted ICP (exclude error/count keys)
                            extracted_icp = {
                                k: v
                                for k, v in tool_result.items()
                                if k not in ("error", "results", "count")
                            }

                        elif tool_name == "calculate_final_scores" and not has_error:
                            results_list = tool_result.get("results", [])
                            if isinstance(results_list, list):
                                final_rankings = results_list

                        elif tool_name == "score_personas" and not has_error:
                            selected = tool_result.get("selected_persona")
                            if selected:
                                persona_results.append(selected)

            total_iterations = iteration_count

            # ------------------------------------------------------------------
            # 6. Enforce max_iterations cap.
            # ------------------------------------------------------------------
            if total_iterations >= self.max_iterations:
                logger.warning(
                    "DetectiveAgent run '%s' reached max_iterations=%d — returning best-effort results.",
                    output_name,
                    self.max_iterations,
                )
                halt_reason = "max_iterations_reached"
            else:
                # Check if we achieved the goal
                qualified = [
                    r for r in final_rankings
                    if r.get("final_score", 0.0) >= self.qualification_threshold
                ]
                if len(qualified) >= desired_lead_count:
                    halt_reason = "goal_achieved"
                else:
                    # Ran to completion without hitting max_iterations
                    halt_reason = "goal_achieved"

        except Exception as exc:
            error_msg = f"DetectiveAgent graph invocation failed: {exc}"
            logger.error(error_msg)
            errors.append(error_msg)
            agent_scratchpad.append(
                self._make_entry(_next_step(), "error", error_msg)
            )
            halt_reason = "max_iterations_reached"
            total_iterations = 0

        # ------------------------------------------------------------------
        # 7. Sort final_rankings by final_score descending.
        # ------------------------------------------------------------------
        final_rankings.sort(key=lambda r: r.get("final_score", 0.0), reverse=True)

        return AgentResult(
            final_rankings=final_rankings,
            persona_results=persona_results,
            agent_scratchpad=agent_scratchpad,
            total_iterations=total_iterations,
            halt_reason=halt_reason,
            extracted_icp=extracted_icp,
            errors=errors,
        )

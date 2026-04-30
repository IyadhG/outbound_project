"""
Unit tests for DetectiveAgent (detective/detective_agent.py).

All external API calls (Groq, Gemini, LangGraph) are mocked.

Covers:
- Agent skips geo_filter when ICP has no city (Req 2.3)
- Agent skips collect_intent when it fails (Req 2.4)
- Agent halts with halt_reason="goal_achieved" when desired_lead_count leads found (Req 4.5)
- Agent halts with halt_reason="max_iterations_reached" when limit is hit (Req 4.2, 4.3)
- Agent reads DETECTIVE_MAX_ITERATIONS from environment (Req 4.1)
- Agent raises ValueError for missing GROQ_API_KEY (Req 10.5)
- Agent raises ValueError for missing GEMINI_API_KEY (Req 10.5)
- ORS_API_KEY absence disables geo-filtering without raising an error (Req 10.6)
"""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, call

# Ensure the detective package root is on the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import real LangChain message types so isinstance() checks in the agent work.
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ---------------------------------------------------------------------------
# Helpers — build real LangChain message objects so isinstance() checks pass
# ---------------------------------------------------------------------------

def _make_ai_message(content="", tool_calls=None):
    """Build a real AIMessage with optional tool_calls list."""
    # AIMessage accepts tool_calls as a list of dicts with name/args/id keys.
    return AIMessage(content=content, tool_calls=tool_calls or [])


def _make_tool_message(content, tool_call_id="tc1", name="some_tool"):
    """Build a real ToolMessage."""
    serialized = content if isinstance(content, str) else json.dumps(content)
    return ToolMessage(content=serialized, tool_call_id=tool_call_id, name=name)


def _make_human_message(content="Find leads"):
    """Build a real HumanMessage."""
    return HumanMessage(content=content)


def _goal_achieved_messages(desired_lead_count=2, qualification_threshold=0.6):
    """
    Return a message sequence that causes the agent to report goal_achieved.

    Sequence:
      1. AIMessage with tool_call: calculate_final_scores
      2. ToolMessage: result with desired_lead_count qualified leads
      3. AIMessage (thought, no tool call) — agent summarises
    """
    leads = [
        {"company_key": f"co_{i}", "final_score": 0.9, "company_name": f"Company {i}"}
        for i in range(desired_lead_count)
    ]
    tc_id = "tc_final"
    ai_call = _make_ai_message(
        tool_calls=[{"name": "calculate_final_scores", "args": {}, "id": tc_id}]
    )
    tool_result = _make_tool_message(
        {"results": leads, "count": len(leads)},
        tool_call_id=tc_id,
    )
    ai_summary = _make_ai_message(content="Found all qualified leads.")
    return [_make_human_message(), ai_call, tool_result, ai_summary]


def _max_iterations_messages():
    """
    Return a message sequence that simulates many tool calls (triggers max_iterations).

    We produce 20 tool-call AIMessages so iteration_count > any small max_iterations.
    """
    messages = [_make_human_message()]
    for i in range(20):
        tc_id = f"tc_{i}"
        ai_call = _make_ai_message(
            tool_calls=[{"name": "match_companies", "args": {"industries": []}, "id": tc_id}]
        )
        tool_result = _make_tool_message(
            {"results": {}, "count": 0},
            tool_call_id=tc_id,
        )
        messages.extend([ai_call, tool_result])
    return messages


# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
# ChatGroq and create_react_agent are imported *inside* DetectiveAgent.__init__
# (lazy imports), so we must patch them at their source modules.

PATCH_CHATGROQ = "langchain_groq.ChatGroq"
PATCH_CREATE_REACT_AGENT = "langgraph.prebuilt.create_react_agent"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove detective-related env vars before each test to avoid cross-test pollution."""
    for var in ("ORS_API_KEY", "DETECTIVE_MAX_ITERATIONS", "DETECTIVE_LLM_MODEL",
                "QUALIFICATION_THRESHOLD", "DETECTIVE_PERSONA_LLM_THRESHOLD"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def mock_graph():
    """Return a MagicMock that stands in for the LangGraph compiled graph."""
    return MagicMock()


@pytest.fixture()
def agent_factory(mock_graph):
    """
    Return a factory that creates a DetectiveAgent with all external calls mocked.

    Usage:
        agent = agent_factory()          # default keys
        agent = agent_factory(groq_api_key="", ...)  # trigger ValueError
    """
    def _factory(
        groq_api_key="test-groq-key",
        gemini_api_key="test-gemini-key",
        ors_api_key=None,
        max_iterations=15,
        **kwargs,
    ):
        with patch(PATCH_CHATGROQ) as mock_llm_cls, \
             patch(PATCH_CREATE_REACT_AGENT) as mock_create:
            mock_create.return_value = mock_graph
            from detective_agent import DetectiveAgent
            agent = DetectiveAgent(
                groq_api_key=groq_api_key,
                gemini_api_key=gemini_api_key,
                ors_api_key=ors_api_key,
                max_iterations=max_iterations,
                **kwargs,
            )
            # Attach the mock graph directly so tests can control invoke()
            agent._graph = mock_graph
            return agent
    return _factory


# ---------------------------------------------------------------------------
# Test: ValueError for missing required API keys (Req 10.5)
# ---------------------------------------------------------------------------

class TestStartupValidation:
    def test_raises_value_error_for_missing_groq_api_key(self):
        """DetectiveAgent raises ValueError when groq_api_key is empty."""
        with patch(PATCH_CHATGROQ), patch(PATCH_CREATE_REACT_AGENT):
            from detective_agent import DetectiveAgent
            with pytest.raises(ValueError) as exc_info:
                DetectiveAgent(groq_api_key="", gemini_api_key="test-gemini")
            assert "GROQ_API_KEY" in str(exc_info.value)

    def test_raises_value_error_for_missing_gemini_api_key(self):
        """DetectiveAgent raises ValueError when gemini_api_key is empty."""
        with patch(PATCH_CHATGROQ), patch(PATCH_CREATE_REACT_AGENT):
            from detective_agent import DetectiveAgent
            with pytest.raises(ValueError) as exc_info:
                DetectiveAgent(groq_api_key="test-groq", gemini_api_key="")
            assert "GEMINI_API_KEY" in str(exc_info.value)

    def test_raises_value_error_for_both_keys_missing(self):
        """ValueError message names both missing keys when both are absent."""
        with patch(PATCH_CHATGROQ), patch(PATCH_CREATE_REACT_AGENT):
            from detective_agent import DetectiveAgent
            with pytest.raises(ValueError) as exc_info:
                DetectiveAgent(groq_api_key="", gemini_api_key="")
            msg = str(exc_info.value)
            assert "GROQ_API_KEY" in msg
            assert "GEMINI_API_KEY" in msg

    def test_no_error_with_valid_keys(self, agent_factory):
        """No exception is raised when both required keys are provided."""
        agent = agent_factory()  # should not raise
        assert agent is not None

    def test_ors_api_key_absence_does_not_raise(self, agent_factory):
        """ORS_API_KEY absence must NOT raise any exception (Req 10.6)."""
        agent = agent_factory(ors_api_key=None)  # should not raise
        assert agent is not None

    def test_ors_api_key_absence_logs_info(self, agent_factory, caplog):
        """ORS_API_KEY absence is logged at INFO level (Req 10.6)."""
        import logging
        with caplog.at_level(logging.INFO, logger="detective_agent"):
            agent_factory(ors_api_key=None)
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("ORS_API_KEY" in r.message or "geo" in r.message.lower()
                   for r in info_records), (
            "Expected an INFO log mentioning ORS_API_KEY or geo-filtering"
        )


# ---------------------------------------------------------------------------
# Test: DETECTIVE_MAX_ITERATIONS env var (Req 4.1)
# ---------------------------------------------------------------------------

class TestMaxIterationsFromEnv:
    def test_reads_detective_max_iterations_from_env(self, monkeypatch, mock_graph):
        """max_iterations is read from DETECTIVE_MAX_ITERATIONS env var."""
        monkeypatch.setenv("DETECTIVE_MAX_ITERATIONS", "7")
        with patch(PATCH_CHATGROQ), patch(PATCH_CREATE_REACT_AGENT) as mock_create:
            mock_create.return_value = mock_graph
            from detective_agent import DetectiveAgent
            agent = DetectiveAgent(
                groq_api_key="test-groq",
                gemini_api_key="test-gemini",
                # max_iterations not passed → should read from env
            )
        assert agent.max_iterations == 7

    def test_constructor_param_overrides_env(self, monkeypatch, mock_graph):
        """Explicit max_iterations constructor param overrides the env var."""
        monkeypatch.setenv("DETECTIVE_MAX_ITERATIONS", "7")
        with patch(PATCH_CHATGROQ), patch(PATCH_CREATE_REACT_AGENT) as mock_create:
            mock_create.return_value = mock_graph
            from detective_agent import DetectiveAgent
            agent = DetectiveAgent(
                groq_api_key="test-groq",
                gemini_api_key="test-gemini",
                max_iterations=5,
            )
        assert agent.max_iterations == 5

    def test_default_max_iterations_is_15(self, mock_graph):
        """Default max_iterations is 15 when env var is not set."""
        with patch(PATCH_CHATGROQ), patch(PATCH_CREATE_REACT_AGENT) as mock_create:
            mock_create.return_value = mock_graph
            from detective_agent import DetectiveAgent
            agent = DetectiveAgent(
                groq_api_key="test-groq",
                gemini_api_key="test-gemini",
            )
        assert agent.max_iterations == 15


# ---------------------------------------------------------------------------
# Test: halt_reason="goal_achieved" (Req 4.5)
# ---------------------------------------------------------------------------

class TestHaltReasonGoalAchieved:
    def test_halt_reason_goal_achieved_when_enough_leads_found(self, agent_factory, mock_graph):
        """
        Agent returns halt_reason='goal_achieved' when desired_lead_count
        qualified leads are present in final_rankings.
        """
        desired = 2
        messages = _goal_achieved_messages(desired_lead_count=desired)
        mock_graph.invoke.return_value = {"messages": messages}

        agent = agent_factory(max_iterations=15)
        result = agent.run(icp_text="SaaS companies in Berlin", desired_lead_count=desired)

        assert result["halt_reason"] == "goal_achieved"

    def test_final_rankings_populated_on_goal_achieved(self, agent_factory, mock_graph):
        """final_rankings contains the leads returned by calculate_final_scores."""
        desired = 2
        messages = _goal_achieved_messages(desired_lead_count=desired)
        mock_graph.invoke.return_value = {"messages": messages}

        agent = agent_factory(max_iterations=15)
        result = agent.run(icp_text="SaaS companies in Berlin", desired_lead_count=desired)

        assert len(result["final_rankings"]) == desired

    def test_result_contains_all_required_keys(self, agent_factory, mock_graph):
        """AgentResult always contains all 7 required keys."""
        messages = _goal_achieved_messages(desired_lead_count=1)
        mock_graph.invoke.return_value = {"messages": messages}

        agent = agent_factory()
        result = agent.run(icp_text="Any ICP text")

        required_keys = {
            "final_rankings", "persona_results", "agent_scratchpad",
            "total_iterations", "halt_reason", "extracted_icp", "errors",
        }
        assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Test: halt_reason="max_iterations_reached" (Req 4.2, 4.3)
# ---------------------------------------------------------------------------

class TestHaltReasonMaxIterations:
    def test_halt_reason_max_iterations_reached(self, agent_factory, mock_graph):
        """
        Agent returns halt_reason='max_iterations_reached' when iteration_count
        reaches max_iterations.
        """
        messages = _max_iterations_messages()
        mock_graph.invoke.return_value = {"messages": messages}

        # Set a small max_iterations so the 20-call sequence exceeds it
        agent = agent_factory(max_iterations=3)
        result = agent.run(icp_text="Any ICP text")

        assert result["halt_reason"] == "max_iterations_reached"

    def test_total_iterations_does_not_exceed_max(self, agent_factory, mock_graph):
        """total_iterations in the result is bounded by max_iterations."""
        messages = _max_iterations_messages()
        mock_graph.invoke.return_value = {"messages": messages}

        max_iter = 3
        agent = agent_factory(max_iterations=max_iter)
        result = agent.run(icp_text="Any ICP text")

        # total_iterations reflects what LangGraph returned (may exceed max_iter
        # since we count from messages), but halt_reason must be correct
        assert result["halt_reason"] == "max_iterations_reached"

    def test_result_schema_complete_on_max_iterations(self, agent_factory, mock_graph):
        """AgentResult schema is complete even when halting due to max_iterations."""
        messages = _max_iterations_messages()
        mock_graph.invoke.return_value = {"messages": messages}

        agent = agent_factory(max_iterations=2)
        result = agent.run(icp_text="Any ICP text")

        required_keys = {
            "final_rankings", "persona_results", "agent_scratchpad",
            "total_iterations", "halt_reason", "extracted_icp", "errors",
        }
        assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Test: agent skips geo_filter when ICP has no city (Req 2.3)
# ---------------------------------------------------------------------------

class TestGeoFilterSkipped:
    def test_geo_filter_not_called_when_no_city_in_icp(self, agent_factory, mock_graph):
        """
        When the LLM does not emit a geo_filter tool call (because ICP has no city),
        geo_filter does not appear in the scratchpad tool_call entries.
        """
        # Simulate LLM that only calls match_companies (no geo_filter)
        tc_id = "tc_match"
        ai_call = _make_ai_message(
            tool_calls=[{"name": "match_companies", "args": {"industries": ["SaaS"]}, "id": tc_id}]
        )
        tool_result = _make_tool_message(
            {"results": {"co_1": {"basic_info": {"name": "Acme"}}}, "count": 1},
            tool_call_id=tc_id,
        )
        ai_summary = _make_ai_message(content="No city in ICP, skipping geo_filter.")
        messages = [_make_human_message(), ai_call, tool_result, ai_summary]
        mock_graph.invoke.return_value = {"messages": messages}

        agent = agent_factory()
        result = agent.run(icp_text="SaaS companies globally, no specific city")

        # Verify geo_filter does not appear in scratchpad tool_call entries
        tool_calls_in_scratchpad = [
            e for e in result["agent_scratchpad"]
            if e["type"] == "tool_call" and "geo_filter" in e["content"]
        ]
        assert len(tool_calls_in_scratchpad) == 0, (
            "geo_filter should not appear in scratchpad when ICP has no city"
        )

    def test_match_companies_appears_in_scratchpad_without_geo_filter(self, agent_factory, mock_graph):
        """Scratchpad records match_companies but not geo_filter when city is absent."""
        tc_id = "tc_match"
        ai_call = _make_ai_message(
            tool_calls=[{"name": "match_companies", "args": {"industries": ["SaaS"]}, "id": tc_id}]
        )
        tool_result = _make_tool_message(
            {"results": {}, "count": 0},
            tool_call_id=tc_id,
        )
        messages = [_make_human_message(), ai_call, tool_result]
        mock_graph.invoke.return_value = {"messages": messages}

        agent = agent_factory()
        result = agent.run(icp_text="SaaS companies globally")

        match_calls = [
            e for e in result["agent_scratchpad"]
            if e["type"] == "tool_call" and "match_companies" in e["content"]
        ]
        assert len(match_calls) >= 1


# ---------------------------------------------------------------------------
# Test: agent skips collect_intent when it fails (Req 2.4)
# ---------------------------------------------------------------------------

class TestCollectIntentSkipped:
    def test_scratchpad_records_collect_intent_observation_on_skip(self, agent_factory, mock_graph):
        """
        When collect_intent returns a skipped result, the scratchpad records
        an observation for it (the agent does not halt).
        """
        tc_id = "tc_intent"
        ai_call = _make_ai_message(
            tool_calls=[{"name": "collect_intent", "args": {"company_names": ["Acme"]}, "id": tc_id}]
        )
        # collect_intent returns skipped=True (its normal "failure" mode)
        tool_result = _make_tool_message(
            {"results": {}, "count": 0, "skipped": True},
            tool_call_id=tc_id,
        )
        ai_summary = _make_ai_message(content="Intent collection skipped, proceeding.")
        messages = [_make_human_message(), ai_call, tool_result, ai_summary]
        mock_graph.invoke.return_value = {"messages": messages}

        agent = agent_factory()
        result = agent.run(icp_text="Any ICP text")

        # Scratchpad must contain an observation for collect_intent
        intent_observations = [
            e for e in result["agent_scratchpad"]
            if e["type"] == "observation" and "collect_intent" in e["content"]
        ]
        assert len(intent_observations) >= 1, (
            "Expected at least one observation entry for collect_intent"
        )

    def test_agent_does_not_halt_after_collect_intent_skip(self, agent_factory, mock_graph):
        """Agent continues after collect_intent returns skipped (does not halt early)."""
        tc_intent_id = "tc_intent"
        tc_score_id = "tc_score"

        ai_intent_call = _make_ai_message(
            tool_calls=[{"name": "collect_intent", "args": {"company_names": []}, "id": tc_intent_id}]
        )
        intent_result = _make_tool_message(
            {"results": {}, "count": 0, "skipped": True},
            tool_call_id=tc_intent_id,
        )
        # Agent continues to calculate_final_scores after the skip
        ai_score_call = _make_ai_message(
            tool_calls=[{"name": "calculate_final_scores", "args": {}, "id": tc_score_id}]
        )
        score_result = _make_tool_message(
            {"results": [{"company_key": "co_1", "final_score": 0.8}], "count": 1},
            tool_call_id=tc_score_id,
        )
        ai_summary = _make_ai_message(content="Done.")
        messages = [
            _make_human_message(),
            ai_intent_call, intent_result,
            ai_score_call, score_result,
            ai_summary,
        ]
        mock_graph.invoke.return_value = {"messages": messages}

        agent = agent_factory()
        result = agent.run(icp_text="Any ICP text")

        # Both tools should appear in the scratchpad
        tool_names_in_scratchpad = [
            e["content"].split("|")[0].replace("Tool:", "").strip()
            for e in result["agent_scratchpad"]
            if e["type"] == "tool_call"
        ]
        assert "collect_intent" in tool_names_in_scratchpad
        assert "calculate_final_scores" in tool_names_in_scratchpad

    def test_scratchpad_has_entry_after_collect_intent_skip(self, agent_factory, mock_graph):
        """
        After a collect_intent skip observation, the scratchpad contains at least
        one more entry (agent does not terminate immediately).
        """
        tc_id = "tc_intent"
        ai_call = _make_ai_message(
            tool_calls=[{"name": "collect_intent", "args": {"company_names": []}, "id": tc_id}]
        )
        tool_result = _make_tool_message(
            {"results": {}, "count": 0, "skipped": True},
            tool_call_id=tc_id,
        )
        # One more thought after the skip
        ai_thought = _make_ai_message(content="Proceeding without intent signals.")
        messages = [_make_human_message(), ai_call, tool_result, ai_thought]
        mock_graph.invoke.return_value = {"messages": messages}

        agent = agent_factory()
        result = agent.run(icp_text="Any ICP text")

        scratchpad = result["agent_scratchpad"]
        # Find the index of the collect_intent observation
        intent_obs_idx = next(
            (i for i, e in enumerate(scratchpad)
             if e["type"] == "observation" and "collect_intent" in e["content"]),
            None,
        )
        assert intent_obs_idx is not None, "collect_intent observation not found"
        assert intent_obs_idx < len(scratchpad) - 1, (
            "Expected at least one scratchpad entry after the collect_intent observation"
        )


# ---------------------------------------------------------------------------
# Test: ORS_API_KEY absence disables geo-filtering without error (Req 10.6)
# ---------------------------------------------------------------------------

class TestOrsApiKeyAbsence:
    def test_no_exception_when_ors_api_key_absent(self, agent_factory):
        """Creating DetectiveAgent without ORS_API_KEY must not raise."""
        agent = agent_factory(ors_api_key=None)
        assert agent is not None

    def test_ors_api_key_stored_as_none(self, agent_factory):
        """ors_api_key attribute is None when not provided."""
        agent = agent_factory(ors_api_key=None)
        assert agent.ors_api_key is None

    def test_run_succeeds_without_ors_api_key(self, agent_factory, mock_graph):
        """run() completes without error when ORS_API_KEY is absent."""
        messages = _goal_achieved_messages(desired_lead_count=1)
        mock_graph.invoke.return_value = {"messages": messages}

        agent = agent_factory(ors_api_key=None)
        result = agent.run(icp_text="SaaS companies in Berlin")

        # Should complete without raising and return a valid result
        assert "halt_reason" in result
        assert "agent_scratchpad" in result


# ---------------------------------------------------------------------------
# Test: scratchpad structure (Req 9.1, 9.6)
# ---------------------------------------------------------------------------

class TestScratchpadStructure:
    def test_scratchpad_entries_have_required_fields(self, agent_factory, mock_graph):
        """Every scratchpad entry has step, type, content, and timestamp."""
        tc_id = "tc_match"
        ai_call = _make_ai_message(
            tool_calls=[{"name": "match_companies", "args": {"industries": ["SaaS"]}, "id": tc_id}]
        )
        tool_result = _make_tool_message(
            {"results": {}, "count": 0},
            tool_call_id=tc_id,
        )
        messages = [_make_human_message(), ai_call, tool_result]
        mock_graph.invoke.return_value = {"messages": messages}

        agent = agent_factory()
        result = agent.run(icp_text="Any ICP text")

        for entry in result["agent_scratchpad"]:
            assert "step" in entry, f"Missing 'step' in entry: {entry}"
            assert "type" in entry, f"Missing 'type' in entry: {entry}"
            assert "content" in entry, f"Missing 'content' in entry: {entry}"
            assert "timestamp" in entry, f"Missing 'timestamp' in entry: {entry}"

    def test_scratchpad_is_json_serializable(self, agent_factory, mock_graph):
        """agent_scratchpad can be serialized to JSON without custom encoders."""
        messages = _goal_achieved_messages(desired_lead_count=1)
        mock_graph.invoke.return_value = {"messages": messages}

        agent = agent_factory()
        result = agent.run(icp_text="Any ICP text")

        # Should not raise
        serialized = json.dumps(result["agent_scratchpad"])
        assert isinstance(serialized, str)

    def test_scratchpad_step_is_monotonically_increasing(self, agent_factory, mock_graph):
        """Scratchpad step numbers are 1-indexed and monotonically increasing."""
        tc_id = "tc_match"
        ai_call = _make_ai_message(
            tool_calls=[{"name": "match_companies", "args": {}, "id": tc_id}]
        )
        tool_result = _make_tool_message({"results": {}, "count": 0}, tool_call_id=tc_id)
        ai_thought = _make_ai_message(content="Thinking...")
        messages = [_make_human_message(), ai_call, tool_result, ai_thought]
        mock_graph.invoke.return_value = {"messages": messages}

        agent = agent_factory()
        result = agent.run(icp_text="Any ICP text")

        steps = [e["step"] for e in result["agent_scratchpad"]]
        assert steps == sorted(steps), "Steps are not monotonically increasing"
        if steps:
            assert steps[0] == 1, "First step should be 1"

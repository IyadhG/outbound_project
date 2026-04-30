# 🕵️ Detective — Agentic B2B Lead Generation

Detective is an LLM-powered B2B lead-generation system built around a **ReAct agent** (Reason → Act → Observe → Reason). Given a natural-language Ideal Customer Profile (ICP), the agent dynamically decides which tools to call, in what order, and how to recover when results are empty or low-quality — no hardcoded pipeline steps.

## What's New in v3.0

The previous version used a fixed 6-step LangGraph pipeline where the LLM acted only as a classifier inside predetermined nodes. v3.0 replaces that with a true agentic loop:

- The LLM **decides** which tools to call and in what order
- The agent **retries** with broadened criteria when a tool returns empty results (up to 3 times per tool)
- The agent **stops early** once enough qualified leads are found, rather than always running all steps
- Every thought, tool call, and observation is recorded in a structured **agent scratchpad** for full auditability
- All existing `brain/` and `ranking/` modules are preserved unchanged — they're wrapped as agent tools

---

## Architecture

```
Entry Points
  app/scorer.py          ← FastAPI real-time single-lead scoring
  mcp_server/mcp_server.py ← MCP tools for external orchestrators
  main.py                ← CLI batch run

        ↓

DetectiveAgent  (detective_agent.py)
  └── ReAct Loop  (langgraph.prebuilt.create_react_agent)
        ├── extract_icp
        ├── match_companies
        ├── geo_filter          (optional — skipped if no city in ICP)
        ├── filter_companies
        ├── rank_companies
        ├── collect_intent      (optional — skipped if unavailable)
        ├── calculate_final_scores
        └── score_personas

        ↓

Existing Modules (unchanged)
  brain/icp_agent.py        brain/company_matcher.py
  brain/geo_agent.py        ranking/company_filter.py
  ranking/company_ranker.py ranking/final_scorer.py
  ranking/persona_ranker.py ranking/embedder.py
```

### ReAct Loop

```
User ICP text
     ↓
[Thought] LLM reasons about next step
     ↓
[Tool Call] Agent invokes a tool
     ↓
[Observation] Result recorded in scratchpad
     ↓
[Thought] LLM evaluates result, decides next action
     ↓  (loop until goal achieved or max_iterations reached)
AgentResult { final_rankings, persona_results, agent_scratchpad, ... }
```

---

## Project Structure

```
detective/
├── detective_agent.py          # DetectiveAgent — ReAct loop, AgentResult, ScratchpadEntry
├── agent_tools.py              # 8 @tool-decorated wrappers for brain/ and ranking/
├── persona_scorer.py           # Hybrid persona scorer (rule-based + LLM escalation)
├── detective_graph.py          # Legacy 6-step pipeline (kept as reference, not used)
├── main.py                     # CLI entry point
│
├── brain/                      # LLM agents (unchanged)
│   ├── icp_agent.py
│   ├── company_matcher.py
│   ├── geo_agent.py
│   └── schema.py
│
├── ranking/                    # Scoring & ranking modules (unchanged)
│   ├── company_filter.py
│   ├── company_ranker.py
│   ├── embedder.py
│   ├── final_scorer.py
│   └── persona_ranker.py
│
├── app/                        # FastAPI service (unchanged except scorer.py)
│   ├── server.py
│   ├── subscriber.py
│   ├── scorer.py               # Delegates to agent_tools wrappers
│   └── event_emitter.py
│
├── mcp_server/
│   └── mcp_server.py           # MCP tools — delegates to DetectiveAgent
│
├── tests/
│   ├── test_detective_agent.py
│   ├── test_persona_scorer.py
│   ├── test_properties.py      # Hypothesis property-based tests
│   └── test_scorer.py
│
├── requirements.txt
├── .env
└── Dockerfile
```

---

## Quick Start

### 1. Install dependencies

```bash
cd detective
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```env
# detective/.env

# Required
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key

# Optional — geo-filtering disabled when absent
ORS_API_KEY=your_openrouteservice_key

# Tuning (all have defaults)
DETECTIVE_MAX_ITERATIONS=15
DETECTIVE_LLM_MODEL=llama-3.1-8b-instant
QUALIFICATION_THRESHOLD=0.6
DETECTIVE_PERSONA_LLM_THRESHOLD=0.4
```

Get API keys:
- **Groq**: https://console.groq.com/
- **Gemini**: https://aistudio.google.com/app/apikey
- **ORS** (optional): https://openrouteservice.org/

### 3. Run

```bash
# CLI batch run
python main.py

# MCP server
python -m mcp_server.mcp_server

# Docker
docker-compose up detective
```

---

## Using DetectiveAgent Directly

```python
from detective_agent import DetectiveAgent

agent = DetectiveAgent(
    groq_api_key="...",
    gemini_api_key="...",
    ors_api_key=None,          # geo-filtering disabled
    max_iterations=15,
    qualification_threshold=0.6,
)

result = agent.run(
    icp_text="""
    SaaS companies with 50–500 employees in Germany and Austria.
    Target roles: VP of Sales, Head of Revenue.
    Must use modern cloud stack.
    """,
    desired_lead_count=10,
    output_name="my_run",
)

print(result["halt_reason"])        # "goal_achieved" or "max_iterations_reached"
print(len(result["final_rankings"])) # number of ranked companies
print(result["total_iterations"])    # ReAct cycles used

# Inspect the full reasoning trace
for entry in result["agent_scratchpad"]:
    print(f"[{entry['step']}] {entry['type']}: {entry['content']}")
```

### AgentResult schema

```python
{
    "final_rankings":   List[Dict],   # companies sorted by final_score desc
    "persona_results":  List[Dict],   # best persona per company
    "agent_scratchpad": List[{        # full reasoning trace
        "step":      int,
        "type":      "thought" | "tool_call" | "observation" | "error",
        "content":   str,
        "timestamp": str,             # ISO 8601 UTC
    }],
    "total_iterations": int,
    "halt_reason":      str,          # "goal_achieved" | "max_iterations_reached"
    "extracted_icp":    Dict,         # ICPAttributes.model_dump()
    "errors":           List[str],    # non-fatal errors
}
```

---

## Agent Tools

Eight tools are exposed to the LLM. Each wraps an existing `brain/` or `ranking/` class, handles serialization, and catches all exceptions — returning `{"error": ..., "results": [], "count": 0}` on failure so the agent can continue.

| Tool | Wraps | Notes |
|---|---|---|
| `extract_icp` | `ICPExtractionAgent.extract_icp_attributes` | Always the first call |
| `match_companies` | `CompanyMatcher.find_matching_companies` | Retried with broader industries on empty |
| `geo_filter` | `GeoAgent.filter_companies_by_proximity` | Skipped when `ORS_API_KEY` absent or no city in ICP |
| `filter_companies` | `CompanyFilter.filter_companies` | Retried with relaxed size/country on empty |
| `rank_companies` | `CompanyRanker.rank_companies` + `GeminiEmbedder` | Falls back to `similarity_score=0.5` on embedder failure |
| `collect_intent` | *(stub)* | Always returns `skipped=true`; agent continues without intent signals |
| `calculate_final_scores` | `FinalScorer.calculate_final_scores` | Combines similarity + intent boost |
| `score_personas` | `PersonaScorer.score` | Hybrid rule-based + LLM escalation |

### Retry behaviour

The agent retries a tool up to **3 times** when it returns empty results. Retry strategies per tool:

- `match_companies` — broaden the industry list with semantically related terms
- `filter_companies` — relax `company_size` by ±20%; if still empty, relax country to continent
- `rank_companies` — retry with the unfiltered matched set
- All others — retry with identical arguments (transient errors)

---

## Persona Scoring

`PersonaScorer` (`persona_scorer.py`) wraps `PersonaRanker` with a hybrid scoring strategy:

1. **Rule-based score** — always called via `PersonaRanker.score_persona()`
2. **LLM escalation** — triggered when `rule_score < 0.4` AND the job title contains no recognized seniority keyword (ceo, vp, director, head of, etc.)
3. **Score selection** — LLM score is used only when `|llm_score - rule_score| > 0.1`; otherwise the rule score is kept
4. **Fallback** — if `analyze_persona_with_llm()` raises, the rule score is returned and a WARNING is logged

---

## MCP Server

Three MCP tools are exposed via `mcp_server/mcp_server.py`. All signatures are unchanged from v2.0 — existing MCP clients require no updates.

| Tool | Description |
|---|---|
| `rank_lead` | Score a single company against an ICP (unchanged) |
| `detect_top_leads` | Run the full agent pipeline; returns ranked leads + `dynamic_graph` |
| `run_full_detective_pipeline` | Full pipeline with `agent_scratchpad` and optional `warning` field |

`run_full_detective_pipeline` response additions:
- `agent_scratchpad` — full reasoning trace as a JSON-serializable list
- `warning` — present when `halt_reason == "max_iterations_reached"`, indicating partial results

```bash
# Test with MCP Inspector
npx @modelcontextprotocol/inspector python mcp_server/mcp_server.py
```

---

## FastAPI Service

`app/scorer.py` exposes `score_single_lead()` for real-time single-lead scoring from the Redis subscriber. The function signature and return schema are unchanged; internally it now delegates to `filter_companies_tool`, `rank_companies_tool`, and `score_personas_tool` from `agent_tools.py`.

```python
result = await score_single_lead(
    payload=lead_ingested_event,
    icp_attributes=icp,
    icp_text="...",
    groq_api_key="...",
)
# Returns: { final_score, icp_match, filters_passed, similarity_score,
#            intent_boost, selected_persona, qualified_for_outreach, company_data }
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key for LLM calls |
| `GEMINI_API_KEY` | — | **Required.** Gemini API key for embeddings |
| `ORS_API_KEY` | — | Optional. OpenRouteService key for geo-filtering |
| `DETECTIVE_MAX_ITERATIONS` | `15` | Maximum ReAct loop cycles before halting |
| `DETECTIVE_LLM_MODEL` | `llama-3.1-8b-instant` | Groq model for the reasoning loop |
| `QUALIFICATION_THRESHOLD` | `0.6` | Minimum `final_score` for a qualified lead |
| `DETECTIVE_PERSONA_LLM_THRESHOLD` | `0.4` | Rule score below which LLM persona escalation is considered |

Constructor parameters override environment variables when explicitly passed.

---

## Running Tests

```bash
cd detective
GROQ_API_KEY=test GEMINI_API_KEY=test pytest tests/ -v
```

73 tests across 4 files:

| File | Tests | Covers |
|---|---|---|
| `tests/test_detective_agent.py` | 26 | Startup validation, halt reasons, env vars, scratchpad structure |
| `tests/test_persona_scorer.py` | 20 | Rule scoring, LLM escalation, fallback on exception |
| `tests/test_properties.py` | 11 | Hypothesis property-based tests for 11 correctness properties |
| `tests/test_scorer.py` | 16 | `score_single_lead` schema, delegation, performance |

All external API calls (Groq, Gemini, ORS) are mocked — tests are fast and cost-free.

---

## Docker

```bash
# Build and run the full pipeline
docker-compose build
docker-compose up detective

# MCP server only
docker-compose up detective-mcp
```

Required environment variables in `.env`:
```env
GROQ_API_KEY=...
GEMINI_API_KEY=...
```

---

## Troubleshooting

**`ValueError: Missing required environment variables: GROQ_API_KEY`**
Set `GROQ_API_KEY` in your `.env` file or shell environment. Both `GROQ_API_KEY` and `GEMINI_API_KEY` are required.

**Geo-filtering not running**
Add `ORS_API_KEY` to `.env`. Without it, `geo_filter` returns the input unchanged and logs an INFO message — this is expected behaviour, not an error.

**`halt_reason: "max_iterations_reached"` with partial results**
The agent hit the iteration cap before finding enough qualified leads. Increase `DETECTIVE_MAX_ITERATIONS` or broaden the ICP criteria.

**Groq rate limit (429)**
The default model (`llama-3.1-8b-instant`) is the most token-efficient option. Wait a few minutes for the rate limit to reset, or upgrade your Groq account.

**No companies matched**
The agent will automatically retry `match_companies` with broader industry terms. If all retries are exhausted, check that `merged_profiles/` contains company data and that the ICP industries are specific enough for the LLM to match against.

---

## Changelog

### v3.0 — Agentic Redesign
- Replaced hardcoded 6-step LangGraph pipeline with `DetectiveAgent` ReAct loop
- Added `agent_tools.py` — 8 `@tool`-decorated wrappers for all `brain/` and `ranking/` modules
- Added `persona_scorer.py` — hybrid rule-based + LLM persona scoring
- Added agent scratchpad — full JSON-serializable reasoning trace on every run
- Added retry logic — up to 3 retries per tool with strategy-specific broadening
- Added goal-directed termination — stops as soon as `desired_lead_count` qualified leads are found
- Updated `app/scorer.py` to delegate to tool wrappers
- Updated `mcp_server/mcp_server.py` to delegate to `DetectiveAgent`; added `agent_scratchpad` and `warning` fields to MCP responses
- 73 unit + property-based tests; all `brain/` and `ranking/` modules unchanged

### v2.0 — LangGraph Migration
- Migrated from procedural script to 6-step LangGraph state machine
- Added intent collection, geo-filtering, and persona ranking

### v1.0 — Initial Release
- ICP extraction and LLM-based company matching

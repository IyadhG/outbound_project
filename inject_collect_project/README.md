# inject_collect_project

Agentic B2B lead enrichment pipeline. Discovers companies via Apollo.io, enriches them through a three-gate decision process, collects intent signals, discovers decision-maker personas, stores versioned profiles in Neo4j, and emits `lead_ingested` events for downstream processing by the Detective module.

---

## What Is Actually Implemented

### Pipeline (`main_discovery.py`)

Entry point: `discover_and_inject(industry, location, limit)` — async function that processes all companies concurrently via `asyncio.gather`. Each company goes through `_process_company()`, which is now an **agentic orchestrator** with three decision gates.

**Full per-company flow:**

1. **Apollo Search** — `ApolloScraper.search_companies()` by industry + location
2. **Apollo Enrich** — `ApolloScraper.enrich_organization()` with subsidiary resolution (if HQ is outside the target country, recurses into `suborganizations` to find the local branch)
3. **Anti-collision** — assigns a synthetic `unknown_<name>` domain when no real domain exists
4. **[Gate 1] Entity Validation** — if domain is synthetic AND company name is valid, calls `ApifyEnricher.search_news()` to find the real domain from news results; corrects the domain before the Apify crawl
5. **Apify Website Crawl** — `ApifyEnricher.crawl_website()` runs `apify~website-content-crawler`
6. **Intent Signals** — `IntentCollector.collect()` runs three Apify actors independently:
   - `apify~google-search-scraper` → recent news
   - `apify~linkedin-jobs-scraper` → job postings count
   - `apify~website-content-crawler` → technology changes
7. **Data Merge** — Apollo + Apify merged (Apollo takes precedence)
8. **[Gate 2] Data Quality** — computes DQS via `compute_dqs()`, then routes:
   - DQS < 0.5 → invokes `SmartScraperAI` (Playwright + Gemini deep scrape, 120s timeout); merges result into profile; recomputes DQS
   - 0.5 ≤ DQS < 0.75 → proceeds, sets `worker_review_flag=True` in processing log
   - DQS ≥ 0.75 → proceeds normally
9. **Neo4j Write** — `db.import_merged_profiles()`, fallback to `db.bulk_import_companies()`
10. **[Gate 3] Persona Worthiness** — only runs persona cascade if DQS ≥ 0.5 AND at least one signal exists (job postings > 0, recent news, or employee count)
11. **Persona Discovery** (conditional) — `search_and_enrich()` via Serper → Hunter → Snov.io → Tomba → AeroLeads; stored via `db.import_personas()`
12. **Detective Format** — `DetectiveFormatter.format()` builds the structured payload including `processing_log`
13. **Event Emission** — A2A HTTP POST to WorkerModule; fallback to Redis pub/sub via `EventEmitter`
14. **Detective Scoring** — A2A call to Detective for ICP scoring

---

## Agentic Decision Gates

The pipeline is no longer deterministic. Each company's path through the pipeline depends on what the gates find.

### Gate 1 — Entity Validation (`_gate_entity_validation`)

Triggers only when the anti-collision step assigned a synthetic `unknown_*` domain AND the company name is recognisable. Uses `ApifyEnricher.search_news()` to find a real domain from news results, then replaces the synthetic domain before the Apify crawl runs.

| Condition | Action | Log action |
|---|---|---|
| Domain not synthetic | Skip gate entirely | (no entry) |
| Synthetic + valid name + news found | Replace domain with extracted netloc | `domain_corrected` |
| Synthetic + valid name + no results | Retain synthetic domain | `correction_failed` |
| Synthetic + valid name + exception | Retain synthetic domain | `search_failed` |

### Gate 2 — Data Quality (`_gate_data_quality`)

Runs after the Apollo + Apify merge. Computes DQS and routes to one of three paths.

| DQS range | Path | `worker_review_flag` | SmartScraperAI |
|---|---|---|---|
| < 0.5 (real domain) | `deep_scrape` | `false` | Called (120s timeout) |
| < 0.5 (synthetic domain) | `proceed_normal` | `false` | Skipped |
| 0.5 – 0.75 | `flag_for_review` | `true` | Not called |
| ≥ 0.75 | `proceed_normal` | `false` | Not called |

When `SmartScraperAI` runs, its Apollo Mirror JSON output is merged into the profile — only filling fields that are currently empty/null/"Non renseigné". DQS is recomputed after the merge.

### Gate 3 — Persona Worthiness (`_gate_persona_worthiness`)

Runs before the persona cascade. Skips the 4-API enrichment chain when the company doesn't have enough signal to justify the cost.

| Condition | Decision |
|---|---|
| DQS < 0.5 | `skip_personas` |
| DQS ≥ 0.5 AND no job postings AND no news AND no employee count | `skip_personas` |
| DQS ≥ 0.5 AND at least one signal present | `run_personas` |

---

## DQS Computation (`dqs_calculator.py`)

`compute_dqs(profile) -> float` — pure function, no I/O, no state.

| Signal | Field | Weight |
|---|---|---|
| Non-synthetic domain | `domain` (not `unknown_*`) | 0.20 |
| Company name | `name` | 0.10 |
| Industry | `industry` | 0.10 |
| Employee count | `estimated_num_employees` | 0.15 |
| Annual revenue | `annual_revenue` | 0.15 |
| Location country | `location.country` or `country` | 0.10 |
| LinkedIn URL | `linkedin_url` | 0.10 |
| Website URL | `website_url` | 0.10 |

`None`, `""`, and `"Non renseigné"` all count as absent. Returns a float in `[0.0, 1.0]`.

---

## Processing Log

Every emitted payload includes a `processing_log` — a list of structured entries recording every gate decision. This lets the Worker module correlate pipeline decisions with downstream conversion outcomes.

Each entry (produced by `make_log_entry()` in `processing_log.py`):

```json
{
  "gate": "entity_validation | data_quality | persona_worthiness",
  "timestamp": "2026-04-30T10:00:00.123456+00:00",
  "action": "domain_corrected | deep_scrape | flag_for_review | proceed_normal | run_personas | skip_personas | ...",
  "dqs_at_gate": 0.65
}
```

Gate-specific extra fields:

**Gate 1:** `trigger`, `result`

**Gate 2:** `dqs_before`, `path_taken`, `dqs_after`, `worker_review_flag`

**Gate 3:** `dqs`, `job_postings_count`, `has_news`, `has_employee_count`, `decision`

The `processing_log` appears in two places in the emitted A2A envelope:
- `payload.processing_log` — inside the Detective payload
- `metadata.processing_log` — at the envelope level for Worker consumption

---

## Project Structure

```
inject_collect_project/
├── main_discovery.py          # Agentic pipeline orchestrator (gates + _process_company)
├── dqs_calculator.py          # Pure DQS computation function
├── processing_log.py          # Log entry factory (make_log_entry)
├── apollo_scraper.py          # Apollo.io search + enrich client
├── apify_enricher.py          # Apify API client (website crawler + news scraper)
├── intent_collector.py        # Intent signal collection (news, jobs, tech changes)
├── smart_scraper_ai.py        # Playwright + Gemini deep scraper (used by Gate 2)
├── detective_formatter.py     # Builds Detective-ready payload (+ processing_log)
├── event_emitter.py           # Redis pub/sub with in-memory fallback
├── a2a_client.py              # A2A HTTP client (Worker + Detective + Writer)
├── persona_search_enrich.py   # Persona discovery + multi-API contact enrichment
├── database_manager.py        # Neo4j versioned import (companies + personas)
│
├── .well-known/agent.json     # A2A agent card
├── documentation/             # Outdated docs (describe old non-agentic flow)
│
├── merged_profiles/           # Output: Apollo + Apify fused JSON profiles
├── personas_discovered/       # Output: Enriched decision-maker contacts
├── scraped_data/              # Output: SmartScraperAI raw text + Apollo JSON
├── comparisons/               # Output: Data quality comparison reports
├── archive/                   # Old test scripts and CSV samples
└── tests/
    ├── test_dqs_calculator.py          # 28 unit tests for compute_dqs
    ├── test_agentic_gates.py           # 38 unit/integration tests for all gates
    ├── test_properties.py              # 7 Hypothesis property-based tests
    ├── test_preservation_properties.py # Regression tests (pipeline behavior preserved)
    └── test_bug_condition_exploration.py # Bug condition tests (pass on unfixed code)
```

---

## Agentic Data Flow

```
Apollo Search
    └─► Apollo Enrich (+ subsidiary resolution)
            └─► Anti-collision domain fix
                    └─► [Gate 1] Entity Validation
                    │       synthetic domain? → search_news → correct domain
                    │
                    └─► Apify Website Crawl
                            └─► Intent Signal Collection (news + jobs + tech)
                                    └─► Data Merge (Apollo takes precedence)
                                            └─► compute_dqs()
                                                    └─► [Gate 2] Data Quality
                                                    │       DQS < 0.5  → SmartScraperAI → merge → recompute DQS
                                                    │       DQS < 0.75 → flag_for_review
                                                    │       DQS ≥ 0.75 → proceed_normal
                                                    │
                                                    └─► Neo4j Write
                                                            └─► [Gate 3] Persona Worthiness
                                                            │       DQS ≥ 0.5 + signal? → run personas
                                                            │       otherwise → skip
                                                            │
                                                            └─► Persona cascade (conditional)
                                                                    └─► DetectiveFormatter.format(processing_log)
                                                                            └─► A2A → WorkerModule (fallback: Redis)
                                                                                    └─► A2A → Detective (ICP scoring)
```

---

## Detective Payload Schema

```json
{
  "company_id": "<uuid>",
  "correlation_id": "<uuid>",
  "company_data": {
    "name": "Example Inc.",
    "domain": "example.com",
    "industry": "Technology",
    "location": { "city": "Berlin", "country": "Germany" },
    "website_url": "https://example.com",
    "linkedin_url": "https://linkedin.com/company/example",
    "estimated_num_employees": "500",
    "annual_revenue": "$50M",
    "founded_year": "2005"
  },
  "enrichment_data": {
    "data_quality_score": 0.87,
    "confidence_scores": {}
  },
  "personas": [
    {
      "name": "Jane Doe",
      "title": "VP Sales",
      "email": "jane@example.com",
      "phone": "+49 30 123456",
      "linkedin_url": "https://linkedin.com/in/janedoe",
      "enrichment_level": ""
    }
  ],
  "intent_signals": {
    "recent_news": [{ "title": "...", "url": "..." }],
    "job_postings_count": 12,
    "technology_changes": ["Kubernetes", "Terraform"]
  },
  "readiness_flags": {
    "has_valid_contact": true,
    "data_completeness": 0.87,
    "ready_for_outreach": true
  },
  "processing_log": [
    {
      "gate": "entity_validation",
      "timestamp": "2026-04-30T10:00:00.123456+00:00",
      "action": "domain_corrected",
      "dqs_at_gate": 0.0,
      "trigger": "synthetic_domain",
      "result": "example.com"
    },
    {
      "gate": "data_quality",
      "timestamp": "2026-04-30T10:00:01.456789+00:00",
      "action": "deep_scrape",
      "dqs_at_gate": 0.70,
      "dqs_before": 0.30,
      "path_taken": "deep_scrape",
      "dqs_after": 0.70,
      "worker_review_flag": false
    },
    {
      "gate": "persona_worthiness",
      "timestamp": "2026-04-30T10:00:02.789012+00:00",
      "action": "run_personas",
      "dqs_at_gate": 0.70,
      "dqs": 0.70,
      "job_postings_count": 12,
      "has_news": true,
      "has_employee_count": true,
      "decision": "run_personas"
    }
  ],
  "event_type": "lead_ingested",
  "timestamp": "2026-04-30T10:00:03.000000Z"
}
```

---

## Neo4j Data Model

```
(:Company {domain, name, apollo_id, uid})
    -[:CURRENT]->  (:Version {captured_at, name, industry, revenue, technologies, ...})
    -[:ARCHIVED]-> (:Version {...})          # historical snapshots
    -[:OWNS]->     (:Company)               # parent → subsidiary
    -[:OWNED_BY]-> (:Company)               # subsidiary → parent

(:Persona {linkedin_url, full_name, title, email, phone, ...})
    -[:WORKS_AT]-> (:Company)
```

Two write paths in `database_manager.py`:
- `import_merged_profiles()` — dynamic `SET new_v += data` merge (primary)
- `bulk_import_companies()` — explicit field-by-field Cypher (fallback)

Both implement versioning: a new `Version` node is only created when data actually changes.

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
playwright install chromium   # required for SmartScraperAI (Gate 2)
```

### 2. Configure environment

Copy `.env.example` to `.env`:

```env
# Neo4j
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=your_user
NEO4J_PASSWORD=your_password

# Apify (website crawl + intent signals + Gate 1 domain correction)
APIFY_API_KEY=your_key

# Gemini (required for SmartScraperAI — Gate 2 deep scrape)
GEMINI_API_KEY=your_key

# Redis (optional — falls back to in-memory queue if unavailable)
REDIS_URL=redis://localhost:6379

# Downstream services
WORKER_A2A_URL=http://api:8000
DETECTIVE_A2A_URL=http://detective:8002

# Persona enrichment
SERPER_API_KEY=your_key
HUNTER_API_KEY=your_key
SNOVIO_CLIENT_ID=your_id
SNOVIO_CLIENT_SECRET=your_secret
TOMBA_API_KEY=your_key
TOMBA_API_SECRET=your_secret
AEROLEADS_API_KEY=your_key
```

> **Note**: `APOLLO_API_KEY`, `APIFY_API_KEY`, and several persona enrichment keys are currently hardcoded in their respective modules. Move them to `.env` before deploying.

### 3. Run

```python
import asyncio
from main_discovery import discover_and_inject

asyncio.run(discover_and_inject(
    industry="IT",
    location="France",
    limit=5,
))
```

---

## Key Components

### `dqs_calculator.py` — `compute_dqs(profile) -> float`

Pure function. Weighted sum of 8 presence signals. Returns `[0.0, 1.0]`. Used by Gate 2 and Gate 3.

### `processing_log.py` — `make_log_entry(gate, action, dqs_at_gate, **extra) -> dict`

Pure factory function. Creates structured log entries with ISO 8601 UTC timestamp. Used by all three gates.

### `main_discovery.py` — Gate functions

- `_gate_entity_validation(domain, company_name, apify_enricher, processing_log)` — async, Gate 1
- `_gate_data_quality(merged_profile, domain, target_location, smart_scraper, processing_log)` — async, Gate 2
- `_merge_ai_result(merged_profile, ai_json_path)` — in-place merge of SmartScraperAI output
- `_gate_persona_worthiness(merged_profile, intent_signals, dqs, processing_log)` — sync, Gate 3

### `smart_scraper_ai.py` — SmartScraperAI

Playwright + Gemini deep scraper. Instantiated once in `discover_and_inject` and shared across all concurrent company tasks. Called by Gate 2 when DQS < 0.5. Has its own internal agentic loop: scrape → identify missing fields → navigate to sub-pages → re-extract (up to depth 2). Returns path to Apollo Mirror JSON file.

### `apollo_scraper.py` — ApolloScraper

- `search_companies(industries, locations, limit)` — POST to `/v1/organizations/search`
- `enrich_organization(domain, target_location)` — GET to `/v1/organizations/enrich`; recurses into `suborganizations` for subsidiary resolution

### `apify_enricher.py` — ApifyEnricher

Async Apify REST client. Start → poll → fetch dataset pattern (60s default timeout).

- `crawl_website(domain)` — `apify~website-content-crawler`
- `search_news(company_name)` — `apify~google-search-scraper`

### `intent_collector.py` — IntentCollector

Three independent signals (each failure isolated): news via `search_news()`, job postings via `apify~linkedin-jobs-scraper`, tech changes via `crawl_website()`.

### `detective_formatter.py` — DetectiveFormatter

`format(merged_profile, personas, intent_signals, processing_log=None)` — builds the Detective-ready payload. `processing_log` defaults to `[]` for backward compatibility.

### `event_emitter.py` — EventEmitter

Redis `PUBLISH lead_ingested <json>`. Falls back to `asyncio.Queue` if Redis is unavailable at startup.

### `a2a_client.py` — A2AClient

- `send_lead_ingested(envelope)` — POST to `WORKER_A2A_URL/tasks/send`; falls back to `EventEmitter`
- `send_to_detective(envelope)` — POST to `DETECTIVE_A2A_URL/tasks/send` with `skill: "score_lead"`

### `persona_search_enrich.py` — `search_and_enrich()`

Serper Google search → Hunter → Snov.io → Tomba → AeroLeads cascade. AeroLeads always runs for deep profile data. Saves to `personas_discovered/<domain>_personas.json`.

### `database_manager.py` — Neo4jManager

Credentials hardcoded (AuraDB). `import_merged_profiles()`, `bulk_import_companies()`, `import_personas()`.

---

## Event Emission

```json
{
  "event_id": "<uuid>",
  "correlation_id": "<uuid>",
  "module": "inject",
  "event_type": "lead_ingested",
  "timestamp": "<iso8601>",
  "payload": { "<detective_payload_with_processing_log>" },
  "metadata": { "processing_log": [ "<same_list>" ] }
}
```

Primary: A2A HTTP POST to WorkerModule. Fallback: Redis `PUBLISH lead_ingested`.

---

## Tests

```bash
pytest tests/ -v --ignore=tests/test_bug_condition_exploration.py
```

| File | Type | Count | What it covers |
|---|---|---|---|
| `test_dqs_calculator.py` | Unit | 28 | All 8 DQS signals, empty values, synthetic domain, location fallback, clamping |
| `test_agentic_gates.py` | Unit + Integration | 38 | All three gates, `_merge_ai_result`, `DetectiveFormatter` backward compat |
| `test_properties.py` | Property-based (Hypothesis) | 7 | DQS range, idempotence, weighted sum, log entry fields, merge semantics, gate logic, payload round-trip |
| `test_preservation_properties.py` | Regression | 6 | Apollo call signature, Neo4j write shape, persona cascade args, anti-collision, subsidiary resolution |
| `test_bug_condition_exploration.py` | Bug condition | 6 | Pass on unfixed code, fail after fixes (designed to fail) |

---

## Known Issues / Technical Debt

| Issue | Location | Notes |
|---|---|---|
| API keys hardcoded | `apollo_scraper.py`, `apify_enricher.py`, `persona_search_enrich.py`, `database_manager.py` | Should be read from env vars |
| `persona_search_enrich.py` uses synchronous `requests` | `persona_search_enrich.py` | Blocks the event loop; should use `asyncio.to_thread` |
| `documentation/` READMEs describe old non-agentic flow | `documentation/` | Outdated; do not reflect current implementation |
| `datetime.utcnow()` deprecated | `detective_formatter.py` | Should use `datetime.now(timezone.utc)` |

---

## Dependencies

| Library | Purpose |
|---|---|
| `requests` | Apollo.io API + persona enrichment APIs (sync) |
| `httpx` | Async HTTP client for Apify API + A2A calls |
| `redis` | Event emission via pub/sub |
| `neo4j` | Graph database driver |
| `python-dotenv` | Environment variable loading |
| `playwright` | Browser automation for SmartScraperAI (Gate 2) |
| `google-genai` | Gemini LLM for SmartScraperAI (Gate 2) |
| `PyMuPDF`, `pillow` | PDF + image processing for SmartScraperAI (Gate 2) |
| `hypothesis` | Property-based testing |

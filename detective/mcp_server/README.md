# 🕵️ Detective MCP Server - Detective Work Tools

MCP (Model Context Protocol) server providing **9 detective tools** that DO the actual work - extract ICP, match companies, rank by similarity, score with intent boost, and rank personas.

## 🚀 Quick Start

```bash
# Test all 9 tools
python test_all_tools.py

# Test specific tools
python test_simple.py
```

## 📋 The 9 Detective Tools

### 1. extract_icp
**Extract ICP attributes from natural language text using LLM**

```python
result = await client.call_tool("extract_icp", {
    "icp_text": """I want IT and SaaS companies with 50-500 employees 
    in United States, Canada, United Kingdom, and Germany.
    Target roles: Sales Managers, CTOs, and Heads of Product."""
})

# Output:
{
  "success": true,
  "icp_attributes": {
    "industry": ["IT", "SaaS"],
    "company_size": {"min": 50, "max": 500},
    "target_countries": ["United States", "Canada", "United Kingdom", "Germany"],
    "target_roles": ["Sales Manager", "CTO", "Head of Product"],
    "tech_stack": ["React", "Python", "AWS"],
    "funding_stage": "Series B"
  }
}
```

### 2. match_companies
**Find companies matching ICP industries using LLM**

```python
result = await client.call_tool("match_companies", {
    "icp_attributes": {"industry": ["IT", "SaaS"]},
    "companies_folder": "../inject_collect_project/merged_profiles"
})

# Output:
{
  "success": true,
  "total_matched": 15,
  "target_industries": ["IT", "SaaS"],
  "matched_companies": ["bosch_us", "alten_de", "nhs_uk", ...]
}
```

### 3. filter_by_employees
**Filter companies by employee count range**

```python
companies = {
    "company_1": {"basic_info": {"name": "Small Co", "employees": 25}},
    "company_2": {"basic_info": {"name": "Mid Co", "employees": 150}},
    "company_3": {"basic_info": {"name": "Large Co", "employees": 5000}}
}

result = await client.call_tool("filter_by_employees", {
    "companies": companies,
    "min_employees": 50,
    "max_employees": 500
})

# Output:
{
  "success": true,
  "input_count": 3,
  "filtered_count": 1,
  "range": "50.0 - 500.0",
  "kept_companies": ["company_2"]
}
```

### 4. filter_by_country
**Filter companies by target countries**

```python
companies = {
    "us_co": {"basic_info": {"name": "US Tech", "country": "United States"}},
    "de_co": {"basic_info": {"name": "DE Tech", "country": "Germany"}},
    "fr_co": {"basic_info": {"name": "FR Tech", "country": "France"}}
}

result = await client.call_tool("filter_by_country", {
    "companies": companies,
    "target_countries": ["United States", "Germany"]
})

# Output:
{
  "success": true,
  "input_count": 3,
  "filtered_count": 2,
  "target_countries": ["United States", "Germany"],
  "kept_companies": ["us_co", "de_co"]
}
```

### 5. rank_by_similarity
**Rank companies by ICP similarity using Gemini embeddings**

```python
result = await client.call_tool("rank_by_similarity", {
    "icp_text": "IT and SaaS companies with cloud solutions for enterprise",
    "companies": {...}
})

# Output:
{
  "success": true,
  "total_ranked": 10,
  "ranked_companies": [
    {"company_key": "bosch_us", "company_name": "Bosch", "similarity_score": 0.85},
    {"company_key": "alten_de", "company_name": "ALTEN", "similarity_score": 0.72}
  ]
}
```

### 6. calculate_final_scores
**Calculate final scores with intent boost**

```python
ranked_companies = [
    {"company_key": "c1", "company_name": "Company 1", "similarity_score": 0.75},
    {"company_key": "c2", "company_name": "Company 2", "similarity_score": 0.68}
]

intent_signals = {
    "c1": {"funding": {"confidence": 0.8}, "news": {"confidence": 0.6}},
    "c2": {"funding": {"confidence": 0.3}}
}

result = await client.call_tool("calculate_final_scores", {
    "ranked_companies": ranked_companies,
    "intent_signals": intent_signals,
    "intent_boost": 0.05
})

# Output:
{
  "success": true,
  "total_scored": 2,
  "intent_boost_factor": 0.05,
  "ranked_companies": [
    {
      "company_key": "c1",
      "company_name": "Company 1",
      "similarity_score": 0.75,
      "intent_boost": 0.05,
      "final_score": 0.80,
      "rank": 1
    }
  ]
}
```

### 7. rank_personas
**Rank and select best personas per company using LLM**

```python
companies = [
    {"company_key": "bosch_us", "company_name": "Bosch in the USA"}
]

result = await client.call_tool("rank_personas", {
    "companies": companies,
    "target_roles": ["Sales Manager", "CTO"],
    "personas_folder": "../inject_collect_project/personas_discovered"
})

# Output:
{
  "success": true,
  "total_companies": 1,
  "target_roles": ["Sales Manager", "CTO"],
  "persona_selections": [
    {
      "company_key": "bosch_us",
      "company_name": "Bosch in the USA",
      "selected_persona": {
        "full_name": "Jessica Katterheinrich",
        "job_title": "Sales Manager",
        "email": "jessica.katterheinrich@bosch.com",
        "linkedin": "https://linkedin.com/in/...",
        "city": "Detroit",
        "country": "United States"
      },
      "persona_score": {
        "final_score": 0.93,
        "is_sales": true
      }
    }
  ]
}
```

### 8. geo_filter
**Filter companies by proximity to a city (requires ORS_API_KEY)**

```python
companies = {
    "berlin_tech": {"basic_info": {"name": "Berlin Tech", "city": "Berlin", "country": "Germany"}},
    "munich_tech": {"basic_info": {"name": "Munich Tech", "city": "Munich", "country": "Germany"}},
    "hamburg_tech": {"basic_info": {"name": "Hamburg Tech", "city": "Hamburg", "country": "Germany"}}
}

result = await client.call_tool("geo_filter", {
    "companies": companies,
    "target_city": "Berlin",
    "target_country": "Germany",
    "range_km": 200
})

# Output:
{
  "success": true,
  "input_count": 3,
  "filtered_count": 2,
  "target_city": "Berlin",
  "range_km": 200,
  "kept_companies": ["berlin_tech", "hamburg_tech"]
}
```

### 9. analyze_company_profile
**Analyze company profile with LLM for insights**

```python
company_profile = {
    "basic_info": {
        "name": "TechCorp Solutions",
        "description": "Enterprise cloud software provider...",
        "country": "United States",
        "employees": 500
    },
    "classification": {
        "industries": ["Software", "SaaS"],
        "specialties": ["Cloud", "AI/ML"]
    }
}

result = await client.call_tool("analyze_company_profile", {
    "company_profile": company_profile,
    "analysis_type": "summary"  # or "icp_fit" or "pain_points"
})

# Output:
{
  "success": true,
  "company": "TechCorp Solutions",
  "analysis_type": "summary",
  "analysis": "TechCorp Solutions is a mid-size SaaS company..."
}
```

## 🧪 Testing

### Test All Tools
```bash
python test_all_tools.py
```

This will test all 9 tools with real examples and show you which ones need API keys.

### Test Specific Tools
```bash
python test_simple.py
```

Tests the 4 tools that work without API keys:
- filter_by_employees
- filter_by_country
- calculate_final_scores
- extract_icp (if Groq key configured)

## 🔌 MCP Client Usage

### From Python with MCP Client
```python
from mcp_client import MCPClient

client = MCPClient()
await client.connect_to_server("mcp_server/mcp_server.py")

# Extract ICP
result = await client.call_tool("extract_icp", {
    "icp_text": "I want IT companies..."
})

# Match companies
matches = await client.call_tool("match_companies", {
    "icp_attributes": {"industry": ["IT", "SaaS"]}
})

# Filter results
filtered = await client.call_tool("filter_by_employees", {
    "companies": {...},
    "min_employees": 50,
    "max_employees": 500
})
```

### With Inspector
```bash
npx @modelcontextprotocol/inspector python mcp_server.py
```

## 📊 API Key Requirements

| Tool | API Key Required |
|------|------------------|
| extract_icp | GROQ_API_KEY |
| match_companies | GROQ_API_KEY |
| filter_by_employees | None |
| filter_by_country | None |
| rank_by_similarity | GOOGLE_API_KEY |
| calculate_final_scores | None |
| rank_personas | GROQ_API_KEY |
| geo_filter | ORS_API_KEY |
| analyze_company_profile | GROQ_API_KEY |

## 🔧 Environment Setup

Create `.env` file in `detective/`:
```env
# Required for LLM tools
GROQ_API_KEY=your_groq_api_key

# Required for embeddings
GOOGLE_API_KEY=your_google_api_key

# Required for geo-filtering
ORS_API_KEY=your_openrouteservice_key
```

Get API keys:
- **Groq**: https://console.groq.com/
- **Google**: https://aistudio.google.com/app/apikey
- **ORS**: https://openrouteservice.org/

## 📁 Files

```
mcp_server/
├── mcp_server.py         # Main server with 9 tools
├── test_all_tools.py     # Comprehensive test of all tools
├── test_simple.py        # Basic test
├── README.md             # This file
└── __init__.py           # Package init
```

## 🎯 Example Workflow

Complete pipeline from ICP to personas:

```python
# Step 1: Extract ICP
icp_result = await client.call_tool("extract_icp", {"icp_text": "..."})
icp = icp_result["icp_attributes"]

# Step 2: Match companies
matches = await client.call_tool("match_companies", {
    "icp_attributes": icp
})

# Step 3: Filter by employees
filtered = await client.call_tool("filter_by_employees", {
    "companies": matches,
    "min_employees": icp["company_size"]["min"],
    "max_employees": icp["company_size"]["max"]
})

# Step 4: Filter by country
filtered = await client.call_tool("filter_by_country", {
    "companies": filtered,
    "target_countries": icp["target_countries"]
})

# Step 5: Rank by similarity
ranked = await client.call_tool("rank_by_similarity", {
    "icp_text": "...",
    "companies": filtered
})

# Step 6: Calculate final scores
scored = await client.call_tool("calculate_final_scores", {
    "ranked_companies": ranked,
    "intent_signals": {...}
})

# Step 7: Rank personas
personas = await client.call_tool("rank_personas", {
    "companies": scored[:10],
    "target_roles": icp["target_roles"]
})
```

## 📝 Notes

- Tools that don't need API keys work immediately
- LLM tools use `llama-3.1-8b-instant` (cheapest Groq model)
- Embeddings use Gemini `models/embedding-001`
- All tools return JSON with `success` field
- Errors include helpful messages in `error` field

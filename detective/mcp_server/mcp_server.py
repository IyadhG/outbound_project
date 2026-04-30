#!/usr/bin/env python3
"""
Detective MCP Server - B2B Lead Detective & Orchestrator

High-level orchestrator tools for B2B lead generation:
- rank_lead: Analyze single company against ICP
- detect_top_leads: Full pipeline from ICP extraction to ranked leads with graph
- run_full_detective_pipeline: Complete agentic pipeline via DetectiveAgent

Usage:
    python mcp_server.py
"""

import io
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# 1. FIX THE PATHS FIRST
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also try one level up (in case we're in a different structure)
parent_of_project = project_root.parent
if str(parent_of_project) not in sys.path:
    sys.path.insert(0, str(parent_of_project))

# 2. LOAD ENVIRONMENT
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# 3. IMPORTS
try:
    from mcp.server.fastmcp import FastMCP
    _fastmcp_available = True
except ImportError:
    try:
        from fastmcp import FastMCP
        _fastmcp_available = True
    except ImportError:
        _fastmcp_available = False
        FastMCP = None  # type: ignore[assignment,misc]

# Import detective components (from parent directory)
from brain.icp_agent import ICPExtractionAgent

# Import the DetectiveAgent (replaces detective_graph.run_detective_pipeline)
try:
    from detective_agent import DetectiveAgent
except ImportError as e:
    DetectiveAgent = None  # type: ignore[assignment,misc]

# 4. INITIALIZE SERVER
if _fastmcp_available and FastMCP is not None:
    mcp = FastMCP("B2B-Detective-Server")
else:
    # Stub for environments where fastmcp is not installed (e.g., tests)
    class _McpStub:
        """Minimal stub so the module can be imported without fastmcp."""
        def tool(self):
            def decorator(fn):
                return fn
            return decorator
        def run(self, **kwargs):
            raise RuntimeError("fastmcp is not installed; cannot run MCP server.")
    mcp = _McpStub()  # type: ignore[assignment]

# Initialize API key references (read at call time, not at import time)
def _get_api_keys():
    """Read API keys from environment at call time."""
    return {
        "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        "groq": os.getenv("GROQ_API_KEY"),
        "ors": os.getenv("ORS_API_KEY"),
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_companies_from_folder(folder_path: str) -> dict:
    """Load company profiles from JSON files."""
    companies = {}
    
    # Resolve path - handle relative paths from mcp_server location
    current_dir = Path(__file__).parent
    project_root_local = current_dir.parent
    
    # Extract the folder name from the path
    folder_name = Path(folder_path).name
    parent_hint = Path(folder_path).parent.name if Path(folder_path).parent.name != "." else ""
    
    # Try multiple path resolutions
    paths_to_try = [
        Path(folder_path),  # As-is
        current_dir / folder_path,  # Relative to mcp_server
        project_root_local / folder_path.lstrip("./").replace("../", ""),  # Relative to detective
        project_root_local.parent / folder_path.lstrip("./").replace("../", ""),  # Relative to outbound_project
        # Hardcoded fallback to known location
        project_root_local.parent / "inject_collect_project" / "merged_profiles",
        Path("c:/Users/Dell/Desktop/outbound_project/inject_collect_project/merged_profiles"),
    ]
    
    # Also search for the folder by name in parent directories
    search_dir = current_dir
    for _ in range(4):  # Search up 4 levels
        potential = search_dir / folder_name
        if potential.exists() and potential not in paths_to_try:
            paths_to_try.append(potential)
        # Also check for inject_collect_project pattern
        if parent_hint:
            alt = search_dir / parent_hint / folder_name
            if alt.exists() and alt not in paths_to_try:
                paths_to_try.insert(0, alt)
        search_dir = search_dir.parent
    
    folder = None
    for p in paths_to_try:
        try:
            if p.exists():
                folder = p
                break
        except Exception:
            continue
    
    if not folder:
        return companies
    
    json_files = list(folder.glob("*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                company_key = file_path.stem
                companies[company_key] = data
        except Exception:
            pass
    
    return companies


def extract_icp_attributes(raw_query: str) -> dict:
    """Extract ICP attributes from natural language."""
    try:
        from groq import Groq
        groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        extractor = ICPExtractionAgent(groq_client)
        icp = extractor.extract_icp_attributes(raw_query)
        return icp.model_dump()
    except Exception as e:
        # Fallback to basic extraction
        return {
            "industry": [],
            "company_size": {"min": 0, "max": 100000},
            "target_countries": [],
            "target_roles": [],
            "tech_stack": [],
            "funding_stage": [],
            "keywords": []
        }


def rank_company_against_icp(company: dict, icp: dict, user_offering: str, signals: list = None) -> dict:
    """Rank a single company against ICP criteria."""
    scores = {
        "industry_match": 0,
        "size_match": 0,
        "location_match": 0,
        "intent_boost": 0,
        "total_score": 0,
        "reason": ""
    }
    
    # Industry match - handle both nested and flat structure
    company_industries_raw = company.get("classification", {}).get("industries") or company.get("industry", "")
    if isinstance(company_industries_raw, str):
        company_industries = [i.strip() for i in company_industries_raw.split(",")]
    else:
        company_industries = company_industries_raw or []
    
    target_industries = icp.get("industry", [])
    if target_industries and company_industries:
        matches = sum(1 for i in company_industries if any(ti.lower() in i.lower() for ti in target_industries))
        scores["industry_match"] = (matches / len(target_industries)) * 30
    
    # Size match - handle both nested and flat structure
    employees_raw = company.get("basic_info", {}).get("employees") or company.get("employees") or company.get("estimated_num_employees", 0)
    # Convert string like "412,800 (Global)" to number
    if isinstance(employees_raw, str):
        import re
        nums = re.findall(r"[\d,]+", employees_raw)
        if nums:
            employees = int(nums[0].replace(",", ""))
        else:
            employees = 0
    else:
        employees = employees_raw or 0
    
    size_range = icp.get("company_size", {})
    min_size = size_range.get("min") or 0
    max_size = size_range.get("max") or 100000
    if min_size <= employees <= max_size:
        scores["size_match"] = 25
    elif employees > 0:
        # Partial score based on proximity
        if employees < min_size:
            scores["size_match"] = max(0, 25 * (employees / min_size))
        else:
            scores["size_match"] = max(0, 25 * (max_size / employees))
    
    # Location match - handle both nested and flat structure
    company_country = company.get("basic_info", {}).get("country") or company.get("country", "")
    target_countries = icp.get("target_countries", [])
    if target_countries and any(tc.lower() in company_country.lower() for tc in target_countries):
        scores["location_match"] = 20
    
    # Intent boost from signals
    if signals:
        for signal in signals:
            confidence = signal.get("confidence", 0)
            relevance = signal.get("relevance", 0)
            scores["intent_boost"] += confidence * relevance * 0.25
    
    scores["intent_boost"] = min(25, scores["intent_boost"])
    
    # Total score
    scores["total_score"] = round(
        scores["industry_match"] + scores["size_match"] + scores["location_match"] + scores["intent_boost"], 2
    )
    
    # Generate reason
    reasons = []
    if scores["industry_match"] > 20:
        reasons.append("Strong industry fit")
    if scores["size_match"] > 20:
        reasons.append("Ideal company size")
    if scores["location_match"] > 15:
        reasons.append("Target location")
    if scores["intent_boost"] > 10:
        reasons.append("High intent signals")
    
    scores["reason"] = "; ".join(reasons) if reasons else "Moderate fit"
    
    return scores


def _build_dynamic_graph_from_scratchpad(agent_scratchpad: list, final_rankings: list) -> dict:
    """
    Build a dynamic graph (nodes + edges) from AgentResult data.

    Uses final_rankings to build company and persona nodes. The scratchpad
    is available for future enrichment but is not required for the graph.
    """
    graph_nodes = []
    graph_edges = []

    for lead in final_rankings:
        comp_id = f"node_{lead.get('company_key', 'unknown')}"

        # Add Company Node
        graph_nodes.append({
            "id": comp_id,
            "type": "company",
            "label": lead.get("company_name", lead.get("company_key", "Unknown")),
            "score": lead.get("final_score", lead.get("total_score", 0)),
        })

        # Add Persona Nodes and Edges from personas list (if present)
        for persona in lead.get("personas", [])[:3]:
            if persona.get("name") or persona.get("full_name"):
                name = persona.get("name") or persona.get("full_name", "")
                safe_name = name.replace(" ", "_").replace(".", "").replace("-", "_")
                p_id = f"person_{safe_name}"

                graph_nodes.append({
                    "id": p_id,
                    "type": "persona",
                    "label": name,
                    "title": persona.get("job_title", persona.get("title", "")),
                    "score": persona.get("match_score", persona.get("score", 0)),
                    "is_target": persona.get("is_target", False),
                })

                graph_edges.append({
                    "source": comp_id,
                    "target": p_id,
                    "weight": persona.get("match_score", persona.get("score", 0)),
                    "type": "employs",
                })

    return {"nodes": graph_nodes, "edges": graph_edges}


# ============================================================================
# MCP TOOLS - HIGH LEVEL ORCHESTRATOR
# ============================================================================

@mcp.tool()
def rank_lead(company_data: str, icp_data: str, user_offering: str, raw_query: str) -> str:
    """Analyzes a single company against an ICP."""
    try:
        company = json.loads(company_data)
        icp = json.loads(icp_data)
        signals = company.get('intent_signals', [])
        result = rank_company_against_icp(company, icp, user_offering, signals)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def detect_top_leads(raw_icp_query: str, user_offering: str, companies_folder: str = "../../inject_collect_project/merged_profiles", max_km: int = 150, limit: int = 3) -> str:
    """
    Automated pipeline that returns a ranked list AND a dynamic relationship graph.
    Delegates to DetectiveAgent for agentic lead discovery.
    - limit: How many top companies to include in the graph.
    """
    try:
        if DetectiveAgent is None:
            return json.dumps({
                "error": "DetectiveAgent not available. Check detective_agent import.",
                "status": "error"
            })

        # Read API keys from environment
        keys = _get_api_keys()
        _groq_api_key = keys["groq"] or ""
        _gemini_api_key = keys["gemini"] or ""
        _ors_api_key = keys["ors"]

        # Instantiate and run the agent
        agent = DetectiveAgent(
            groq_api_key=_groq_api_key,
            gemini_api_key=_gemini_api_key,
            ors_api_key=_ors_api_key,
        )

        agent_result = agent.run(
            icp_text=raw_icp_query,
            desired_lead_count=limit,
            output_name="detect_top_leads",
        )

        # Extract results from AgentResult
        final_rankings = agent_result.get("final_rankings", [])
        top_leads = final_rankings[:limit]

        # Build dynamic graph from agent results
        dynamic_graph = _build_dynamic_graph_from_scratchpad(
            agent_result.get("agent_scratchpad", []),
            top_leads,
        )

        response = {
            "status": "success",
            "summary": {
                "total_found": len(final_rankings),
                "threshold_limit": limit,
                "total_iterations": agent_result.get("total_iterations", 0),
                "halt_reason": agent_result.get("halt_reason", "unknown"),
            },
            "leads": top_leads,
            "dynamic_graph": dynamic_graph,
        }

        # Add warning if max iterations reached
        if agent_result.get("halt_reason") == "max_iterations_reached":
            response["warning"] = (
                "Agent reached the maximum number of iterations. "
                "Results may be partial."
            )

        return json.dumps(response, indent=2, ensure_ascii=False)

    except Exception as e:
        import traceback
        return json.dumps({
            "error": f"Orchestration Error: {str(e)}",
            "traceback": traceback.format_exc(),
            "status": "error"
        })


@mcp.tool()
def run_full_detective_pipeline(
    icp_description: str,
    user_offering: str,
    companies_folder: str = "../../inject_collect_project/merged_profiles",
    output_name: str = "mcp_pipeline_result",
    max_leads: int = 10
) -> str:
    """
    FULL DETECTIVE PIPELINE - Agent as a Tool

    This is the complete agentic pipeline, exposed as an MCP tool.
    The orchestrator can call this single tool to run the entire detective workflow
    via DetectiveAgent (ReAct loop):

    1. ICP Extraction (Groq LLM)
    2. Company Matching & Filtering
    3. Ranking by Similarity (Gemini Embeddings)
    4. Persona Analysis
    5. Final Results with Dynamic Graph

    Args:
        icp_description: Natural language description of ideal customer profile
        user_offering: What product/service you're selling
        companies_folder: Path to merged_profiles folder
        output_name: Name for the output/results
        max_leads: Maximum number of top leads to return

    Returns:
        JSON string with complete pipeline results including:
        - extracted_icp: Structured ICP attributes
        - final_rankings: Top ranked companies
        - persona_results: Key personas for each lead
        - dynamic_graph: Nodes and edges for visualization
        - agent_scratchpad: Full reasoning trace from the agent
        - summary: Pipeline execution summary
        - warning: (optional) Present when agent hit max iterations
    """
    try:
        if DetectiveAgent is None:
            return json.dumps({
                "error": "DetectiveAgent not available. Check detective_agent import.",
                "status": "error"
            })

        # Read API keys from environment
        keys = _get_api_keys()
        _groq_api_key = keys["groq"] or ""
        _gemini_api_key = keys["gemini"] or ""
        _ors_api_key = keys["ors"]

        # Instantiate the DetectiveAgent
        agent = DetectiveAgent(
            groq_api_key=_groq_api_key,
            gemini_api_key=_gemini_api_key,
            ors_api_key=_ors_api_key,
        )

        # Run the agentic pipeline
        agent_result = agent.run(
            icp_text=icp_description,
            desired_lead_count=max_leads,
            output_name=output_name,
        )

        # Extract fields from AgentResult
        final_rankings = agent_result.get("final_rankings", [])
        top_leads = final_rankings[:max_leads]
        persona_results = agent_result.get("persona_results", [])
        agent_scratchpad = agent_result.get("agent_scratchpad", [])
        extracted_icp = agent_result.get("extracted_icp", {})
        halt_reason = agent_result.get("halt_reason", "unknown")
        total_iterations = agent_result.get("total_iterations", 0)
        errors = agent_result.get("errors", [])

        # Build dynamic graph from agent results
        dynamic_graph = _build_dynamic_graph_from_scratchpad(agent_scratchpad, top_leads)

        # Build final response
        response = {
            "status": "success",
            "summary": {
                "total_ranked": len(final_rankings),
                "top_leads_returned": len(top_leads),
                "persona_targets": len(persona_results),
                "total_iterations": total_iterations,
                "halt_reason": halt_reason,
                "errors": len(errors),
            },
            "extracted_icp": extracted_icp,
            "top_leads": top_leads,
            "persona_results": persona_results,
            "dynamic_graph": dynamic_graph,
            "agent_scratchpad": agent_scratchpad,
        }

        # Add warning field when agent hit max iterations (Req 8.5)
        if halt_reason == "max_iterations_reached":
            response["warning"] = (
                "Agent reached the maximum number of iterations. "
                "Results may be partial — not all leads may have been fully evaluated."
            )

        return json.dumps(response, indent=2, ensure_ascii=False)

    except Exception as e:
        import traceback
        return json.dumps({
            "error": f"Pipeline Error: {str(e)}",
            "traceback": traceback.format_exc(),
            "status": "error"
        })


if __name__ == "__main__":
    # Force UTF-8 for Windows Terminal Compatibility (only when running as script)
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, ValueError):
        pass

    print(f"[DEBUG] Current file: {current_file}", file=sys.stderr)
    print(f"[DEBUG] Project root: {project_root}", file=sys.stderr)
    print(f"[INIT] Detective MCP Server Starting...", file=sys.stderr)
    keys = _get_api_keys()
    print(f"[INIT] Gemini API: {'OK' if keys['gemini'] else 'MISSING'}", file=sys.stderr)
    print(f"[INIT] Groq API: {'OK' if keys['groq'] else 'MISSING'}", file=sys.stderr)
    print(f"[INIT] ORS API: {'OK' if keys['ors'] else 'MISSING'}", file=sys.stderr)

    if not _fastmcp_available:
        print("[ERROR] fastmcp/mcp is not installed. Cannot start MCP server.", file=sys.stderr)
        sys.exit(1)

    mcp.run(transport="stdio")

#!/usr/bin/env python3
"""
Detective MCP Server - B2B Lead Detective & Orchestrator

High-level orchestrator tools for B2B lead generation:
- rank_lead: Analyze single company against ICP
- detect_top_leads: Full pipeline from ICP extraction to ranked leads with graph

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

# Debug: Print paths
print(f"[DEBUG] Current file: {current_file}", file=sys.stderr)
print(f"[DEBUG] Project root: {project_root}", file=sys.stderr)
print(f"[DEBUG] Current sys.path: {sys.path[:3]}...", file=sys.stderr)

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"[DEBUG] Added {project_root} to sys.path", file=sys.stderr)

# Also try one level up (in case we're in a different structure)
parent_of_project = project_root.parent
if str(parent_of_project) not in sys.path:
    sys.path.insert(0, str(parent_of_project))
    print(f"[DEBUG] Added {parent_of_project} to sys.path", file=sys.stderr)

# Force UTF-8 for Windows Terminal Compatibility
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 2. LOAD ENVIRONMENT
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"[DEBUG] Loaded .env from {env_path}", file=sys.stderr)
else:
    load_dotenv()
    print(f"[DEBUG] Loaded .env from default location", file=sys.stderr)

# 3. IMPORTS
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

# Import detective components (from parent directory)
from brain.icp_agent import ICPExtractionAgent

# Import the full LangGraph pipeline
try:
    from detective_graph import run_detective_pipeline
    print("[DEBUG] Imported run_detective_pipeline from detective_graph", file=sys.stderr)
except ImportError as e:
    print(f"[WARN] Could not import run_detective_pipeline: {e}", file=sys.stderr)
    run_detective_pipeline = None

# 4. INITIALIZE SERVER
mcp = FastMCP("B2B-Detective-Server")

# Initialize components
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
ors_api_key = os.getenv("ORS_API_KEY")

print(f"[INIT] Detective MCP Server Starting...", file=sys.stderr)
print(f"[INIT] Gemini API: {'OK' if api_key else 'MISSING'}", file=sys.stderr)
print(f"[INIT] Groq API: {'OK' if groq_api_key else 'MISSING'}", file=sys.stderr)
print(f"[INIT] ORS API: {'OK' if ors_api_key else 'MISSING'}", file=sys.stderr)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_companies_from_folder(folder_path: str) -> dict:
    """Load company profiles from JSON files."""
    companies = {}
    
    # Resolve path - handle relative paths from mcp_server location
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Extract the folder name from the path
    folder_name = Path(folder_path).name
    parent_hint = Path(folder_path).parent.name if Path(folder_path).parent.name != "." else ""
    
    # Try multiple path resolutions
    paths_to_try = [
        Path(folder_path),  # As-is
        current_dir / folder_path,  # Relative to mcp_server
        project_root / folder_path.lstrip("./").replace("../", ""),  # Relative to detective
        project_root.parent / folder_path.lstrip("./").replace("../", ""),  # Relative to outbound_project
        # Hardcoded fallback to known location
        project_root.parent / "inject_collect_project" / "merged_profiles",
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
            print(f"[DEBUG] Trying path: {p.absolute()}", file=sys.stderr)
            if p.exists():
                folder = p
                print(f"[DEBUG] Found folder at: {p.absolute()}", file=sys.stderr)
                break
        except Exception as e:
            continue
    
    if not folder:
        print(f"[ERROR] Could not find folder: {folder_path}", file=sys.stderr)
        print(f"[ERROR] Tried: {[str(p) for p in paths_to_try[:5]]}...", file=sys.stderr)
        return companies
    
    json_files = list(folder.glob("*.json"))
    print(f"[DEBUG] Found {len(json_files)} JSON files in {folder}", file=sys.stderr)
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                company_key = file_path.stem
                companies[company_key] = data
                print(f"[DEBUG] Loaded: {company_key}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Failed to load {file_path}: {e}", file=sys.stderr)
    
    print(f"[DEBUG] Total companies loaded: {len(companies)}", file=sys.stderr)
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
        print(f"[WARN] ICP extraction failed: {e}", file=sys.stderr)
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
    - limit: How many top companies to include in the graph.
    """
    try:
        # Step 1: Extraction
        icp_attributes = extract_icp_attributes(raw_icp_query)
        constraints = icp_attributes
        target_roles = constraints.get('target_roles', [])
        
        # Step 2: Load companies
        companies = load_companies_from_folder(companies_folder)
        if not companies:
            return json.dumps({
                "error": f"No companies found in folder: {companies_folder}",
                "status": "error"
            })
        
        # Step 3: Filter & Ranking
        final_rankings = []
        
        min_size = constraints.get('company_size', {}).get('min') or 0
        max_size = constraints.get('company_size', {}).get('max') or 1000000
        target_countries = [c.lower() for c in constraints.get('target_countries', [])]
        target_industries = [i.lower() for i in constraints.get('industry', [])]
        
        print(f"[DEBUG] ICP Constraints: size={min_size}-{max_size}, countries={target_countries}, industries={target_industries}", file=sys.stderr)
        
        filtered_count = 0
        for company_key, company in companies.items():
            # Size filter - handle both nested and flat structure
            employees_raw = company.get('basic_info', {}).get('employees') or company.get('employees') or company.get('estimated_num_employees') or 0
            # Convert string like "412,800 (Global)" to number
            if isinstance(employees_raw, str):
                import re
                nums = re.findall(r'[\d,]+', employees_raw)
                if nums:
                    employees = int(nums[0].replace(',', ''))
                else:
                    employees = 0
            else:
                employees = employees_raw or 0
            if not (min_size <= employees <= max_size):
                if filtered_count < 3:
                    print(f"[DEBUG] Filtered {company_key}: employees={employees} not in range {min_size}-{max_size}", file=sys.stderr)
                filtered_count += 1
                continue
            
            # Country filter - handle both nested and flat structure
            country = (company.get('basic_info', {}).get('country') or company.get('country', '')).lower()
            if target_countries and country not in target_countries:
                if filtered_count < 3:
                    print(f"[DEBUG] Filtered {company_key}: country='{country}' not in {target_countries}", file=sys.stderr)
                filtered_count += 1
                continue
            
            # Industry filter - handle both nested and flat structure
            industries_raw = company.get('classification', {}).get('industries') or company.get('industry', '')
            if isinstance(industries_raw, str):
                # Split comma-separated string into list
                industries = [i.strip().lower() for i in industries_raw.split(',')]
            else:
                industries = [i.lower() for i in industries_raw]
            if target_industries and not any(ti in ind for ti in target_industries for ind in industries):
                if filtered_count < 3:
                    print(f"[DEBUG] Filtered {company_key}: industries={industries} don't match {target_industries}", file=sys.stderr)
                filtered_count += 1
                continue
            
            print(f"[DEBUG] PASSED: {company_key} - employees={employees}, country={country}, industries={industries}", file=sys.stderr)
            
            # Score company
            signals = company.get('intent_signals', [])
            score_data = rank_company_against_icp(company, constraints, user_offering, signals)
            
            # Score personas - handle both flat and nested structure
            personas_raw = company.get('personas', [])
            # If personas is a dict (from personas_discovered folder), convert to list
            if isinstance(personas_raw, dict):
                personas_list = []
                for p_key, p_data in personas_raw.items():
                    if isinstance(p_data, dict):
                        personas_list.append(p_data)
                personas_raw = personas_list
            
            personas = []
            for p in personas_raw:
                base_p_score = p.get('is_likely_to_engage', 0.5) * (p.get('intent_strength', 5) / 10)
                is_target = any(
                    role.lower().split()[0] in p.get('job_title', '').lower() 
                    for role in target_roles if role
                )
                persona_final_score = round(base_p_score * (1.5 if is_target else 0.7), 2)
                
                personas.append({
                    "name": p.get('full_name', ''),
                    "title": p.get('job_title', ''),
                    "score": persona_final_score,
                    "is_target": is_target
                })
            
            personas.sort(key=lambda x: x["score"], reverse=True)
            
            # Get company name - handle both nested and flat structure
            company_name = company.get("basic_info", {}).get("name") or company.get("name", company_key)
            
            final_rankings.append({
                "company_key": company_key,
                "company": company_name,
                "total_score": score_data["total_score"],
                "reasoning": score_data["reason"],
                "top_contacts": personas[:3],
                "raw_data": company
            })
        
        final_rankings.sort(key=lambda x: x["total_score"], reverse=True)
        
        # --- DYNAMIC GRAPH ARCHITECT LOGIC ---
        top_subset = final_rankings[:limit]
        graph_nodes = []
        graph_edges = []
        
        for lead in top_subset:
            comp_id = f"node_{lead['company_key']}"
            
            # 1. Add Company Node
            graph_nodes.append({
                "id": comp_id,
                "type": "company",
                "label": lead['company'],
                "score": lead['total_score'],
                "info": lead['reasoning']
            })
            
            # 2. Add Persona Nodes and Edges
            for p in lead.get('top_contacts', []):
                if p.get('name'):
                    safe_name = p['name'].replace(' ', '_').replace('.', '').replace('-', '_')
                    p_id = f"person_{safe_name}"
                    
                    # Add Persona Node
                    graph_nodes.append({
                        "id": p_id,
                        "type": "persona",
                        "label": p['name'],
                        "title": p.get('title', ''),
                        "score": p.get('score', 0),
                        "is_target": p.get('is_target', False)
                    })
                    
                    # Add Edge
                    graph_edges.append({
                        "source": comp_id,
                        "target": p_id,
                        "weight": p.get('score', 0),
                        "type": "employs"
                    })
        
        return json.dumps({
            "status": "success",
            "summary": {
                "total_found": len(final_rankings),
                "threshold_limit": limit
            },
            "leads": top_subset,
            "dynamic_graph": {
                "nodes": graph_nodes,
                "edges": graph_edges
            }
        }, indent=2, ensure_ascii=False)
    
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
    
    This is the complete LangGraph pipeline from main.py, exposed as an MCP tool.
    The orchestrator can call this single tool to run the entire detective workflow:
    
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
        - matched_companies: All matching companies
        - final_rankings: Top ranked companies
        - persona_results: Key personas for each lead
        - dynamic_graph: Nodes and edges for visualization
        - summary: Pipeline execution summary
    """
    try:
        if run_detective_pipeline is None:
            return json.dumps({
                "error": "run_detective_pipeline not available. Check detective_graph import.",
                "status": "error"
            })
        
        print(f"[MCP PIPELINE] Starting full detective pipeline...", file=sys.stderr)
        print(f"[MCP PIPELINE] ICP: {icp_description[:100]}...", file=sys.stderr)
        print(f"[MCP PIPELINE] User Offering: {user_offering}", file=sys.stderr)
        
        # Run the full LangGraph pipeline
        result = run_detective_pipeline(
            icp_text=icp_description,
            output_name=output_name
        )
        
        # Extract and format the key results
        final_rankings = result.get("final_rankings", [])
        
        # Limit to max_leads
        top_leads = final_rankings[:max_leads]
        
        # Build dynamic graph from the results
        graph_nodes = []
        graph_edges = []
        
        for lead in top_leads:
            comp_id = f"node_{lead.get('company_key', 'unknown')}"
            
            # Add Company Node
            graph_nodes.append({
                "id": comp_id,
                "type": "company",
                "label": lead.get("company_name", lead.get("company_key", "Unknown")),
                "score": lead.get("total_score", 0),
                "industry_match": lead.get("industry_match", False),
                "location_match": lead.get("location_match", False)
            })
            
            # Add Persona Nodes and Edges
            for persona in lead.get("personas", [])[:3]:
                if persona.get("name"):
                    safe_name = persona["name"].replace(" ", "_").replace(".", "").replace("-", "_")
                    p_id = f"person_{safe_name}"
                    
                    graph_nodes.append({
                        "id": p_id,
                        "type": "persona",
                        "label": persona["name"],
                        "title": persona.get("job_title", ""),
                        "score": persona.get("match_score", 0),
                        "is_target": persona.get("is_target", False)
                    })
                    
                    graph_edges.append({
                        "source": comp_id,
                        "target": p_id,
                        "weight": persona.get("match_score", 0),
                        "type": "employs"
                    })
        
        # Build final response
        extracted_icp = result.get("extracted_icp", {})
        # Convert ICPAttributes to dict if needed
        if hasattr(extracted_icp, 'model_dump'):
            extracted_icp = extracted_icp.model_dump()
        elif hasattr(extracted_icp, 'dict'):
            extracted_icp = extracted_icp.dict()
        
        response = {
            "status": "success",
            "summary": {
                "pipeline_step": result.get("step_completed", "unknown"),
                "total_matched": len(result.get("matched_companies", {})),
                "final_ranked": len(final_rankings),
                "top_leads_returned": len(top_leads),
                "persona_targets": len(result.get("persona_results", [])),
                "errors": len(result.get("errors", []))
            },
            "extracted_icp": extracted_icp,
            "top_leads": top_leads,
            "dynamic_graph": {
                "nodes": graph_nodes,
                "edges": graph_edges
            }
        }
        
        print(f"[MCP PIPELINE] Completed: {response['summary']}", file=sys.stderr)
        
        return json.dumps(response, indent=2, ensure_ascii=False)
    
    except Exception as e:
        import traceback
        return json.dumps({
            "error": f"Pipeline Error: {str(e)}",
            "traceback": traceback.format_exc(),
            "status": "error"
        })


if __name__ == "__main__":
    mcp.run(transport="stdio")

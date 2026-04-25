"""
Detective Graph - LangGraph-based ICP extraction and company targeting pipeline
"""

import os
import json
import logging
from typing import Dict, List, Optional, TypedDict, Any
from pathlib import Path
from dotenv import load_dotenv

from groq import Groq

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from brain import ICPExtractionAgent, CompanyMatcher, ICPAttributes, GeoAgent
from ranking import (
    GeminiEmbedder, CompanyRanker, CompanyFilter, 
    FinalScorer, PersonaRanker
)

load_dotenv()

logger = logging.getLogger(__name__)


class DetectiveState(TypedDict):
    """State maintained across graph execution"""
    # Input
    icp_text: str
    output_name: str
    
    # Intermediate results
    icp_attributes: Optional[ICPAttributes]
    matched_companies: Dict[str, Dict]
    geo_filtered_companies: Dict[str, Dict]
    filtered_companies: Dict[str, Dict]
    ranking_results: List[Dict]
    final_rankings: List[Dict]
    persona_results: List[Dict]
    
    # Configuration
    target_roles: List[str]
    target_countries: List[str]
    employee_range: Optional[tuple]
    geo_config: Optional[Dict]
    
    # Status
    errors: List[str]
    step_completed: str
    
    # Output paths
    output_files: Dict[str, str]


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def node_extract_icp(state: DetectiveState) -> DetectiveState:
    """STEP 1: Extract ICP attributes from text"""
    logger.info("[NODE] Extracting ICP attributes...")
    
    try:
        # Initialize Groq client with cheapest model to save tokens
        groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        # Use cheapest model for ICP extraction
        agent = ICPExtractionAgent(groq_client, model="llama-3.1-8b-instant")
        icp_attributes = agent.extract_icp_attributes(state['icp_text'])
        
        # Save ICP
        output_path = Path(f"{state['output_name']}_icp.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(icp_attributes.model_dump(), f, indent=2)
        
        # Extract target roles
        target_roles = icp_attributes.model_dump().get('target_roles', [])
        target_countries = icp_attributes.model_dump().get('target_countries', [])
        
        state['icp_attributes'] = icp_attributes
        state['target_roles'] = target_roles
        state['target_countries'] = target_countries
        state['step_completed'] = 'icp_extraction'
        state['output_files']['icp'] = str(output_path)
        
        logger.info(f"[SUCCESS] ICP extracted: {len(icp_attributes.industry)} industries, {len(target_roles)} roles")
        
    except Exception as e:
        logger.error(f"[ERROR] ICP extraction failed: {e}")
        state['errors'].append(f"ICP extraction: {str(e)}")
    
    return state


def node_match_companies(state: DetectiveState) -> DetectiveState:
    """STEP 2: Match companies with ICP industries"""
    logger.info("[NODE] Matching companies...")
    
    if not state.get('icp_attributes'):
        state['errors'].append("No ICP attributes available")
        return state
    
    try:
        # Initialize Groq client with cheaper model for matching
        groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        # Use cheaper/faster model for company matching to save tokens
        matcher = CompanyMatcher(groq_client, model="llama-3.1-8b-instant")
        
        if state['icp_attributes'].industry:
            matched = matcher.find_matching_companies(state['icp_attributes'].industry)
            
            if matched:
                # Save matches
                output_folder = Path(f"matched_companies_{state['output_name']}")
                output_folder.mkdir(exist_ok=True)
                
                for key, profile in matched.items():
                    output_file = output_folder / f"{key}_MATCHED.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(profile, f, indent=2)
                
                state['matched_companies'] = matched
                state['step_completed'] = 'company_matching'
                state['output_files']['matched_companies'] = str(output_folder)
                
                logger.info(f"[SUCCESS] Matched {len(matched)} companies")
            else:
                logger.warning("[WARNING] No companies matched")
        else:
            logger.warning("[WARNING] No industries in ICP")
            
    except Exception as e:
        logger.error(f"[ERROR] Company matching failed: {e}")
        state['errors'].append(f"Company matching: {str(e)}")
    
    return state


def node_geo_filter(state: DetectiveState) -> DetectiveState:
    """STEP 2b: Geo-filter by city proximity (optional)"""
    logger.info("[NODE] Geo-filtering companies...")
    
    if not state.get('matched_companies'):
        logger.info("[SKIP] No matched companies to geo-filter")
        return state
    
    try:
        geo_agent = GeoAgent()
        
        if geo_agent.is_enabled():
            geo_config = geo_agent.parse_icp_location(
                state['icp_text'], 
                state['icp_attributes'].model_dump()
            )
            
            if geo_config['enabled']:
                logger.info(f"[GEO] Filtering: {geo_config['city']} within {geo_config['range_km']}km")
                
                filtered = geo_agent.filter_companies_by_proximity(
                    state['matched_companies'],
                    geo_config['city'],
                    geo_config['country'],
                    geo_config['range_km']
                )
                
                state['geo_filtered_companies'] = filtered
                state['geo_config'] = geo_config
                logger.info(f"[SUCCESS] Geo-filtered: {len(filtered)} companies")
            else:
                logger.info("[SKIP] Geo-filtering disabled (no city specified)")
                state['geo_filtered_companies'] = state['matched_companies']
        else:
            logger.info("[SKIP] Geo agent not enabled (no ORS_API_KEY)")
            state['geo_filtered_companies'] = state['matched_companies']
            
    except Exception as e:
        logger.error(f"[ERROR] Geo-filter failed: {e}")
        state['errors'].append(f"Geo-filter: {str(e)}")
        # Continue with original matches
        state['geo_filtered_companies'] = state['matched_companies']
    
    state['step_completed'] = 'geo_filter'
    return state


def node_collect_intent(state: DetectiveState) -> DetectiveState:
    """STEP 3: Collect intent signals for matched companies (optional)"""
    logger.info("[NODE] Collecting intent signals...")
    
    companies = state.get('geo_filtered_companies') or state.get('matched_companies')
    
    if not companies:
        logger.info("[SKIP] No companies to collect intent for")
        return state
    
    try:
        # Extract company names
        company_names = [company.get('name', key) for key, company in companies.items()]
        
        logger.info(f"[INTENT] Collecting signals for {len(company_names)} companies...")
        
        # Try to run agentic_intent (with error handling for rate limits)
        import sys
        import importlib.util
        
        agentic_intent_folder = str(Path(__file__).parent.parent / 'agentic_intent')
        
        # Add agentic_intent folder to sys.path for its internal imports to work
        if agentic_intent_folder not in sys.path:
            sys.path.insert(0, agentic_intent_folder)
        
        # Also add parent directory so agentic_intent can import its submodules
        parent_folder = str(Path(__file__).parent.parent)
        if parent_folder not in sys.path:
            sys.path.insert(0, parent_folder)
        
        try:
            # Use a unique module name to avoid conflicts
            spec = importlib.util.spec_from_file_location(
                "agentic_intent_detective_main", 
                Path(agentic_intent_folder) / 'main.py'
            )
            agentic_module = importlib.util.module_from_spec(spec)
            sys.modules['agentic_intent_detective_main'] = agentic_module
            spec.loader.exec_module(agentic_module)
            
            # Run intent collection (async)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Collect intent
            intent_result = loop.run_until_complete(agentic_module.main(
                companies=company_names,
                save_to_file=True
            ))
            
            if intent_result:
                logger.info(f"[SUCCESS] Intent signals collected for {len(company_names)} companies")
                state['output_files']['intent'] = '../agentic_intent/output/intent_results.json'
            else:
                logger.warning("[WARNING] Intent collection returned no results")
                
        except Exception as e:
            logger.warning(f"[SKIP] Intent collection failed (rate limit or error): {e}")
            # Save for manual run
            companies_file = Path('matched_companies_for_intent.json')
            with open(companies_file, 'w', encoding='utf-8') as f:
                import json
                json.dump({
                    'companies': company_names,
                    'total': len(company_names)
                }, f, indent=2)
            logger.info(f"[WORKAROUND] Saved company list for manual intent collection")
            
    except Exception as e:
        logger.error(f"[ERROR] Intent collection setup failed: {e}")
        state['errors'].append(f"Intent collection: {str(e)}")
    
    state['step_completed'] = 'intent_collection'
    return state


def node_filter_and_rank(state: DetectiveState) -> DetectiveState:
    """STEP 4: Filter by criteria and rank by similarity"""
    logger.info("[NODE] Filtering and ranking companies...")
    
    companies = state.get('geo_filtered_companies') or state.get('matched_companies')
    
    if not companies:
        logger.warning("[SKIP] No companies to filter/rank")
        return state
    
    try:
        # Step 4a: Filter by employees and country
        company_filter = CompanyFilter(state['icp_attributes'].model_dump())
        filtered = company_filter.filter_companies(companies)
        
        if not filtered:
            logger.warning("[WARNING] No companies passed filters")
            return state
        
        # Step 4b: Rank by similarity
        from ranking import GeminiEmbedder
        embedder = GeminiEmbedder()
        ranker = CompanyRanker(embedder=embedder)
        ranker.embed_icp(state['icp_text'])
        
        # Rank filtered companies
        rankings = ranker.rank_companies(filtered)
        
        if rankings:
            ranker.save_ranking(rankings, f"{state['output_name']}_filtered")
            
            state['filtered_companies'] = filtered
            state['ranking_results'] = rankings
            state['step_completed'] = 'filter_and_rank'
            state['output_files']['filtered_ranking'] = f"ranking/{state['output_name']}_filtered_ranking.json"
            
            logger.info(f"[SUCCESS] Ranked {len(rankings)} companies")
        else:
            logger.warning("[WARNING] No rankings generated")
        
    except Exception as e:
        logger.error(f"[ERROR] Filter/rank failed: {e}")
        state['errors'].append(f"Filter/rank: {str(e)}")
    
    return state


def node_final_scoring(state: DetectiveState) -> DetectiveState:
    """STEP 5: Calculate final scores with optional intent boost"""
    logger.info("[NODE] Calculating final scores...")
    
    if not state.get('ranking_results'):
        logger.warning("[SKIP] No ranking results for final scoring")
        return state
    
    try:
        # Find intent file (may not exist)
        intent_file = Path(f'../agentic_intent/output/intent_results.json')
        if not intent_file.exists():
            intent_file = Path('intent_results.json')
        
        final_scorer = FinalScorer()  # Uses GROQ_API_KEY from env
        
        final_rankings = final_scorer.calculate_final_scores(
            state['ranking_results'],
            state.get('intent_signals', {})
        )
        
        if final_rankings:
            output_file = f"ranking/{state['output_name']}_final_ranking.json"
            final_scorer.save_final_ranking(
                final_rankings,
                output_file
            )
            
            state['final_rankings'] = final_rankings
            state['step_completed'] = 'final_scoring'
            state['output_files']['final_ranking'] = f"ranking/{state['output_name']}_final_ranking.json"
            
            logger.info(f"[SUCCESS] Final scores calculated for {len(final_rankings)} companies")
        else:
            logger.warning("[WARNING] No final rankings calculated")
            
    except Exception as e:
        logger.error(f"[ERROR] Final scoring failed: {e}")
        state['errors'].append(f"Final scoring: {str(e)}")
    
    return state


def node_rank_personas(state: DetectiveState) -> DetectiveState:
    """STEP 6: Rank personas for each company"""
    logger.info("[NODE] Ranking personas...")
    
    if not state.get('final_rankings'):
        logger.warning("[SKIP] No final rankings for persona ranking")
        return state
    
    try:
        personas_folder = Path('../inject_collect_project/personas_discovered')
        if not personas_folder.exists():
            personas_folder = Path('personas_discovered')
        
        target_roles = state['icp_attributes'].target_roles if hasattr(state['icp_attributes'], 'target_roles') else []
        persona_ranker = PersonaRanker(target_roles=target_roles)  # Uses GROQ_API_KEY from env
        
        persona_results = persona_ranker.rank_personas_for_all_companies(
            state['final_rankings'],
            str(personas_folder)
        )
        
        if persona_results:
            output_path = f"ranking/{state['output_name']}_personas.json"
            persona_ranker.save_persona_rankings(
                persona_results,
                output_path
            )
            
            state['persona_results'] = persona_results
            state['step_completed'] = 'persona_ranking'
            state['output_files']['personas'] = f"ranking/{state['output_name']}_personas.json"
            
            logger.info(f"[SUCCESS] Ranked personas for {len(persona_results)} companies")
        else:
            logger.warning("[WARNING] No persona results generated")
            
    except Exception as e:
        logger.error(f"[ERROR] Persona ranking failed: {e}")
        state['errors'].append(f"Persona ranking: {str(e)}")
    
    return state


def node_print_final_results(state: DetectiveState) -> DetectiveState:
    """Print final comprehensive results"""
    print("\n" + "=" * 80)
    print("🎯 FINAL TARGETING RESULTS - Companies + Personas")
    print("=" * 80)
    
    if state.get('persona_results'):
        for i, pr in enumerate(state['persona_results'][:10]):
            company_rank = i + 1
            company_name = pr.get('company_name', 'Unknown')
            
            print(f"\n#{company_rank} {company_name}")
            
            # Get top persona
            top_personas = pr.get('top_personas', [])
            if top_personas:
                persona = top_personas[0]
                print(f"   🎯 Target: {persona.get('name', 'N/A')}")
                print(f"   📋 Title: {persona.get('job_title', 'N/A')}")
                print(f"   📍 Department: {persona.get('department', 'N/A')}")
                print(f"   📊 Persona Score: {persona.get('persona_score', 0):.3f}")
                
                if persona.get('is_target'):
                    print(f"   ✅ TARGET ROLE")
                
                if persona.get('email'):
                    print(f"   📧 {persona['email']}")
                if persona.get('linkedin'):
                    print(f"   � {persona['linkedin']}")
            else:
                print(f"   ⚠️ No personas found")
    
    print("\n" + "=" * 80)
    print("📁 OUTPUT FILES:")
    print("=" * 80)
    for name, path in state['output_files'].items():
        print(f"   • {name}: {path}")
    
    if state['errors']:
        print("\n⚠️ ERRORS ENCOUNTERED:")
        for err in state['errors']:
            print(f"   • {err}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    
    return state


# =============================================================================
# CONDITIONAL EDGES
# =============================================================================

def should_continue_after_icp(state: DetectiveState) -> str:
    """Determine next step after ICP extraction"""
    if state.get('icp_attributes') and state['icp_attributes'].industry:
        return "match_companies"
    return "end"


def should_geo_filter(state: DetectiveState) -> str:
    """Determine if geo-filtering should run"""
    if state.get('matched_companies'):
        return "geo_filter"
    return "end"


def should_filter_and_rank(state: DetectiveState) -> str:
    """Determine if filtering/ranking should run"""
    companies = state.get('geo_filtered_companies') or state.get('matched_companies')
    if companies:
        return "filter_and_rank"
    return "end"


def should_final_score(state: DetectiveState) -> str:
    """Determine if final scoring should run"""
    if state.get('ranking_results'):
        return "final_scoring"
    return "end"


def should_rank_personas(state: DetectiveState) -> str:
    """Determine if persona ranking should run"""
    if state.get('final_rankings'):
        return "rank_personas"
    return "end"


# =============================================================================
# BUILD GRAPH
# =============================================================================

def build_detective_graph() -> StateGraph:
    """Build and return the LangGraph state graph"""
    
    # Create graph
    workflow = StateGraph(DetectiveState)
    
    # Add nodes
    workflow.add_node("extract_icp", node_extract_icp)
    workflow.add_node("match_companies", node_match_companies)
    workflow.add_node("geo_filter", node_geo_filter)
    workflow.add_node("collect_intent", node_collect_intent)
    workflow.add_node("filter_and_rank", node_filter_and_rank)
    workflow.add_node("final_scoring", node_final_scoring)
    workflow.add_node("rank_personas", node_rank_personas)
    workflow.add_node("print_results", node_print_final_results)
    
    # Set entry point
    workflow.set_entry_point("extract_icp")
    
    # Add edges with conditions
    workflow.add_conditional_edges(
        "extract_icp",
        should_continue_after_icp,
        {
            "match_companies": "match_companies",
            "end": "print_results"
        }
    )
    
    workflow.add_conditional_edges(
        "match_companies",
        should_geo_filter,
        {
            "geo_filter": "geo_filter",
            "end": "print_results"
        }
    )
    
    workflow.add_edge("geo_filter", "collect_intent")
    workflow.add_edge("collect_intent", "filter_and_rank")
    
    workflow.add_conditional_edges(
        "filter_and_rank",
        should_final_score,
        {
            "final_scoring": "final_scoring",
            "end": "print_results"
        }
    )
    
    workflow.add_conditional_edges(
        "final_scoring",
        should_rank_personas,
        {
            "rank_personas": "rank_personas",
            "end": "print_results"
        }
    )
    
    workflow.add_edge("rank_personas", "print_results")
    workflow.add_edge("print_results", END)
    
    return workflow.compile()


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def run_detective_pipeline(icp_text: str, output_name: str = "example_icp") -> DetectiveState:
    """
    Run the complete detective pipeline using LangGraph
    
    Args:
        icp_text: Raw ICP text
        output_name: Name for output files
        
    Returns:
        Final state with all results
    """
    # Initialize state
    initial_state: DetectiveState = {
        'icp_text': icp_text,
        'output_name': output_name,
        'icp_attributes': None,
        'matched_companies': {},
        'geo_filtered_companies': {},
        'filtered_companies': {},
        'ranking_results': [],
        'final_rankings': [],
        'persona_results': [],
        'target_roles': [],
        'target_countries': [],
        'employee_range': None,
        'geo_config': None,
        'errors': [],
        'step_completed': '',
        'output_files': {}
    }
    
    # Build and run graph
    graph = build_detective_graph()
    
    print("\n" + "=" * 80)
    print("🕵️ DETECTIVE PIPELINE - LangGraph Orchestration")
    print("=" * 80)
    
    final_state = graph.invoke(initial_state)
    
    return final_state


if __name__ == "__main__":
    # Example ICP
    example_icp = """
    I want IT companies with 50-500 employees and annual revenue between $10M - $100M.
    Target companies should be in North America and Europe, specifically in the United States,
    Canada, United Kingdom, and Germany.
    
    We want to connect with Sales Mangers , CTOs, and Heads of Product.
    
    Must-have traits:
    - Using modern tech stack (React, Python, AWS)
    - In growth stage with Series B or C funding
    - Product-led growth model
    """
    
    result = run_detective_pipeline(example_icp, "test_icp")
    
    print(f"\n✅ Pipeline completed with {len(result.get('persona_results', []))} persona targets")

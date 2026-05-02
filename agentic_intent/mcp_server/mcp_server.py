from fastmcp import FastMCP
from ddgs import DDGS
from dotenv import load_dotenv
import os
import json
import sys
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluator import SystemEvaluator
from evaluation.xai import ExplainabilityEngine, ABTester
from utils.config_store import ConfigStore

load_dotenv()
mcp = FastMCP("intent-tools")


# Initialize evaluators
evaluator = SystemEvaluator()
xai_engine = ExplainabilityEngine()
ab_tester = ABTester()

@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": False,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def search_company_funding(company_name: str) -> list:
    """
    Search for recent news about a company related to funding.

    Args:
        company_name: Name of the company

    Returns:
        List of top 5 news results with title, url, snippet
    """
    config = ConfigStore()
    max_results = config.get("search_params.funding_max_results", 5)

    query = f"{company_name} funding"

    custom_queries = config.get("custom_search_queries", {}).get(company_name, [])
    if custom_queries:
        query += " " + " ".join(custom_queries)


    results_list = []

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results= max_results)

            for r in results:
                results_list.append({
                    "title": r.get("title"),
                    "url": r.get("href"),
                    "snippet": r.get("body")
                })

    except Exception as e:
        return [{"error": str(e)}]

    return results_list




@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": False,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def search_company_news(company_name: str) -> list:
    """
    Search for recent general news about a company.

    Args:
        company_name: Name of the company

    Returns:
        List of top 5 news results with title, url, date, snippet, source
    """
    config = ConfigStore()
    max_results = config.get("search_params.news_max_results", 5)

    query = f"{company_name} news"

    custom_queries = config.get("custom_search_queries", {}).get(company_name, [])
    if custom_queries:
        query += " " + " ".join(custom_queries)
    results_list = []

    try:
        with DDGS() as ddgs:
            results = ddgs.news(query, max_results=max_results)

            for r in results:
                results_list.append({
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "snippet": r.get("body"),
                    "date": r.get("date"),
                    "source": r.get("source")
                })

    except Exception as e:
        return [{"error": str(e)}]

    return results_list






# ================= CONTEXT TOOLS =================
@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def retrieve_company_intent(company_name: str, retrieval_type: str = "essential") -> str:
    """
    Retrieve stored intelligence about a company.
    
    Args:
        company_name: Name of the company
        retrieval_type: Type of retrieval - 'essential' or 'full'
            - 'essential': Last 5 signals + older signals summarized
            - 'full': Complete data for last 5 signals with all metadata
    
    Returns:
        JSON string with company intelligence
    """
    from utils.intent_store import IntentStore
    
    # Use absolute path to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "intents.db")
    
    intent_store = IntentStore(db_path)

    if retrieval_type == "essential":
        data = intent_store.retrieve_essential(company_name)
        return json.dumps(data, indent=2, default=str)
    elif retrieval_type == "full":
        data = intent_store.retrieve_full(company_name)
        return json.dumps(data, indent=2, default=str)
    else:
        return json.dumps({"error": f"Invalid retrieval_type: {retrieval_type}. Use 'essential' or 'full'."})



@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def search_company_event(company_name: str, event_query: str) -> str:
    """
    Search for a specific event related to a company using semantic search.
    Provide a description of the event you're looking for.
    
    Args:
        company_name: Name of the company
        event_query: Description of the event to search for (e.g., "the funding round from last month", "partnership with Toyota")
    
    Returns:
        JSON string with matching events and their details
    """
    from utils.intent_store import IntentStore
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "intents.db")
    
    intent_store = IntentStore(db_path)
    
    results = intent_store.search_events_by_similarity(company_name, event_query, top_k=3)
    
    if not results:
        return json.dumps({
            "company": company_name,
            "query": event_query,
            "message": f"No matching events found for {company_name}",
            "events": []
        }, indent=2)
    
    return json.dumps({
        "company": company_name,
        "query": event_query,
        "total_matches": len(results),
        "events": results
    }, indent=2, default=str)






# ================= FEEDBACK LOOP TOOLS =================

@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "idempotentHint": False,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def set_search_max_results(tool_name: str, max_results: int) -> str:
    """
    Adjust the maximum number of search results for a search tool.
    
    Args:
        tool_name: One of 'funding' or 'news'
        max_results: New maximum number of results (1-20)
    
    Returns:
        Confirmation message
    """
    config = ConfigStore()
    
    valid_tools = ['funding', 'news']
    if tool_name not in valid_tools:
        return json.dumps({"error": f"Invalid tool_name. Use one of: {valid_tools}"})
    
    if not 1 <= max_results <= 20:
        return json.dumps({"error": "max_results must be between 1 and 20"})
    
    config.set(f"search_params.{tool_name}_max_results", max_results)
    return json.dumps({
        "success": True,
        "message": f"Updated {tool_name}_max_results to {max_results}",
        "current_settings": {
            k: v for k, v in config.load()["search_params"].items()
        }
    }, indent=2)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "idempotentHint": False,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def set_confidence_threshold(event_type: str, threshold: float) -> str:
    """
    Set minimum confidence threshold for events.
    
    Args:
        event_type: 'funding', 'news', or 'event'
        threshold: Minimum confidence score (0.0 to 1.0)
    
    Returns:
        Confirmation message
    """
    config = ConfigStore()
    
    valid_types = {
        'funding': 'funding_min_confidence',
        'news': 'news_min_confidence',
        'event': 'event_confidence_threshold'
    }
    
    if event_type not in valid_types:
        return json.dumps({"error": f"Invalid event_type. Use one of: {list(valid_types.keys())}"})
    
    if not 0.0 <= threshold <= 1.0:
        return json.dumps({"error": "threshold must be between 0.0 and 1.0"})
    
    key = valid_types[event_type]
    config.set(f"confidence_thresholds.{key}", threshold)
    
    return json.dumps({
        "success": True,
        "message": f"Updated {key} to {threshold}",
        "current_thresholds": config.load()["confidence_thresholds"]
    }, indent=2)




@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "idempotentHint": False,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def add_custom_search_query(company: str, query: str) -> str:
    """
    Add a custom search query for a specific company.
    
    Args:
        company: Company name
        query: Additional search query terms (e.g., 'IPO', 'merger', 'acquisition')
    
    Returns:
        Updated custom queries
    """
    config = ConfigStore()
    
    custom_queries = config.get("custom_search_queries", {})
    if company not in custom_queries:
        custom_queries[company] = []
    
    if query not in custom_queries[company]:
        custom_queries[company].append(query)
        config.set("custom_search_queries", custom_queries)
    
    return json.dumps({
        "success": True,
        "message": f"Added custom query '{query}' for {company}",
        "company_queries": custom_queries.get(company, [])
    }, indent=2)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "idempotentHint": False,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def update_prompt_template(prompt_name: str, new_template: str) -> str:
    """
    Update a prompt template used by the LLM.
    
    Args:
        prompt_name: Name of the prompt to update ('funding_aggregation', 'news_aggregation', 'funding_extraction')
        new_template: New prompt template with {variables}
    
    Returns:
        Confirmation message
    """
    config = ConfigStore()
    
    valid_prompts = ['funding_aggregation', 'news_aggregation', 'funding_extraction']
    if prompt_name not in valid_prompts:
        return json.dumps({"error": f"Invalid prompt_name. Use one of: {valid_prompts}"})
    
    config.update_prompt(prompt_name, new_template)
    
    prompt_data = config.get(f"prompts.{prompt_name}")
    
    return json.dumps({
        "success": True,
        "message": f"Updated {prompt_name} to version {prompt_data['version']}",
        "new_version": prompt_data['version']
    }, indent=2)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def get_system_config() -> str:
    """
    View the current system configuration.
    
    Returns:
        Full system config as JSON
    """
    config = ConfigStore()
    current_config = config.load()
    
    # Remove prompt templates from display to keep it clean
    display_config = current_config.copy()
    display_config["prompts"] = {
        k: {"version": v.get("version"), "template_length": len(v.get("template", ""))}
        for k, v in current_config.get("prompts", {}).items()
    }
    
    return json.dumps(display_config, indent=2)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def get_prompt_template(prompt_name: str) -> str:
    """
    Get a specific prompt template.
    
    Args:
        prompt_name: Name of the prompt ('funding_aggregation', 'news_aggregation', 'funding_extraction')
    
    Returns:
        The full prompt template
    """
    config = ConfigStore()
    
    prompt_data = config.get(f"prompts.{prompt_name}")
    if not prompt_data:
        return json.dumps({"error": f"Prompt '{prompt_name}' not found"})
    
    return json.dumps({
        "name": prompt_name,
        "version": prompt_data["version"],
        "template": prompt_data["template"]
    }, indent=2)



# ================= EVALUATION TOOLS =================


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def evaluate_last_run() -> str:
    """
    Evaluate the quality of the last system run.
    
    Returns:
        Evaluation metrics and report
    """
    metrics = evaluator.metrics_history[-1] if evaluator.metrics_history else None
    
    if not metrics:
        return json.dumps({"error": "No evaluation data available. Run main.py first."})
    
    report = evaluator.generate_report(metrics)
    return report


# ================= FIX =================

# @mcp.tool(
#     annotations={
#         "readOnlyHint": True,
#         "idempotentHint": False,
#         "destructiveHint": False,
#         "openWorldHint": True
#     }
# )
# def search_company_partnerships(company_name: str) -> list:
#     """
#     Search for recent news about a company related to partnerships.

#     Args:
#         company_name: Name of the company

#     Returns:
#         List of top 5 news results with title, url, snippet
#     """
#     query = f"{company_name} partnership"

#     results_list = []

#     try:
#         with DDGS() as ddgs:
#             results = ddgs.text(query, max_results=5)

#             for r in results:
#                 results_list.append({
#                     "title": r.get("title"),
#                     "url": r.get("href"),
#                     "snippet": r.get("body")
#                 })

#     except Exception as e:
#         return [{"error": str(e)}]

#     return results_list





# @mcp.tool(
#     annotations={
#         "readOnlyHint": True,
#         "idempotentHint": True,
#         "destructiveHint": False,
#         "openWorldHint": True
#     }
# )
# def explain_event_confidence(company_name: str, event_title: str) -> str:
#     """
#     Explain why an event has its confidence score.
    
#     Args:
#         company_name: Company name
#         event_title: Title of the event to explain
    
#     Returns:
#         Explanation of confidence factors
#     """
#     from utils.intent_store import IntentStore
    
#     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     db_path = os.path.join(project_root, "intents.db")
#     store = IntentStore(db_path)
    
#     # Search for the event
#     results = store.search_events_by_similarity(company_name, event_title, top_k=1)
    
#     if not results:
#         return json.dumps({"error": "Event not found"})
    
#     # Convert to expected format
#     event_data = {
#         "event": {
#             "title": results[0]["title"],
#             "confidence": results[0]["confidence"]
#         },
#         "source": results[0]["source"],
#         "date": results[0]["date"],
#         "financial_details": results[0].get("funding_details", {})
#     }
    
#     explanation = xai_engine.explain_confidence(event_data)
#     return json.dumps(explanation, indent=2)




# @mcp.tool(
#     annotations={
#         "readOnlyHint": False,
#         "idempotentHint": False,
#         "destructiveHint": False,
#         "openWorldHint": True
#     }
# )
# def manage_sources(action: str, source: str) -> str:
#     """
#     Manage trusted and blocked sources.
    
#     Args:
#         action: 'trust' to add trusted source, 'block' to block source, 'untrust' or 'unblock' to remove
#         source: Source name to add/remove
    
#     Returns:
#         Updated source lists
#     """
#     config = ConfigStore()
    
#     if action == 'trust':
#         trusted = config.get("source_preferences.trusted_sources", [])
#         if source not in trusted:
#             trusted.append(source)
#             config.set("source_preferences.trusted_sources", trusted)
#     elif action == 'block':
#         blocked = config.get("source_preferences.blocked_sources", [])
#         if source not in blocked:
#             blocked.append(source)
#             config.set("source_preferences.blocked_sources", blocked)
#     elif action == 'untrust':
#         trusted = config.get("source_preferences.trusted_sources", [])
#         if source in trusted:
#             trusted.remove(source)
#             config.set("source_preferences.trusted_sources", trusted)
#     elif action == 'unblock':
#         blocked = config.get("source_preferences.blocked_sources", [])
#         if source in blocked:
#             blocked.remove(source)
#             config.set("source_preferences.blocked_sources", blocked)
#     else:
#         return json.dumps({"error": "Invalid action. Use: trust, block, untrust, unblock"})
    
#     return json.dumps({
#         "success": True,
#         "message": f"Source '{source}' {action}ed",
#         "current_preferences": config.load()["source_preferences"]
#     }, indent=2)



if __name__ == "__main__":
    mcp.run()
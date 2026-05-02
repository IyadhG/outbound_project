import asyncio
import json
import os
import time
from datetime import datetime
from graph.funding_graph import build_funding_graph
from graph.news_graph import build_news_graph
from mcp_client.client import MCPClient
from utils.intent_store import IntentStore

# Optional evaluation imports
try:
    from evaluation.evaluator import SystemEvaluator
    from evaluation.xai import ExplainabilityEngine
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    print("Note: Evaluation module not available. Running without evaluation.")


# build graphs
funding_graph = build_funding_graph()
news_graph = build_news_graph()

# Default companies - can be overridden by external modules
COMPANIES = ["Tesla", "Rivian", "Nio"]

# Output directory structure
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
INTEL_DIR = os.path.join(OUTPUT_DIR, "company_intel")
EVALUATION_DIR = os.path.join(OUTPUT_DIR, "evaluation")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Create all output directories
for directory in [OUTPUT_DIR, INTEL_DIR, EVALUATION_DIR, METRICS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)


async def init_client():
    print(" Initializing MCP client...")
    
    client = MCPClient()
    print(" Connecting to MCP server...")
    await client.connect("mcp_server/mcp_server.py")
    print(" MCP client connected!")
    return client


async def run_company(company: str, client):
    """Run both graphs for a company sequentially in the same event loop"""
    print(f"\nProcessing {company}...")
    
    def invoke_graph(graph, state):
        """Invoke graph synchronously"""
        return graph.invoke(state)
    
    # Run funding graph
    print(f"[{company}] Starting funding graph...")
    funding_result = await asyncio.get_running_loop().run_in_executor(
        None,
        invoke_graph,
        funding_graph,
        {"company": company, "mcp_client": client}
    )
    print(f"[{company}] Funding graph completed")
    
    # Run news graph
    print(f"[{company}] Starting news graph...")
    news_result = await asyncio.get_running_loop().run_in_executor(
        None,
        invoke_graph,
        news_graph,
        {"company": company, "mcp_client": client}
    )
    print(f"[{company}] News graph completed")

    return {
        "company": company,
        "funding": funding_result.get("funding_aggregated_final", []),
        "news": news_result.get("news_final", [])
    }


def create_structured_output(results):
    """Create a well-structured output from the results"""
    
    timestamp = datetime.now().isoformat()
    
    structured_data = {
        "metadata": {
            "generated_at": timestamp,
            "total_companies": len(results),
            "companies_analyzed": [r["company"] for r in results]
        },
        "companies": {}
    }
    
    for result in results:
        company_name = result["company"]
        
        # Structure funding data
        funding_data = []
        for funding in result["funding"]:
            funding_data.append({
                "event": {
                    "title": funding.get("title", ""),
                    "confidence": funding.get("event_confidence", 0.0)
                },
                "financial_details": {
                    "amount": funding.get("amount", "Unknown"),
                    "amount_confidence": funding.get("amount_confidence", 0.0),
                    "investor": funding.get("investor", "Unknown"),
                    "investor_confidence": funding.get("investor_confidence", 0.0)
                },
                "date": {
                    "value": funding.get("date", "Unknown"),
                    "confidence": funding.get("date_confidence", 0.0)
                },
                "source": {
                    "name": funding.get("source", ""),
                    "url": funding.get("url", "")
                },
                "type": funding.get("flag", "funding")
            })
        
        # Structure news data
        news_data = []
        for news in result["news"]:
            news_data.append({
                "event": {
                    "title": news.get("title", ""),
                    "confidence": news.get("event_confidence", 0.0)
                },
                "source": {
                    "name": news.get("source", ""),
                    "url": news.get("url", "")
                },
                "date": news.get("date", "Unknown"),
                "type": news.get("flag", "news")
            })
        
        structured_data["companies"][company_name] = {
            "funding_events": funding_data,
            "news_events": news_data,
            "summary": {
                "total_funding_events": len(funding_data),
                "total_news_events": len(news_data),
                "high_confidence_funding": len([f for f in funding_data if f["event"]["confidence"] >= 0.7]),
                "high_confidence_news": len([n for n in news_data if n["event"]["confidence"] >= 0.7])
            }
        }
    
    return structured_data


def save_output(structured_data, filename=None):
    """Save the structured data to the company_intel folder"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"company_intel_{timestamp}.json"
    
    # Save to company_intel folder
    filepath = os.path.join(INTEL_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput saved to: {filepath}")
    
    # Also save a copy as "latest.json" for easy access
    latest_path = os.path.join(INTEL_DIR, "latest.json")
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
    
    return filepath


def print_formatted_output(structured_data):
    """Print the structured data in a readable format"""
    print("\n" + "="*80)
    print("Final output object")
    print("="*80)
    print(f"Generated at: {structured_data['metadata']['generated_at']}")
    print(f"Companies analyzed: {', '.join(structured_data['metadata']['companies_analyzed'])}")
    
    for company_name, data in structured_data["companies"].items():
        print("\n" + "-"*80)
        print(f" {company_name.upper()}")
        print("-"*80)
        
        summary = data["summary"]
        print(f"\nSummary:")
        print(f"  • Funding Events: {summary['total_funding_events']} ({summary['high_confidence_funding']} high confidence)")
        print(f"  • News Events: {summary['total_news_events']} ({summary['high_confidence_news']} high confidence)")
        
        if data["funding_events"]:
            print(f"\n FUNDING EVENTS:")
            for i, event in enumerate(data["funding_events"], 1):
                print(f"\n  {i}. {event['event']['title']}")
                print(f"     Confidence: {event['event']['confidence']:.0%}")
                if event["financial_details"]["amount"] != "Unknown":
                    print(f"     Amount: {event['financial_details']['amount']} (confidence: {event['financial_details']['amount_confidence']:.0%})")
                if event["financial_details"]["investor"] != "Unknown":
                    print(f"     Investor: {event['financial_details']['investor']}")
                print(f"     Source: {event['source']['name']}")
                if event['source']['url']:
                    print(f"     URL: {event['source']['url']}")
        
        if data["news_events"]:
            print(f"\n NEWS EVENTS:")
            for i, event in enumerate(data["news_events"], 1):
                print(f"\n  {i}. {event['event']['title']}")
                print(f"     Confidence: {event['event']['confidence']:.0%}")
                print(f"     Source: {event['source']['name']} | Date: {event.get('date', 'Unknown')}")
                if event['source']['url']:
                    print(f"     URL: {event['source']['url']}")
    
    print("\n" + "="*80)


def run_evaluation(structured_data, processing_time):
    """Run evaluation if module is available"""
    if not EVALUATION_AVAILABLE:
        return
    
    print("\n" + "="*60)
    print("RUNNING EVALUATION...")
    print("="*60)
    
    try:
        evaluator = SystemEvaluator()
        metrics = evaluator.evaluate_output(structured_data, processing_time)
        eval_report = evaluator.generate_report(metrics)
        print(eval_report)
        
        xai_engine = ExplainabilityEngine()
        xai_report = xai_engine.generate_explanation_report(structured_data)
        print(xai_report)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save evaluation report to evaluation folder
        eval_filename = f"evaluation_report_{timestamp}.txt"
        eval_filepath = os.path.join(EVALUATION_DIR, eval_filename)
        with open(eval_filepath, 'w', encoding='utf-8') as f:
            f.write(eval_report)
            f.write("\n\n")
            f.write(xai_report)
        print(f"Evaluation report saved to: {eval_filepath}")
        
        # Also save as latest
        latest_eval_path = os.path.join(EVALUATION_DIR, "latest_evaluation.txt")
        with open(latest_eval_path, 'w', encoding='utf-8') as f:
            f.write(eval_report)
            f.write("\n\n")
            f.write(xai_report)
        
        # Save metrics to metrics folder
        metrics_filename = f"metrics_{timestamp}.json"
        metrics_filepath = os.path.join(METRICS_DIR, metrics_filename)
        with open(metrics_filepath, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "companies_analyzed": structured_data['metadata']['companies_analyzed'],
                "metrics": {
                    "total_events": metrics.total_events,
                    "events_per_company": metrics.events_per_company,
                    "avg_confidence": metrics.avg_confidence,
                    "high_confidence_ratio": metrics.high_confidence_ratio,
                    "low_confidence_ratio": metrics.low_confidence_ratio,
                    "unique_sources": metrics.unique_sources,
                    "missing_financial_data_ratio": metrics.missing_financial_data_ratio,
                    "date_availability_ratio": metrics.date_availability_ratio,
                    "duplicate_events": metrics.duplicate_events,
                    "conflicting_info": metrics.conflicting_info,
                    "avg_processing_time": metrics.avg_processing_time,
                    "total_processing_time": metrics.total_processing_time
                }
            }, f, indent=2)
        print(f"Metrics saved to: {metrics_filepath}")
        
        # Also save metrics as latest
        latest_metrics_path = os.path.join(METRICS_DIR, "latest_metrics.json")
        with open(latest_metrics_path, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "companies_analyzed": structured_data['metadata']['companies_analyzed'],
                "metrics": {
                    "total_events": metrics.total_events,
                    "events_per_company": metrics.events_per_company,
                    "avg_confidence": metrics.avg_confidence,
                    "high_confidence_ratio": metrics.high_confidence_ratio,
                    "low_confidence_ratio": metrics.low_confidence_ratio,
                    "unique_sources": metrics.unique_sources,
                    "missing_financial_data_ratio": metrics.missing_financial_data_ratio,
                    "date_availability_ratio": metrics.date_availability_ratio,
                    "duplicate_events": metrics.duplicate_events,
                    "conflicting_info": metrics.conflicting_info,
                    "avg_processing_time": metrics.avg_processing_time,
                    "total_processing_time": metrics.total_processing_time
                }
            }, f, indent=2)
        
    except Exception as e:
        print(f"Evaluation error (non-critical): {e}")
        import traceback
        traceback.print_exc()


def save_run_log(companies, structured_data, success=True, error=None):
    """Save a run log for tracking"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_log_{timestamp}.json"
    log_filepath = os.path.join(LOGS_DIR, log_filename)
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "companies_requested": companies,
        "companies_processed": structured_data['metadata']['companies_analyzed'] if structured_data else [],
        "total_events": sum(
            data['summary']['total_funding_events'] + data['summary']['total_news_events']
            for data in (structured_data.get('companies', {}).values() if structured_data else [])
        ) if structured_data else 0,
        "error": str(error) if error else None
    }
    
    with open(log_filepath, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # Update latest log
    latest_log_path = os.path.join(LOGS_DIR, "latest_run.json")
    with open(latest_log_path, 'w') as f:
        json.dump(log_data, f, indent=2)


async def main(companies=None, save_to_file=True, evaluate=True):
    """
    Main entry point for the intent system.
    
    Args:
        companies: List of company names to analyze. If None, uses COMPANIES list.
        save_to_file: Whether to save output to JSON file.
        evaluate: Whether to run evaluation metrics.
    
    Returns:
        Structured data dictionary, or None if no results.
    """
    if companies is None:
        companies = COMPANIES

    print(f"\n{'='*60}")
    print(f"Output directories:")
    print(f"  Company Intel: {INTEL_DIR}")
    print(f"  Evaluation:    {EVALUATION_DIR}")
    print(f"  Metrics:       {METRICS_DIR}")
    print(f"  Logs:          {LOGS_DIR}")
    print(f"{'='*60}\n")

    client = await init_client()
    start_time = time.time()

    try:
        results = []
        for company in companies:
            result = await run_company(company, client)
            results.append(result)
        success = True
        error = None
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        success = False
        error = e
    finally:
        print("Shutting down...")
        try:
            await client.close()
        except Exception as e:
            print(f"Shutdown error: {e}")

    # Create structured output
    if 'results' in locals() and results:
        processing_time = time.time() - start_time
        
        structured_data = create_structured_output(results)
        
        # Store intent in database
        intent_store = IntentStore()
        intent_store.store_intent(structured_data)

        # Print formatted output
        print_formatted_output(structured_data)
        
        # Save run log
        save_run_log(companies, structured_data, success, error)
        
        # Run evaluation if requested and available
        if evaluate:
            run_evaluation(structured_data, processing_time)
        
        # Save to file
        if save_to_file:
            filename = save_output(structured_data)
            print(f"\nData structure is available in the variable 'structured_data'")
            company_names = list(structured_data['companies'].keys())
            print(f"Example access: structured_data['companies']['{company_names[0]}']['funding_events']")
        
        return structured_data
    
    # Log failed run
    if not success:
        save_run_log(companies, None, success, error)
    
    return None


if __name__ == "__main__":
    intent_result = asyncio.run(main(evaluate=True))
    
    if intent_result:
        print("\n\nRAW DATA STRUCTURE:")
        print(json.dumps(intent_result, indent=2, default=str))
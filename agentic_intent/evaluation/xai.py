import json
from typing import Dict, List, Tuple
from datetime import datetime
from evaluation.evaluator import EvaluationMetrics


class ExplainabilityEngine:
    """Provides explanations for system decisions"""
    
    def explain_confidence(self, event: Dict) -> Dict:
        """
        Explain why an event has its confidence score
        """
        reasons = []
        confidence = event.get("event", {}).get("confidence", 0)
        
        # Check source reliability
        source = event.get("source", {}).get("name", "unknown")
        trusted_sources = ["Reuters", "Bloomberg", "Financial Times", "SEC", "NYTimes"]
        if any(ts in source for ts in trusted_sources):
            reasons.append(f"✅ From trusted source: {source}")
        else:
            reasons.append(f"⚠️ Unverified source: {source}")
        
        # Check data completeness
        if event.get("financial_details", {}).get("amount") not in [None, "", "Unknown"]:
            reasons.append("✅ Includes specific amount")
        else:
            reasons.append("⚠️ Missing amount information")
        
        if event.get("financial_details", {}).get("investor") not in [None, "", "Unknown"]:
            reasons.append("✅ Identifies investor")
        else:
            reasons.append("⚠️ Missing investor information")
        
        # Check date
        date = event.get("date", {}).get("value")
        if date and date not in ["Unknown", ""]:
            reasons.append("✅ Has specific date")
        else:
            reasons.append("⚠️ Missing date")
        
        # Check cross-referencing
        date_conf = event.get("date", {}).get("confidence", 0)
        amount_conf = event.get("financial_details", {}).get("amount_confidence", 0)
        
        supporting_factors = sum([
            date_conf > 0.5,
            amount_conf > 0.5
        ])
        
        if supporting_factors >= 2:
            reasons.append("✅ Multiple high-confidence details")
        elif supporting_factors == 1:
            reasons.append("⚠️ Only one high-confidence detail")
        else:
            reasons.append("❌ No high-confidence details")
        
        return {
            "event_title": event.get("event", {}).get("title"),
            "final_confidence": confidence,
            "confidence_breakdown": reasons,
            "suggestion": "Consider verifying with additional sources" if confidence < 0.5 else "Event appears reliable"
        }
    
    def explain_aggregation(self, old_events: List[Dict], new_event: Dict) -> Dict:
        """
        Explain why events were grouped together
        """
        explanation = {
            "new_event": new_event.get("event", {}).get("title"),
            "matched_with": [],
            "matching_factors": []
        }
        
        new_title = new_event.get("event", {}).get("title", "").lower()
        new_amount = new_event.get("financial_details", {}).get("amount", "")
        new_investor = new_event.get("financial_details", {}).get("investor", "")
        
        for old_event in old_events:
            old_title = old_event.get("title", "").lower()
            old_amount = old_event.get("amount", "")
            old_investor = old_event.get("investor", "")
            
            match_reasons = []
            
            # Title similarity (simple keyword overlap)
            new_words = set(new_title.split())
            old_words = set(old_title.split())
            overlap = len(new_words & old_words) / max(len(new_words | old_words), 1)
            
            if overlap > 0.3:
                match_reasons.append(f"Title similarity: {overlap:.0%}")
            
            # Amount match
            if new_amount and old_amount and new_amount == old_amount:
                match_reasons.append(f"Matching amount: {new_amount}")
            
            # Investor match
            if new_investor and old_investor and new_investor == old_investor:
                match_reasons.append(f"Matching investor: {new_investor}")
            
            if match_reasons:
                explanation["matched_with"].append({
                    "event": old_event.get("title"),
                    "reasons": match_reasons
                })
        
        explanation["matching_factors"] = list(set(
            reason for match in explanation["matched_with"]
            for reason in match["reasons"]
        ))
        
        return explanation
    
    def trace_event_creation(self, raw_data: Dict, final_event: Dict) -> Dict:
        """
        Trace how a final event was created from raw data
        """
        trace = {
            "final_output": final_event.get("event", {}).get("title"),
            "pipeline_steps": [
                {
                    "step": "1. Data Fetching",
                    "description": f"Raw search results obtained"
                },
                {
                    "step": "2. Normalization",
                    "description": f"Data cleaned and structured"
                },
                {
                    "step": "3. Information Extraction",
                    "description": "LLM extracted: " + json.dumps({
                        "amount": final_event.get("financial_details", {}).get("amount"),
                        "investor": final_event.get("financial_details", {}).get("investor"),
                        "date": final_event.get("date", {}).get("value")
                    })
                },
                {
                    "step": "4. Aggregation",
                    "description": f"Events grouped with confidence: {final_event.get('event', {}).get('confidence')}"
                },
                {
                    "step": "5. Finalization",
                    "description": "Output formatted and validated"
                }
            ],
            "data_quality": {
                "source": final_event.get("source", {}).get("name"),
                "has_amount": final_event.get("financial_details", {}).get("amount") not in [None, "", "Unknown"],
                "has_date": final_event.get("date", {}).get("value") not in [None, "", "Unknown"],
                "has_investor": final_event.get("financial_details", {}).get("investor") not in [None, "", "Unknown"]
            }
        }
        
        return trace
    
    def generate_explanation_report(self, structured_data: Dict) -> str:
        """Generate a comprehensive explainability report"""
        report = f"""
{'='*60}
EXPLAINABILITY REPORT
{'='*60}
Timestamp: {datetime.now().isoformat()}

"""
        for company, data in structured_data.get("companies", {}).items():
            report += f"\n{'='*40}\n{company.upper()}\n{'='*40}\n"
            
            # Funding events
            for i, event in enumerate(data.get("funding_events", []), 1):
                report += f"\n📊 Funding Event {i}: {event['event']['title']}\n"
                conf_explanation = self.explain_confidence(event)
                for reason in conf_explanation["confidence_breakdown"]:
                    report += f"  {reason}\n"
                report += f"  💡 {conf_explanation['suggestion']}\n"
            
            # News events
            for i, event in enumerate(data.get("news_events", []), 1):
                report += f"\n📰 News Event {i}: {event['event']['title']}\n"
                report += f"  Source: {event['source']['name']}\n"
                report += f"  Date: {event.get('date', 'Unknown')}\n"
        
        report += f"\n{'='*60}\n"
        return report


class ABTester:
    """A/B testing for prompt/config changes"""
    
    def __init__(self):
        self.tests = []
    
    def create_test(self, name: str, variant_a: Dict, variant_b: Dict) -> Dict:
        """Create an A/B test"""
        test = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "variant_a": variant_a,
            "variant_b": variant_b,
            "results_a": None,
            "results_b": None,
            "winner": None
        }
        self.tests.append(test)
        return test
    
    def record_result(self, test_name: str, variant: str, metrics: EvaluationMetrics):
        """Record results for a variant"""
        for test in self.tests:
            if test["name"] == test_name:
                if variant == "a":
                    test["results_a"] = metrics
                else:
                    test["results_b"] = metrics
    
    def determine_winner(self, test_name: str) -> Dict:
        """Compare variants and determine winner"""
        for test in self.tests:
            if test["name"] == test_name:
                if not test["results_a"] or not test["results_b"]:
                    return {"error": "Both variants not yet tested"}
                
                a = test["results_a"]
                b = test["results_b"]
                
                # Scoring criteria
                score_a = (
                    a.avg_confidence * 0.3 +
                    a.high_confidence_ratio * 0.2 +
                    (1 - a.missing_financial_data_ratio) * 0.3 +
                    a.date_availability_ratio * 0.2
                )
                
                score_b = (
                    b.avg_confidence * 0.3 +
                    b.high_confidence_ratio * 0.2 +
                    (1 - b.missing_financial_data_ratio) * 0.3 +
                    b.date_availability_ratio * 0.2
                )
                
                winner = "a" if score_a > score_b else "b"
                test["winner"] = winner
                
                return {
                    "test_name": test_name,
                    "winner": winner,
                    "score_a": score_a,
                    "score_b": score_b,
                    "difference": abs(score_a - score_b),
                    "recommendation": f"Use variant {winner.upper()}"
                }
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating system output"""
    # Coverage metrics
    total_events: int
    events_per_company: Dict[str, int]
    
    # Quality metrics
    avg_confidence: float
    high_confidence_ratio: float  # Events with confidence > 0.7
    low_confidence_ratio: float   # Events with confidence < 0.5
    
    # Source diversity
    unique_sources: int
    source_distribution: Dict[str, int]
    
    # Completeness
    missing_financial_data_ratio: float  # Events missing amount/investor
    date_availability_ratio: float       # Events with dates
    
    # Consistency
    duplicate_events: int  # Same event reported multiple times
    conflicting_info: int  # Events with contradicting amounts/dates
    
    # Latency
    avg_processing_time: float
    total_processing_time: float


class SystemEvaluator:
    """Evaluates the intent system's output quality"""
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_output(self, structured_data: Dict, processing_time: float = 0) -> EvaluationMetrics:
        """Evaluate a single run's output"""
        companies_data = structured_data.get("companies", {})
        
        all_events = []
        events_per_company = {}
        confidence_scores = []
        sources = []
        missing_financial = 0
        total_funding = 0
        dates_present = 0
        total_events = 0
        
        for company, data in companies_data.items():
            funding = data.get("funding_events", [])
            news = data.get("news_events", [])
            
            company_events = len(funding) + len(news)
            events_per_company[company] = company_events
            total_events += company_events
            
            # Analyze funding events
            for event in funding:
                total_funding += 1
                confidence = event["event"]["confidence"]
                confidence_scores.append(confidence)
                
                sources.append(event["source"]["name"])
                
                # Check financial data completeness
                fin_details = event.get("financial_details", {})
                if fin_details.get("amount") in ["Unknown", "", None] or \
                   fin_details.get("investor") in ["Unknown", "", None]:
                    missing_financial += 1
                
                # Check date availability
                if event.get("date", {}).get("value") not in ["", None, "Unknown"]:
                    dates_present += 1
            
            # Analyze news events
            for event in news:
                confidence = event["event"]["confidence"]
                confidence_scores.append(confidence)
                sources.append(event["source"]["name"])
                
                if event.get("date") not in ["", None, "Unknown"]:
                    dates_present += 1
        
        # Calculate metrics
        avg_conf = mean(confidence_scores) if confidence_scores else 0
        high_conf = sum(1 for c in confidence_scores if c > 0.7) / max(len(confidence_scores), 1)
        low_conf = sum(1 for c in confidence_scores if c < 0.5) / max(len(confidence_scores), 1)
        
        source_dist = {}
        for s in sources:
            source_dist[s] = source_dist.get(s, 0) + 1
        
        metrics = EvaluationMetrics(
            total_events=total_events,
            events_per_company=events_per_company,
            avg_confidence=avg_conf,
            high_confidence_ratio=high_conf,
            low_confidence_ratio=low_conf,
            unique_sources=len(set(sources)),
            source_distribution=source_dist,
            missing_financial_data_ratio=missing_financial / max(total_funding, 1),
            date_availability_ratio=dates_present / max(total_events, 1),
            duplicate_events=self._detect_duplicates(structured_data),
            conflicting_info=self._detect_conflicts(structured_data),
            avg_processing_time=processing_time / max(total_events, 1) if total_events else 0,
            total_processing_time=processing_time
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _detect_duplicates(self, data: Dict) -> int:
        """Detect potential duplicate events"""
        duplicates = 0
        titles_seen = set()
        
        for company_data in data.get("companies", {}).values():
            all_events = company_data.get("funding_events", []) + company_data.get("news_events", [])
            for event in all_events:
                title = event["event"]["title"].lower()[:50]
                if title in titles_seen:
                    duplicates += 1
                titles_seen.add(title)
        
        return duplicates
    
    def _detect_conflicts(self, data: Dict) -> int:
        """Detect conflicting information (different amounts for same event)"""
        conflicts = 0
        # Simple heuristic: same investor + different amounts = conflict
        for company_data in data.get("companies", {}).values():
            funding_events = company_data.get("funding_events", [])
            investors_seen = {}
            
            for event in funding_events:
                investor = event.get("financial_details", {}).get("investor", "")
                amount = event.get("financial_details", {}).get("amount", "")
                
                if investor and amount and investor != "Unknown":
                    if investor in investors_seen and investors_seen[investor] != amount:
                        conflicts += 1
                    investors_seen[investor] = amount
        
        return conflicts
    
    def generate_report(self, metrics: EvaluationMetrics = None) -> str:
        """Generate a human-readable evaluation report"""
        if metrics is None and self.metrics_history:
            metrics = self.metrics_history[-1]
        elif metrics is None:
            return "No metrics available"
        
        report = f"""
{'='*60}
SYSTEM EVALUATION REPORT
{'='*60}
Timestamp: {datetime.now().isoformat()}

COVERAGE:
- Total Events: {metrics.total_events}
- Events per Company: {json.dumps(metrics.events_per_company, indent=2)}

QUALITY:
- Average Confidence: {metrics.avg_confidence:.2%}
- High Confidence Events (>0.7): {metrics.high_confidence_ratio:.2%}
- Low Confidence Events (<0.5): {metrics.low_confidence_ratio:.2%}
- Unique Sources: {metrics.unique_sources}

COMPLETENESS:
- Missing Financial Data: {metrics.missing_financial_data_ratio:.2%}
- Date Availability: {metrics.date_availability_ratio:.2%}
- Potential Duplicates: {metrics.duplicate_events}
- Conflicting Info: {metrics.conflicting_info}

PERFORMANCE:
- Total Processing Time: {metrics.total_processing_time:.2f}s
- Avg Time per Event: {metrics.avg_processing_time:.3f}s

TOP SOURCES:
{chr(10).join(f'- {k}: {v}' for k, v in sorted(metrics.source_distribution.items(), key=lambda x: x[1], reverse=True)[:5])}
{'='*60}
"""
        return report
    
    def compare_runs(self) -> Dict:
        """Compare metrics across multiple runs"""
        if len(self.metrics_history) < 2:
            return {"message": "Need at least 2 runs to compare"}
        
        prev = self.metrics_history[-2]
        curr = self.metrics_history[-1]
        
        return {
            "confidence_change": curr.avg_confidence - prev.avg_confidence,
            "events_change": curr.total_events - prev.total_events,
            "source_diversity_change": curr.unique_sources - prev.unique_sources,
            "completeness_improvement": prev.missing_financial_data_ratio - curr.missing_financial_data_ratio,
            "speed_change": prev.total_processing_time - curr.total_processing_time
        }
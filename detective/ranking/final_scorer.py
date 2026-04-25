"""
Final Scorer - Calculate final scores with LLM-evaluated intent boost
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from groq import Groq

logger = logging.getLogger(__name__)


class FinalScorer:
    """Calculate final scores combining similarity and LLM-evaluated intent signals."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        """Initialize with Groq LLM for intent evaluation."""
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not provided")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"Final Scorer initialized with model: {model}")
    
    def evaluate_intent_with_llm(
        self,
        company: Dict[str, Any],
        intent_signals: List[Dict[str, Any]]
    ) -> float:
        """Use LLM to evaluate if intent signals are useful and relevant."""
        if not intent_signals:
            return 0.0
        
        company_name = company.get('company_name', '')
        company_industries = company.get('company_data', {}).get('industries', [])
        
        # Build signal summary
        signal_text = "\n".join([
            f"- {s.get('type', 'unknown')}: {s.get('description', '')} (confidence: {s.get('confidence', 0)})"
            for s in intent_signals
        ])
        
        prompt = f"""Evaluate the following intent signals for {company_name} in industries {', '.join(company_industries)}.

Intent Signals:
{signal_text}

Rate the overall usefulness and relevance of these signals for B2B sales outreach on a scale of 0.0 to 1.0.

Consider:
- Are the signals relevant to the company's industry?
- Do they indicate real buying intent or just noise?
- Would these signals help identify a good sales opportunity?

Respond with ONLY a number between 0.0 and 1.0."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You evaluate B2B intent signals. Return only a number 0.0-1.0."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip()
            # Extract number
            import re
            match = re.search(r'(\d+\.?\d*)', result)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)
        except Exception as e:
            logger.error(f"LLM intent evaluation failed: {e}")
        
        return 0.0
    
    def calculate_final_scores(
        self,
        ranked_companies: List[Dict[str, Any]],
        intent_results: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Calculate final scores with LLM-evaluated intent boost."""
        final_scores = []
        
        for company in ranked_companies:
            company_key = company['company_key']
            base_score = company['similarity_score']  # From CompanyRanker cosine similarity
            
            # Get intent signals for this company
            intent_signals = []
            
            if intent_results and company_key in intent_results:
                intent_signals = intent_results[company_key]
            
            # Use LLM to evaluate intent usefulness
            llm_intent_score = self.evaluate_intent_with_llm(company, intent_signals)
            
            # Calculate final score: base + intent bonus (max 0.2 boost)
            intent_boost = llm_intent_score * 0.2
            final_score = min(base_score + intent_boost, 1.0)  # Cap at 1.0
            
            final_scores.append({
                'company_key': company_key,
                'company_name': company['company_name'],
                'similarity_score': base_score,
                'llm_intent_score': llm_intent_score,
                'intent_signals': intent_signals[:3],  # Top 3 signals
                'final_score': round(final_score, 4),
                'company_data': company.get('company_data', {})
            })
            
            logger.debug(f"Final score for {company_key}: similarity={base_score:.3f}, "
                        f"intent={llm_intent_score:.3f}, final={final_score:.4f}")
        
        # Sort by final score
        final_scores.sort(key=lambda x: x['final_score'], reverse=True)
        
        logger.info(f"Calculated final scores for {len(final_scores)} companies")
        return final_scores
    
    def filter_by_employee_range(
        self,
        scored_companies: List[Dict[str, Any]],
        min_employees: float,
        max_employees: float
    ) -> List[Dict[str, Any]]:
        """Filter companies by employee count."""
        filtered = []
        
        for company in scored_companies:
            company_data = company.get('company_data', {})
            basic_info = company_data.get('basic_info', {})
            employees = basic_info.get('employees') or company_data.get('employees', 0)
            
            if employees:
                emp_val = float(employees) if isinstance(employees, (int, float, str)) else 0
                if min_employees <= emp_val <= max_employees:
                    filtered.append(company)
                    logger.debug(f"Keeping {company['company_key']}: {emp_val} employees")
                else:
                    logger.debug(f"Dropping {company['company_key']}: {emp_val} employees (outside {min_employees}-{max_employees})")
            else:
                # Keep if no employee data
                filtered.append(company)
        
        logger.info(f"Employee filter: {len(filtered)}/{len(scored_companies)} companies remain")
        return filtered
    
    def save_final_ranking(self, final_scores: List[Dict[str, Any]], output_path: str):
        """Save final ranking to file."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean output (remove full company data)
        clean_scores = []
        for item in final_scores:
            clean_scores.append({
                'company_key': item['company_key'],
                'company_name': item['company_name'],
                'similarity_score': item.get('similarity_score', 0),
                'intent_score': item.get('llm_intent_score', item.get('intent_score', 0)),
                'final_score': item.get('final_score', 0),
                'rank': len(clean_scores) + 1
            })
        
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(clean_scores, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved final ranking to: {output}")
    
    def apply(
        self,
        ranked_companies: List[Dict[str, Any]],
        intent_results: Optional[Dict[str, Any]] = None,
        employee_range: Optional[tuple] = None,
        output_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Full pipeline: score, filter, and save."""
        final_scores = self.calculate_final_scores(ranked_companies, intent_results)
        
        # Apply employee filter if specified
        if employee_range and len(employee_range) == 2:
            min_emp, max_emp = employee_range
            final_scores = self.filter_by_employee_range(final_scores, min_emp, max_emp)
            # Re-sort after filtering
            final_scores.sort(key=lambda x: x['final_score'], reverse=True)
        
        if output_path:
            self.save_final_ranking(final_scores, output_path)
        
        return final_scores

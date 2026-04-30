"""
Company ranking with LLM-based sentence construction and semantic similarity.
"""

import os
import logging
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

try:
    from groq import Groq
except ImportError:
    Groq = None

from .embedder import GeminiEmbedder

logger = logging.getLogger(__name__)


class CompanyRanker:
    """Ranks companies using LLM sentence construction + embeddings + LLM intent boost."""
    
    def __init__(self, embedder: Optional[GeminiEmbedder] = None, llm_model: str = "llama-3.1-8b-instant"):
        """Initialize with optional embedder and LLM model."""
        self.embedder = embedder
        self.llm_model = llm_model
        
        # Initialize Groq client
        if Groq:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.llm = Groq(api_key=api_key)
            else:
                logger.warning("GROQ_API_KEY not found")
                self.llm = None
        else:
            self.llm = None
        
        self._icp_embedding = None
        
        if not self.embedder:
            logger.warning("CompanyRanker initialized without embedder")
        else:
            logger.info("CompanyRanker initialized with provided embedder")
    
    def _build_icp_text(self, icp: Dict[str, Any]) -> str:
        """Build descriptive text for ICP."""
        parts = []
        
        if 'industries' in icp:
            parts.append(f"Industries: {', '.join(icp['industries'])}")
        if 'technologies' in icp:
            parts.append(f"Technologies: {', '.join(icp['technologies'])}")
        if 'size_range' in icp:
            sr = icp['size_range']
            parts.append(f"Company size: {sr.get('min', 'N/A')} - {sr.get('max', 'N/A')} employees")
        if 'geography' in icp:
            geo = icp['geography']
            parts.append(f"Countries: {', '.join(geo.get('countries', []))}")
        if 'business_model' in icp:
            parts.append(f"Business model: {', '.join(icp['business_model'])}")
        if 'target_roles' in icp:
            parts.append(f"Target roles: {', '.join(icp['target_roles'])}")
        
        return " | ".join(parts)
    
    def _extract_company_data(self, company: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key company data fields."""
        basic = company.get('basic_info', {})
        
        # Get revenue
        revenue = basic.get('annual_revenue') or company.get('annual_revenue', 0)
        
        # Get location
        hq = basic.get('headquarters', '')
        city = ''
        country = ''
        if hq:
            parts = hq.split(',')
            city = parts[0].strip() if parts else ''
            country = parts[-1].strip() if len(parts) > 1 else ''
        
        # Get employees
        employees = basic.get('employees') or company.get('estimated_num_employees', 0)
        if isinstance(employees, str):
            try:
                employees = int(employees.replace(',', ''))
            except:
                employees = 0
        
        # Get founded year
        founded = basic.get('founded_year') or company.get('founded', '')
        
        # Get technologies
        techs = company.get('technologies', [])
        tech_names = [t.get('name', '') for t in techs[:10] if t.get('name')]
        
        # Get keywords/description
        keywords = company.get('keywords', [])
        if not keywords:
            desc = basic.get('description', '')
            keywords = desc[:200].split()[:20] if desc else []
        
        return {
            'name': basic.get('name', ''),
            'revenue': revenue,
            'city': city,
            'country': country,
            'founded_year': founded,
            'employees': employees,
            'technologies': tech_names,
            'keywords': keywords[:10],
            'raw_data': company
        }
    
    def _construct_company_sentence_with_llm(self, company_data: Dict[str, Any]) -> str:
        """Use LLM to construct a descriptive sentence about the company."""
        if not self.llm:
            # Fallback to manual construction
            return self._construct_company_sentence_manual(company_data)
        
        try:
            # Safely format values, handling None and non-numeric types
            revenue = company_data.get('revenue', 0) or 0
            employees = company_data.get('employees', 0) or 0
            founded = company_data.get('founded_year', '') or ''
            city = company_data.get('city', '') or ''
            country = company_data.get('country', '') or ''
            name = company_data.get('name', '') or ''
            techs = company_data.get('technologies', []) or []
            keywords = company_data.get('keywords', []) or []
            
            # Format numbers safely
            revenue_str = f"${revenue:,.0f}" if isinstance(revenue, (int, float)) else "Unknown"
            employees_str = f"{employees:,.0f}" if isinstance(employees, (int, float)) else "Unknown"
            
            prompt = f"""Create a concise 1-2 sentence description of this company for semantic comparison:

Company: {name}
Annual Revenue: {revenue_str}
Location: {city}, {country}
Founded: {founded}
Employees: {employees_str}
Technologies: {', '.join(techs)}
Keywords: {', '.join(keywords)}

Write only the descriptive sentence(s), no additional commentary."""

            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            sentence = response.choices[0].message.content.strip()
            logger.debug(f"LLM constructed sentence for {company_data['name']}: {sentence[:100]}...")
            return sentence
            
        except Exception as e:
            logger.warning(f"LLM sentence construction failed, using manual: {e}")
            return self._construct_company_sentence_manual(company_data)
    
    def _construct_company_sentence_manual(self, company_data: Dict[str, Any]) -> str:
        """Manual fallback for company sentence construction."""
        parts = []
        
        # Safely get values
        name = company_data.get('name', '') or ''
        founded = company_data.get('founded_year', '') or ''
        employees = company_data.get('employees', 0) or 0
        city = company_data.get('city', '') or ''
        country = company_data.get('country', '') or ''
        techs = company_data.get('technologies', []) or []
        keywords = company_data.get('keywords', []) or []
        
        if name:
            parts.append(f"{name}")
        if founded:
            parts.append(f"founded in {founded}")
        if employees and isinstance(employees, (int, float)):
            parts.append(f"with {employees:,.0f} employees")
        elif employees:
            parts.append(f"with {employees} employees")
        if city or country:
            loc = f"based in {city}" if city else ""
            if country:
                loc += f", {country}" if loc else f"based in {country}"
            parts.append(loc.strip())
        if techs:
            parts.append(f"uses technologies: {', '.join(techs[:5])}")
        if keywords:
            parts.append(f"focuses on: {', '.join(keywords[:5])}")
        
        return ". ".join(parts) if parts else "Company information unavailable"
    
    def _analyze_intent_with_llm(self, company_data: Dict[str, Any], company_sentence: str) -> float:
        """Use LLM to analyze how useful/intentful this company is for the ICP."""
        if not self.llm:
            return 0.0
        
        try:
            # Safely get values
            name = company_data.get('name', '') or ''
            revenue = company_data.get('revenue', 0) or 0
            employees = company_data.get('employees', 0) or 0
            techs = company_data.get('technologies', []) or []
            
            # Format safely
            revenue_str = f"${revenue:,.0f}" if isinstance(revenue, (int, float)) else "Unknown"
            employees_str = f"{employees:,.0f}" if isinstance(employees, (int, float)) else "Unknown"
            
            prompt = f"""Analyze this company profile and determine if it shows strong buying intent signals.

Company: {name}
Description: {company_sentence}
Annual Revenue: {revenue_str}
Employees: {employees_str}
Technologies: {', '.join(techs)}

Score from 0.0 to 1.0 based on these intent signals:
- High revenue + growth (0.2-0.4)
- Modern technology stack (0.1-0.3)
- Clear business focus matching keywords (0.1-0.3)
- Company size indicating buying power (0.1-0.2)

Respond with ONLY a number between 0.0 and 1.0."""

            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            content = response.choices[0].message.content.strip()
            # Extract number
            match = re.search(r'(\d+\.?\d*)', content)
            if match:
                score = float(match.group(1))
                score = max(0.0, min(1.0, score))
                logger.debug(f"LLM intent score for {company_data['name']}: {score}")
                return score
            return 0.0
            
        except Exception as e:
            logger.warning(f"LLM intent analysis failed: {e}")
            return 0.0
    
    def embed_icp(self, icp_text: str) -> None:
        """Pre-embed ICP text for ranking (stores for later use)."""
        if not self.embedder:
            raise ValueError("No embedder configured for ranking")
        self._icp_embedding = self.embedder.embed_text(icp_text)
        logger.info("ICP embedded successfully")
    
    def rank_companies(
        self,
        companies: Dict[str, Any],
        icp: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Rank companies by similarity to ICP with LLM sentence construction and intent boost."""
        if not self.embedder:
            raise ValueError("No embedder configured for ranking")
        
        if not companies:
            logger.warning("No companies to rank")
            return []
        
        # Get ICP embedding (pre-embedded or create new)
        icp_embedding = None
        if self._icp_embedding is not None:
            icp_embedding = self._icp_embedding
            logger.info("Using pre-embedded ICP")
        elif icp:
            icp_text = self._build_icp_text(icp)
            icp_embedding = self.embedder.embed_text(icp_text)
            logger.info(f"ICP embedded successfully")
        else:
            raise ValueError("No ICP provided and no pre-embedded ICP available")
        
        # Score each company
        ranked = []
        logger.info(f"Ranking {len(companies)} companies...")
        
        for company_key, company in companies.items():
            try:
                # Extract company data
                company_data = self._extract_company_data(company)
                
                # Use LLM to construct descriptive sentence
                company_sentence = self._construct_company_sentence_with_llm(company_data)
                
                # Embed the constructed sentence
                company_embedding = self.embedder.embed_text(company_sentence)
                
                # Calculate cosine similarity
                from scipy.spatial.distance import cosine
                similarity_score = 1 - cosine(icp_embedding, company_embedding)
                
                # Use LLM to analyze intent signals and add boost
                intent_score = self._analyze_intent_with_llm(company_data, company_sentence)
                
                # Combine scores: similarity + intent boost (capped at 1.0)
                final_score = min(similarity_score + (intent_score * 0.2), 1.0)
                
                ranked.append({
                    'company_key': company_key,
                    'company_name': company_data['name'] or company_key,
                    'similarity_score': similarity_score,
                    'intent_score': intent_score,
                    'final_score': final_score,
                    'company_sentence': company_sentence,
                    'company_data': company
                })
                
                logger.info(f"Ranked: {company_data['name'] or company_key} - Similarity: {similarity_score:.3f}, Intent: {intent_score:.3f}, Final: {final_score:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to rank {company_key}: {e}")
                continue
        
        # Sort by final score descending
        ranked.sort(key=lambda x: x['final_score'], reverse=True)
        
        logger.info(f"Ranking complete. Top company: {ranked[0]['company_name'] if ranked else 'None'}")
        return ranked
    
    def save_ranking(self, ranked: List[Dict[str, Any]], output_path: str):
        """Save ranking results to file."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove full company data for cleaner output
        clean_ranked = []
        for item in ranked:
            clean_ranked.append({
                'company_key': item['company_key'],
                'company_name': item['company_name'],
                'similarity_score': item['similarity_score']
            })
        
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(clean_ranked, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved ranking to: {output}")
    
    def apply(
        self,
        companies: Dict[str, Any],
        icp: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Full pipeline: rank and optionally save."""
        ranked = self.rank_companies(companies, icp)
        
        if output_path and ranked:
            self.save_ranking(ranked, output_path)
        
        return ranked

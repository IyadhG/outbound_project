"""
Persona Ranker - Rank personas with LLM-based seniority and position analysis
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from groq import Groq

logger = logging.getLogger(__name__)


class PersonaRanker:
    """Rank personas within companies using LLM for seniority and position analysis."""
    
    def __init__(self, target_roles: Optional[List[str]] = None, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        """Initialize with target roles and LLM."""
        self.target_roles = [r.lower() for r in (target_roles or [])]
        
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not provided")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"PersonaRanker initialized with {len(self.target_roles)} target roles, model: {model}")
    
    def load_personas_for_company(self, company_key: str, personas_folder: str) -> List[Dict[str, Any]]:
        """Load personas for a specific company."""
        personas = []
        folder = Path(personas_folder)
        
        # Look for company-specific persona file
        persona_file = folder / f"{company_key}_personas.json"
        
        if not persona_file.exists():
            logger.warning(f"No personas file found for {company_key}")
            return personas
        
        try:
            with open(persona_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                personas = data if isinstance(data, list) else data.get('personas', [])
            logger.debug(f"Loaded {len(personas)} personas for {company_key}")
        except Exception as e:
            logger.error(f"Failed to load personas for {company_key}: {e}")
        
        return personas
    
    def _extract_persona_fields(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """Extract persona fields from the JSON structure."""
        # Get name
        name = persona.get('full_name', '')
        if not name:
            first = persona.get('first_name', '')
            last = persona.get('last_name', '')
            name = f"{first} {last}".strip()
        
        # Get job title components
        job_role = persona.get('job_title_role', '')
        job_level = persona.get('job_title_level', '')
        job_title = f"{job_level} {job_role}".strip() if job_level and job_role else job_role or job_level
        
        # Get department from job_role or industry
        department = job_role or persona.get('industry', '')
        
        # Get job description
        job_desc = persona.get('job_description', '')
        
        # Get experience summary
        experience = persona.get('experience', [])
        exp_summary = ""
        if experience and len(experience) > 0:
            current = experience[0]
            exp_title = current.get('title', '')
            exp_company = current.get('company', {}).get('name', '')
            exp_summary = f"Currently {exp_title} at {exp_company}"
        
        # Get email
        emails = persona.get('emails', [])
        email = emails[0].get('address', '') if emails else persona.get('email', '')
        
        # Get LinkedIn
        linkedin = persona.get('linkedin_url', '')
        
        return {
            'name': name or 'Unknown',
            'job_title': job_title or 'Unknown',
            'department': department or 'Unknown',
            'job_description': job_desc or '',
            'experience_summary': exp_summary,
            'email': email,
            'linkedin': linkedin
        }
    
    def analyze_persona_with_llm(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze persona seniority and position."""
        fields = self._extract_persona_fields(persona)
        
        target_roles_text = ', '.join(self.target_roles) if self.target_roles else 'None specified'
        
        prompt = f"""Analyze this B2B persona for lead scoring:

Name: {fields['name']}
Job Title: {fields['job_title']}
Department/Role: {fields['department']}
Job Description: {fields['job_description'][:200]}
Experience: {fields['experience_summary']}
Target Roles to Match: {target_roles_text}

Rate the following on a scale of 0.0 to 1.0:
1. Seniority level (0.0 = junior/entry, 1.0 = C-suite/VP/Director)
2. Strategic position for B2B sales (0.0 = low influence, 1.0 = decision maker)
3. Match with target roles (0.0 = no match, 1.0 = perfect match)

Respond in this exact format:
Seniority: X.XX
Position: X.XX
TargetMatch: X.XX"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You analyze B2B personas. Return only the three scores in the specified format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse scores
            import re
            seniority = re.search(r'Seniority:\s*(\d+\.?\d*)', result)
            position = re.search(r'Position:\s*(\d+\.?\d*)', result)
            target_match = re.search(r'TargetMatch:\s*(\d+\.?\d*)', result)
            
            return {
                'seniority_score': float(seniority.group(1)) if seniority else 0.5,
                'position_score': float(position.group(1)) if position else 0.5,
                'target_match_score': float(target_match.group(1)) if target_match else 0.0
            }
        except Exception as e:
            logger.error(f"LLM persona analysis failed: {e}")
            return {'seniority_score': 0.5, 'position_score': 0.5, 'target_match_score': 0.0}
    
    def score_persona(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single persona with sales-first logic and target role bonus."""
        # Extract fields properly
        fields = self._extract_persona_fields(persona)
        
        job_title = fields['job_title']
        name = fields['name']
        department = fields['department']
        job_title_lower = job_title.lower()
        department_lower = department.lower()
        
        # Check if sales department (primary target)
        is_sales_dept = any(term in department_lower for term in ['sales', 'business development', 'bd', 'revenue', 'commercial'])
        
        # Check if CEO/Founder (fallback target)
        is_ceo = any(term in job_title_lower for term in ['ceo', 'founder', 'co-founder', 'chief executive', 'owner'])
        
        # Check if matches target roles from ICP
        target_match = False
        target_bonus = 0.0
        if self.target_roles:
            for role in self.target_roles:
                if role.lower() in job_title_lower:
                    target_match = True
                    target_bonus = 0.05  # Boost for matching target role
                    break
        
        # Sales department gets priority scoring
        sales_bonus = 0.1 if is_sales_dept else 0.0
        
        # CEO gets fallback bonus (lower than sales)
        ceo_bonus = 0.03 if is_ceo else 0.0
        
        # Engagement likelihood
        engagement = persona.get('is_likely_to_engage', 0.5)
        
        # Intent strength (1-10)
        intent = persona.get('intent_strength', 5) / 10
        
        # Seniority score based on job title
        seniority_score = 0.5
        if any(term in job_title_lower for term in ['cfo', 'cto', 'coo', 'cmo', 'chief', 'vp', 'vice president', 'director', 'head of']):
            seniority_score = 0.8
        elif any(term in job_title_lower for term in ['manager', 'lead', 'senior']):
            seniority_score = 0.6
        elif any(term in job_title_lower for term in ['ceo', 'founder']):
            seniority_score = 1.0
        
        # Calculate final score with weights
        # Sales department prioritized, target roles get bonus, CEO as fallback
        final_score = (
            seniority_score * 0.20 +           # Seniority
            engagement * 0.15 +                # Engagement
            intent * 0.15 +                    # Intent
            sales_bonus +                      # Sales priority (0.1)
            ceo_bonus +                        # CEO fallback (0.03)
            target_bonus                       # Target role match (0.05)
        )
        
        return {
            'name': name,
            'job_title': job_title,
            'department': department,
            'job_description': fields['job_description'][:100] if fields['job_description'] else '',
            'location': persona.get('city', '') or persona.get('country', ''),
            'email': fields['email'],
            'linkedin': fields['linkedin'],
            'engagement_likelihood': engagement,
            'intent_strength': persona.get('intent_strength', 5),
            'is_sales_dept': is_sales_dept,
            'is_ceo': is_ceo,
            'is_target': target_match,
            'target_match_score': 1.0 if target_match else 0.0,
            'seniority_score': seniority_score,
            'sales_bonus': sales_bonus,
            'ceo_bonus': ceo_bonus,
            'target_bonus': target_bonus,
            'persona_score': round(min(final_score, 1.0), 3),
            'raw_persona': persona
        }
    
    def rank_personas_for_company(
        self,
        company_key: str,
        company_data: Dict[str, Any],
        personas_folder: str,
        max_personas: int = 5
    ) -> Dict[str, Any]:
        """Rank personas for a single company with sales-first logic and CEO fallback."""
        personas = self.load_personas_for_company(company_key, personas_folder)
        
        if not personas:
            logger.warning(f"No personas found for {company_key}")
            # Return unknown persona when no file exists
            return {
                'company_key': company_key,
                'company_name': company_data.get('basic_info', {}).get('name', company_key),
                'top_personas': [],
                'all_personas': [],
                'selected_persona': {
                    'name': 'Unknown',
                    'job_title': 'Unknown',
                    'department': 'Unknown',
                    'email': '',
                    'linkedin': '',
                    'persona_score': 0,
                    'is_sales_dept': False,
                    'is_ceo': False,
                    'is_target': False,
                    'selection_reason': 'No personas file found'
                },
                'selection_reason': 'No personas file found'
            }
        
        logger.info(f"Ranking {len(personas)} personas for {company_key}")
        
        # Score all personas
        scored = []
        for persona in personas:
            try:
                scored_persona = self.score_persona(persona)
                scored.append(scored_persona)
            except Exception as e:
                logger.error(f"Failed to score persona: {e}")
                continue
        
        if not scored:
            logger.warning(f"No valid personas scored for {company_key}")
            return {
                'company_key': company_key,
                'company_name': company_data.get('basic_info', {}).get('name', company_key),
                'top_personas': [],
                'all_personas': [],
                'selected_persona': {
                    'name': 'Unknown',
                    'job_title': 'Unknown',
                    'department': 'Unknown',
                    'email': '',
                    'linkedin': '',
                    'persona_score': 0,
                    'is_sales_dept': False,
                    'is_ceo': False,
                    'is_target': False,
                    'selection_reason': 'No valid personas'
                },
                'selection_reason': 'No valid personas'
            }
        
        # Sort by score
        scored.sort(key=lambda x: x['persona_score'], reverse=True)
        
        # Get top personas
        top_personas = scored[:max_personas]
        
        # Select best persona with priority: Sales > Target Role > CEO > Best Score
        selected_persona = None
        selection_reason = ""
        
        # Priority 1: Sales department with target role match
        for p in scored:
            if p['is_sales_dept'] and p['is_target']:
                selected_persona = p
                selection_reason = "Sales + Target Role Match"
                break
        
        # Priority 2: Sales department (any)
        if not selected_persona:
            for p in scored:
                if p['is_sales_dept']:
                    selected_persona = p
                    selection_reason = "Sales Department"
                    break
        
        # Priority 3: Target role match (any department)
        if not selected_persona:
            for p in scored:
                if p['is_target']:
                    selected_persona = p
                    selection_reason = "Target Role Match"
                    break
        
        # Priority 4: CEO/Founder as fallback
        if not selected_persona:
            for p in scored:
                if p['is_ceo']:
                    selected_persona = p
                    selection_reason = "CEO/Founder (Fallback)"
                    break
        
        # Fallback: Best scored persona
        if not selected_persona and scored:
            selected_persona = scored[0]
            selection_reason = "Best Overall Score"
        
        if selected_persona:
            logger.info(f"Selected persona for {company_key}: {selected_persona['name']} ({selection_reason})")
        
        return {
            'company_key': company_key,
            'company_name': company_data.get('basic_info', {}).get('name', company_key),
            'top_personas': top_personas,
            'all_personas': scored,
            'target_personas': [p for p in top_personas if p['is_target']],
            'selected_persona': selected_persona,
            'selection_reason': selection_reason
        }
    
    def rank_personas_for_all_companies(
        self,
        companies: List[Dict[str, Any]],
        personas_folder: str,
        output_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Rank personas for multiple companies."""
        results = []
        
        for company in companies:
            company_key = company.get('company_key', '')
            company_data = company.get('company_data', {})
            
            if not company_key:
                continue
            
            try:
                ranking = self.rank_personas_for_company(
                    company_key,
                    company_data,
                    personas_folder
                )
                results.append(ranking)
                logger.info(f"Ranked personas for {ranking['company_name']}")
            except Exception as e:
                logger.error(f"Failed to rank personas for {company_key}: {e}")
                continue
        
        if output_path:
            self.save_persona_rankings(results, output_path)
        
        return results
    
    def save_persona_rankings(self, rankings: List[Dict[str, Any]], output_path: str):
        """Save persona rankings to file."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean output
        clean_rankings = []
        for ranking in rankings:
            clean_rankings.append({
                'company_key': ranking['company_key'],
                'company_name': ranking['company_name'],
                'top_personas': [
                    {
                        'name': p['name'],
                        'job_title': p['job_title'],
                        'persona_score': p['persona_score'],
                        'is_target': p['is_target']
                    }
                    for p in ranking['top_personas']
                ],
                'total_personas': len(ranking['all_personas'])
            })
        
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(clean_rankings, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved persona rankings to: {output}")
    
    def apply(
        self,
        companies: List[Dict[str, Any]],
        personas_folder: str,
        output_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Full pipeline: rank personas for all companies."""
        return self.rank_personas_for_all_companies(companies, personas_folder, output_path)

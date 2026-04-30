"""
Company Matcher Module - Matches companies with ICP attributes using LLM
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from groq import Groq

logger = logging.getLogger(__name__)


class CompanyMatcher:
    """
    Matches companies from merged_profiles with ICP industry attributes
    """
    
    def __init__(self, groq_client: Groq, model: str = "llama-3.3-70b-versatile", temperature: float = 0.1):
        """
        Initialize the company matcher
        
        Args:
            groq_client: Configured Groq client
            model: Groq model to use
            temperature: Temperature for LLM responses
        """
        self.client = groq_client
        self.model = model
        self.temperature = temperature
        self.merged_profiles_path = Path('../inject_collect_project/merged_profiles')
        
        logger.info(f"Company Matcher initialized with model: {model}")
    
    def load_company_profiles(self) -> Dict[str, Dict]:
        """
        Load all company profiles from merged_profiles folder
        
        Returns:
            Dictionary of company profiles
        """
        if not self.merged_profiles_path.exists():
            logger.warning(f"Merged profiles folder not found: {self.merged_profiles_path}")
            return {}
        
        profiles = {}
        json_files = list(self.merged_profiles_path.glob('*_MERGED.json'))
        logger.info(f"Found {len(json_files)} company profiles")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                    company_key = file_path.stem.replace('_MERGED', '')
                    profiles[company_key] = profile
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return profiles
    
    def match_company_industry(self, company_industry: str, target_industries: List[str]) -> Tuple[bool, str, str]:
        """
        Use LLM to determine if company industry matches any target industries
        
        Args:
            company_industry: The industry from company profile
            target_industries: List of target industries from ICP
            
        Returns:
            Tuple of (is_match, matched_industry, explanation)
        """
        if not target_industries:
            return False, None, "No target industries specified"
        
        target_list = ', '.join(target_industries)
        
        prompt = f"""You are an industry classification expert. Your task is to determine if a company's industry matches any of the target industries.

Company Industry: "{company_industry}"
Target Industries: [{target_list}]

Analyze if the company industry matches or is closely related to any of the target industries. Consider:
- Core business activities
- Industry classifications
- Business models
- Market sectors

Respond with a JSON object:
{{
    "is_match": true/false,
    "matched_industry": "which target industry it matches (or null if no match)",
    "confidence_score": 0.0-1.0,
    "explanation": "Brief explanation of why it matches or doesn't match"
}}

Be strict but reasonable - matching industries should have significant overlap in business activities."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an industry classification expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            return (
                result.get('is_match', False),
                result.get('matched_industry'),
                result.get('explanation', 'No explanation provided')
            )
            
        except Exception as e:
            logger.error(f"Error in LLM industry matching: {e}")
            return False, None, f"Error: {str(e)}"
    
    def find_matching_companies(self, target_industries: List[str]) -> Dict[str, Dict]:
        """
        Find companies that match the target industries
        
        Args:
            target_industries: List of target industries from ICP
            
        Returns:
            Dictionary of matching companies with match details
        """
        if not target_industries:
            logger.warning("No target industries specified, cannot filter companies")
            return {}
        
        logger.info(f"Starting company matching for industries: {target_industries}")
        
        # Load all company profiles
        profiles = self.load_company_profiles()
        if not profiles:
            logger.warning("No company profiles loaded")
            return {}
        
        matching_companies = {}
        total_companies = len(profiles)
        
        logger.info(f"Analyzing {total_companies} companies for industry match")
        
        for company_key, profile in profiles.items():
            company_industry = profile.get('industry', '')
            company_name = profile.get('name', company_key)
            
            if not company_industry:
                logger.debug(f"No industry field for {company_key}")
                continue
            
            # Use LLM to check industry match
            is_match, matched_industry, explanation = self.match_company_industry(
                company_industry, target_industries
            )
            
            if is_match:
                profile['industry_match'] = {
                    'matched_icp_industry': matched_industry,
                    'company_industry': company_industry,
                    'explanation': explanation
                }
                matching_companies[company_key] = profile
                logger.info(f"[MATCH] {company_name} -> {matched_industry}")
            else:
                logger.debug(f"[NO MATCH] {company_name} - {company_industry}")
        
        logger.info(f"Found {len(matching_companies)} matching companies out of {total_companies}")
        return matching_companies
    
    def save_matches(self, companies: Dict[str, Dict], output_name: str = "matches"):
        """
        Save matched companies to output folder
        
        Args:
            companies: Dictionary of matching companies
            output_name: Name identifier for the output
        """
        if not companies:
            logger.warning("No companies to save")
            return
        
        # Create output folder
        output_folder = Path(f'matched_companies_{output_name}')
        output_folder.mkdir(exist_ok=True)
        
        # Save individual company files
        for company_key, profile in companies.items():
            output_file = output_folder / f"{company_key}_MATCHED.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(companies)} matched companies to {output_folder}")
        print(f"\n[RESULTS] Saved {len(companies)} matched companies to folder: {output_folder}")

#!/usr/bin/env python3
"""
Industry Detective - Filter companies based on industry matching using LLM
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detective.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IndustryDetective:
    def __init__(self):
        """Initialize the Industry Detective with Groq client"""
        load_dotenv()
        
        # Validate environment variables
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.groq_model = os.getenv('GROQ_MODEL', 'llama3-70b-8192')
        self.groq_temperature = float(os.getenv('GROQ_TEMPERATURE', '0.1'))
        
        # Initialize Groq client
        self.client = Groq(api_key=self.groq_api_key)
        
        # Set up paths
        self.merged_profiles_path = Path('../inject_collect_project/merged_profiles')
        self.filtered_output_path = Path('filtered_companies')
        self.filtered_output_path.mkdir(exist_ok=True)
        
        logger.info(f"Industry Detective initialized with model: {self.groq_model}")
    
    def load_company_profiles(self) -> Dict[str, Dict]:
        """Load all company profiles from merged_profiles folder"""
        profiles = {}
        
        if not self.merged_profiles_path.exists():
            raise FileNotFoundError(f"Merged profiles folder not found: {self.merged_profiles_path}")
        
        json_files = list(self.merged_profiles_path.glob('*_MERGED.json'))
        logger.info(f"Found {len(json_files)} company profiles")
        
        for file_path in tqdm(json_files, desc="Loading company profiles"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                    company_key = file_path.stem.replace('_MERGED', '')
                    profiles[company_key] = profile
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return profiles
    
    def check_industry_match(self, company_industry: str, target_industry: str) -> Tuple[bool, str]:
        """
        Use LLM to determine if company industry matches target industry
        
        Args:
            company_industry: The industry field from company profile
            target_industry: The target industry to match against
            
        Returns:
            Tuple of (is_match, explanation)
        """
        prompt = f"""You are an industry classification expert. Your task is to determine if two industry descriptions match.

Company Industry: "{company_industry}"
Target Industry: "{target_industry}"

Analyze these industries and determine if they represent the same or closely related business sectors. Consider:
- Core business activities
- Industry classifications
- Business models
- Market sectors

Respond with a JSON object:
{{
    "is_match": true/false,
    "confidence_score": 0.0-1.0,
    "explanation": "Brief explanation of why they match or don't match"
}}

Be strict but reasonable - matching industries should have significant overlap in business activities."""

        try:
            response = self.client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": "You are an industry classification expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.groq_temperature,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            result = json.loads(result_text)
            
            return result.get('is_match', False), result.get('explanation', 'No explanation provided')
            
        except Exception as e:
            logger.error(f"Error in LLM industry matching: {e}")
            return False, f"Error: {str(e)}"
    
    def filter_companies_by_industry(self, target_industry: str) -> Dict[str, Dict]:
        """
        Filter companies based on target industry using LLM matching
        
        Args:
            target_industry: The target industry to filter for
            
        Returns:
            Dictionary of matching companies
        """
        logger.info(f"Starting industry filtering for: {target_industry}")
        
        # Load all company profiles
        profiles = self.load_company_profiles()
        
        matching_companies = {}
        total_companies = len(profiles)
        
        logger.info(f"Analyzing {total_companies} companies for industry match")
        
        with tqdm(total=total_companies, desc="Matching industries") as pbar:
            for company_key, profile in profiles.items():
                company_industry = profile.get('industry', '')
                
                if not company_industry:
                    logger.warning(f"No industry field for {company_key}")
                    pbar.update(1)
                    continue
                
                # Use LLM to check industry match
                is_match, explanation = self.check_industry_match(company_industry, target_industry)
                
                if is_match:
                    profile['industry_match_explanation'] = explanation
                    matching_companies[company_key] = profile
                    logger.info(f"✓ Match: {company_key} - {company_industry}")
                else:
                    logger.debug(f"✗ No match: {company_key} - {company_industry}")
                
                pbar.update(1)
        
        logger.info(f"Found {len(matching_companies)} matching companies out of {total_companies}")
        return matching_companies
    
    def save_filtered_companies(self, companies: Dict[str, Dict], target_industry: str):
        """Save filtered companies to output directory"""
        # Create industry-specific folder
        industry_folder = self.filtered_output_path / target_industry.lower().replace(' ', '_').replace('/', '_')
        industry_folder.mkdir(exist_ok=True)
        
        # Save individual company files
        for company_key, profile in companies.items():
            output_file = industry_folder / f"{company_key}_FILTERED.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
        
        # Save summary file
        summary_file = industry_folder / f"summary_{target_industry.lower().replace(' ', '_')}.json"
        summary = {
            'target_industry': target_industry,
            'total_matches': len(companies),
            'companies': list(companies.keys()),
            'filtered_at': str(Path.cwd())
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(companies)} filtered companies to {industry_folder}")
    
    def run_detection(self, target_industry: str):
        """Main method to run the industry detection process"""
        try:
            logger.info(f"Starting industry detection for: {target_industry}")
            
            # Filter companies
            matching_companies = self.filter_companies_by_industry(target_industry)
            
            # Save results
            self.save_filtered_companies(matching_companies, target_industry)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"INDUSTRY DETECTION COMPLETE")
            print(f"{'='*60}")
            print(f"Target Industry: {target_industry}")
            print(f"Companies Analyzed: {len(self.load_company_profiles())}")
            print(f"Matches Found: {len(matching_companies)}")
            print(f"Success Rate: {len(matching_companies)/len(self.load_company_profiles())*100:.1f}%")
            print(f"Results saved to: {self.filtered_output_path}")
            print(f"{'='*60}")
            
        except Exception as e:
            logger.error(f"Error in detection process: {e}")
            raise

def main():
    """Main entry point"""
    try:
        detective = IndustryDetective()
        
        # Get target industry from user or use default
        target_industry = input("Enter target industry to filter for (e.g., 'Technology', 'Retail', 'Healthcare'): ").strip()
        
        if not target_industry:
            print("No target industry provided. Using 'Technology' as default.")
            target_industry = "Technology"
        
        detective.run_detection(target_industry)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

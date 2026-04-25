"""
Company Filter - Filter companies based on ICP constraints
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class CompanyFilter:
    """Filter companies based on ICP criteria."""
    
    def __init__(self, icp_attributes: Dict[str, Any]):
        """Initialize with ICP attributes."""
        self.icp = icp_attributes
        self.target_countries = [c.lower() for c in icp_attributes.get('target_countries', [])]
        self.target_industries = [i.lower() for i in icp_attributes.get('industry', [])]
        size_range = icp_attributes.get('company_size', {})
        self.min_size = size_range.get('min', 0) or 0
        self.max_size = size_range.get('max', 1000000) or 1000000
        
        logger.info(f"CompanyFilter initialized: size={self.min_size}-{self.max_size}, "
                   f"countries={self.target_countries}, industries={self.target_industries}")
    
    def load_companies_from_folder(self, folder_path: str) -> Dict[str, Any]:
        """Load all company profiles from folder."""
        companies = {}
        folder = Path(folder_path)
        
        if not folder.exists():
            logger.error(f"Companies folder not found: {folder}")
            return companies
        
        for file_path in folder.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    companies[file_path.stem] = data
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(companies)} companies from {folder}")
        return companies
    
    def filter_companies(self, companies: Dict[str, Any]) -> Dict[str, Any]:
        """Filter companies based on ICP criteria."""
        filtered = {}
        
        for company_key, company in companies.items():
            # Handle both nested and flat structures
            basic_info = company.get('basic_info', {})
            employees = basic_info.get('employees') or company.get('employees', 0)
            country = basic_info.get('country', '') or company.get('country', '')
            country_lower = country.lower()
            
            classification = company.get('classification', {})
            industries = classification.get('industries', []) or company.get('industries', [])
            industries_lower = [i.lower() for i in industries]
            
            # Size filter
            if employees:
                employees_val = float(employees) if isinstance(employees, (int, float, str)) else 0
                if not (self.min_size <= employees_val <= self.max_size):
                    logger.debug(f"Filtered {company_key}: employees={employees_val} not in range {self.min_size}-{self.max_size}")
                    continue
            
            # Country filter
            if self.target_countries and country_lower:
                if not any(tc in country_lower for tc in self.target_countries):
                    logger.debug(f"Filtered {company_key}: country='{country}' not in target list")
                    continue
            
            # Industry filter
            if self.target_industries and industries_lower:
                if not any(ti in ind for ti in self.target_industries for ind in industries_lower):
                    logger.debug(f"Filtered {company_key}: industries={industries} don't match target")
                    continue
            
            logger.debug(f"PASSED: {company_key} - employees={employees}, country={country_lower}, industries={industries_lower[:2]}")
            filtered[company_key] = company
        
        logger.info(f"Filtered {len(filtered)}/{len(companies)} companies passed all criteria")
        return filtered
    
    def apply(self, companies_folder: str) -> Dict[str, Any]:
        """Load and filter companies from folder."""
        companies = self.load_companies_from_folder(companies_folder)
        return self.filter_companies(companies)

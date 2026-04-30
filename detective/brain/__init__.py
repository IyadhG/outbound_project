"""
Brain Module - ICP Extraction and Processing Agents

This module contains the agentic components for extracting and processing
Ideal Customer Profile (ICP) attributes from raw text using LLM.
"""

from .schema import ICPAttributes, Range, CityContext, DynamicAttributes
from .icp_agent import ICPExtractionAgent
from .company_matcher import CompanyMatcher
from .geo_agent import GeoAgent

__version__ = "1.0.0"
__all__ = [
    "ICPAttributes",
    "ICPExtractionAgent",
    "CompanyMatcher",
    "GeoAgent"
]

"""
ICP Attributes Schema - Pydantic models for Ideal Customer Profile extraction
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Range(BaseModel):
    """Represents a numeric range with optional min and max values"""
    min: Optional[float] = None
    max: Optional[float] = None


class CityContext(BaseModel):
    """Represents a city with its geographic context"""
    city: str
    country: str
    continent: Optional[str] = None


class DynamicAttributes(BaseModel):
    """Dynamic attributes that can vary based on ICP requirements"""
    tech_stack: List[str] = Field(default_factory=list)
    growth_stage: Optional[str] = None
    funding_stage: Optional[str] = None


class ICPAttributes(BaseModel):
    """
    Complete Ideal Customer Profile attributes schema
    
    This model defines the structure for extracting and storing
    ICP attributes from raw text descriptions.
    """
    industry: List[str] = Field(default_factory=list)

    company_size: Range = Field(default_factory=Range)
    revenue_range: Range = Field(default_factory=Range)

    target_continents: List[str] = Field(default_factory=list)
    target_countries: List[str] = Field(default_factory=list)
    target_cities: List[CityContext] = Field(default_factory=list)

    target_roles: List[str] = Field(default_factory=list)

    must_have_traits: List[str] = Field(default_factory=list)
    nice_to_have_traits: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)

    dynamic_attributes: DynamicAttributes = Field(
        default_factory=DynamicAttributes
    )

    class Config:
        """Pydantic configuration"""
        json_encoders = {
            # Custom encoders if needed
        }
        
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return self.model_dump_json(indent=2)

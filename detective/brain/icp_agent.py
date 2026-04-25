"""
ICP Extraction Agent - Uses LLM to extract Ideal Customer Profile attributes from raw text
"""

import json
import logging
from typing import Dict, Any, Optional
from groq import Groq
from .schema import ICPAttributes, Range, CityContext, DynamicAttributes

logger = logging.getLogger(__name__)


class ICPExtractionAgent:
    """
    Agentic ICP extractor that uses LLM to parse raw text and extract
    structured Ideal Customer Profile attributes
    """
    
    def __init__(self, groq_client: Groq, model: str = "llama-3.3-70b-versatile", temperature: float = 0.1):
        """
        Initialize the ICP extraction agent
        
        Args:
            groq_client: Configured Groq client
            model: Groq model to use for extraction
            temperature: Temperature for LLM responses
        """
        self.client = groq_client
        self.model = model
        self.temperature = temperature
        
        logger.info(f"ICP Extraction Agent initialized with model: {model}")
    
    def extract_icp_attributes(self, raw_text: str) -> ICPAttributes:
        """
        Extract ICP attributes from raw text using LLM
        
        Args:
            raw_text: Raw text description of the ideal customer profile
            
        Returns:
            ICPAttributes object with extracted information
        """
        logger.info("Starting ICP attribute extraction from raw text")
        
        # Create the extraction prompt
        prompt = self._create_extraction_prompt(raw_text)
        
        try:
            # Get LLM response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert at analyzing business requirements and extracting structured Ideal Customer Profile (ICP) attributes. Always respond with valid JSON that matches the exact schema provided."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM response: {result_text}")
            
            # Parse and validate the response
            extracted_data = self._parse_llm_response(result_text)
            
            # Create ICPAttributes object
            icp_attributes = self._create_icp_object(extracted_data)
            
            logger.info(f"Successfully extracted ICP attributes: {len(icp_attributes.industry)} industries, "
                       f"{len(icp_attributes.target_countries)} countries, "
                       f"{len(icp_attributes.target_roles)} roles")
            
            return icp_attributes
            
        except Exception as e:
            logger.error(f"Error in ICP extraction: {e}")
            # Return empty ICP attributes on error
            return ICPAttributes()
    
    def _create_extraction_prompt(self, raw_text: str) -> str:
        """
        Create the extraction prompt for the LLM
        
        Args:
            raw_text: Raw ICP description text
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
Extract Ideal Customer Profile (ICP) attributes from the following text. 

TEXT TO ANALYZE:
\"\"\"
{raw_text}
\"\"\"

Please extract the following attributes and format as JSON:

{{
    "industry": ["list of target industries"],
    "company_size": {{
        "min": minimum_employees_or_null,
        "max": maximum_employees_or_null
    }},
    "revenue_range": {{
        "min": minimum_revenue_or_null,
        "max": maximum_revenue_or_null
    }},
    "target_continents": ["list of target continents"],
    "target_countries": ["list of target countries"],
    "target_cities": [
        {{
            "city": "city_name",
            "country": "country_name",
            "continent": "continent_name_or_null"
        }}
    ],
    "target_roles": ["list of target job roles"],
    "must_have_traits": ["list of must-have characteristics"],
    "nice_to_have_traits": ["list of nice-to-have characteristics"],
    "exclude": ["list of things to exclude"],
    "dynamic_attributes": {{
        "tech_stack": ["list of required technologies"],
        "growth_stage": "growth_stage_or_null",
        "funding_stage": "funding_stage_or_null"
    }}
}}

GUIDELINES:
- Extract only information explicitly mentioned or strongly implied in the text
- Use null for any fields that are not mentioned
- For ranges, extract numeric values (revenue in USD, employees as numbers)
- Be precise and accurate - don't make up information
- Industries should be specific (e.g., "SaaS" instead of just "technology")
- Countries and cities should be real geographical locations
- Roles should be job titles or departments
- Traits should be business characteristics or requirements

Respond with valid JSON only, no additional text.
"""
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate LLM response
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            Parsed dictionary data
        """
        try:
            # Try to parse as JSON directly
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]  # Remove ```json and ```
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]  # Remove ``` and ```
            
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response text: {response_text}")
            raise ValueError("Invalid JSON response from LLM")
    
    def _create_icp_object(self, data: Dict[str, Any]) -> ICPAttributes:
        """
        Create ICPAttributes object from extracted data
        
        Args:
            data: Dictionary of extracted data
            
        Returns:
            Validated ICPAttributes object
        """
        try:
            # Extract and validate each component
            industry = data.get('industry', [])
            
            # Handle company size range
            company_size_data = data.get('company_size', {})
            company_size = Range(
                min=company_size_data.get('min'),
                max=company_size_data.get('max')
            )
            
            # Handle revenue range
            revenue_data = data.get('revenue_range', {})
            revenue_range = Range(
                min=revenue_data.get('min'),
                max=revenue_data.get('max')
            )
            
            # Handle geographic targets
            target_continents = data.get('target_continents', [])
            target_countries = data.get('target_countries', [])
            
            # Handle cities with context
            cities_data = data.get('target_cities', [])
            target_cities = []
            for city_data in cities_data:
                if isinstance(city_data, dict):
                    target_cities.append(CityContext(
                        city=city_data.get('city', ''),
                        country=city_data.get('country', ''),
                        continent=city_data.get('continent')
                    ))
            
            # Handle roles and traits
            target_roles = data.get('target_roles', [])
            must_have_traits = data.get('must_have_traits', [])
            nice_to_have_traits = data.get('nice_to_have_traits', [])
            exclude = data.get('exclude', [])
            
            # Handle dynamic attributes
            dynamic_data = data.get('dynamic_attributes', {})
            dynamic_attributes = DynamicAttributes(
                tech_stack=dynamic_data.get('tech_stack', []),
                growth_stage=dynamic_data.get('growth_stage'),
                funding_stage=dynamic_data.get('funding_stage')
            )
            
            # Create and return ICPAttributes object
            return ICPAttributes(
                industry=industry,
                company_size=company_size,
                revenue_range=revenue_range,
                target_continents=target_continents,
                target_countries=target_countries,
                target_cities=target_cities,
                target_roles=target_roles,
                must_have_traits=must_have_traits,
                nice_to_have_traits=nice_to_have_traits,
                exclude=exclude,
                dynamic_attributes=dynamic_attributes
            )
            
        except Exception as e:
            logger.error(f"Error creating ICP object: {e}")
            logger.error(f"Data: {data}")
            # Return empty ICP on validation error
            return ICPAttributes()
    
    def validate_extraction(self, icp_attributes: ICPAttributes) -> Dict[str, Any]:
        """
        Validate extracted ICP attributes and provide feedback
        
        Args:
            icp_attributes: Extracted ICP attributes
            
        Returns:
            Validation results with suggestions
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "suggestions": []
        }
        
        # Check for empty critical fields
        if not icp_attributes.industry:
            validation_result["warnings"].append("No industries specified")
            validation_result["suggestions"].append("Consider adding target industries for better filtering")
        
        if not icp_attributes.target_countries and not icp_attributes.target_cities:
            validation_result["warnings"].append("No geographic targets specified")
            validation_result["suggestions"].append("Add target countries or cities for geographic filtering")
        
        if not icp_attributes.target_roles:
            validation_result["warnings"].append("No target roles specified")
            validation_result["suggestions"].append("Specify target job roles for better matching")
        
        # Check for reasonable ranges
        if icp_attributes.company_size.min and icp_attributes.company_size.max:
            if icp_attributes.company_size.min > icp_attributes.company_size.max:
                validation_result["is_valid"] = False
                validation_result["warnings"].append("Company size range is invalid (min > max)")
        
        if icp_attributes.revenue_range.min and icp_attributes.revenue_range.max:
            if icp_attributes.revenue_range.min > icp_attributes.revenue_range.max:
                validation_result["is_valid"] = False
                validation_result["warnings"].append("Revenue range is invalid (min > max)")
        
        return validation_result

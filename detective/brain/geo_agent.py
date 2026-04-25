"""
Geo Agent - Filters companies by city proximity using OpenRouteService API
"""

import os
import logging
import requests
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GeoAgent:
    """Filters companies by geographic proximity to target cities"""
    
    ORS_BASE_URL = "https://api.openrouteservice.org/v2/directions/driving-car"
    ORS_GEOCODE_URL = "https://api.openrouteservice.org/geocode/search"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Geo Agent with ORS API key"""
        self.api_key = api_key or os.getenv('ORS_API_KEY')
        if not self.api_key:
            logger.warning("ORS_API_KEY not found - geo filtering disabled")
        else:
            logger.info("Geo Agent initialized with ORS API")
    
    def is_enabled(self) -> bool:
        """Check if geo agent has API key and is enabled"""
        return bool(self.api_key)
    
    def parse_icp_location(self, icp_text: str, icp_attributes: Dict) -> Dict:
        """
        Parse ICP to extract city and range threshold
        
        Returns:
            Dict with 'city', 'country', 'range_km', 'enabled'
        """
        result = {
            'city': None,
            'country': None,
            'range_km': None,
            'enabled': False
        }
        
        # Check if ICP has target cities
        cities = icp_attributes.get('target_cities', []) or icp_attributes.get('city', [])
        countries = icp_attributes.get('target_countries', []) or icp_attributes.get('country', [])
        
        if not cities:
            logger.info("No specific cities mentioned in ICP - geo filtering disabled")
            return result
        
        # Get the primary city
        result['city'] = cities[0] if cities else None
        result['country'] = countries[0] if countries else None
        
        # Parse range from ICP text
        # Look for patterns like "within 150km", "150 km radius", "range of 150 km"
        import re
        range_patterns = [
            r'within\s+(\d+)\s*(?:km|kilometers?|kms?)',
            r'(\d+)\s*(?:km|kilometers?|kms?)\s+(?:radius|range|distance)',
            r'range\s+(?:of\s+)?(\d+)\s*(?:km|kilometers?|kms?)',
            r'radius\s+(?:of\s+)?(\d+)\s*(?:km|kilometers?|kms?)',
        ]
        
        for pattern in range_patterns:
            match = re.search(pattern, icp_text, re.IGNORECASE)
            if match:
                result['range_km'] = int(match.group(1))
                break
        
        # Default range if city mentioned but no range specified
        if result['city'] and not result['range_km']:
            result['range_km'] = 100  # Default 100km
            logger.info(f"No range specified, using default: {result['range_km']}km")
        
        result['enabled'] = bool(result['city'] and result['range_km'])
        
        if result['enabled']:
            logger.info(f"Geo filtering enabled: {result['city']} within {result['range_km']}km")
        
        return result
    
    def geocode_city(self, city: str, country: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """
        Get coordinates for a city using ORS geocoding
        
        Returns:
            Tuple of (lat, lon) or None if not found
        """
        if not self.api_key:
            return None
        
        query = f"{city}, {country}" if country else city
        
        try:
            headers = {
                'Authorization': self.api_key,
                'Accept': 'application/json'
            }
            params = {
                'text': query,
                'size': 1  # Get only the best match
            }
            
            response = requests.get(
                self.ORS_GEOCODE_URL,
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            features = data.get('features', [])
            
            if features:
                coords = features[0]['geometry']['coordinates']
                lon, lat = coords[0], coords[1]
                logger.info(f"Geocoded '{query}': {lat}, {lon}")
                return (lat, lon)
            else:
                logger.warning(f"No geocoding results for: {query}")
                return None
                
        except Exception as e:
            logger.error(f"Geocoding error for '{query}': {e}")
            return None
    
    def get_distance_km(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> Optional[float]:
        """
        Get driving distance between two points using ORS
        
        Returns:
            Distance in kilometers or None if error
        """
        if not self.api_key:
            return None
        
        try:
            headers = {
                'Authorization': self.api_key,
                'Content-Type': 'application/json'
            }
            
            body = {
                'coordinates': [
                    [origin[1], origin[0]],      # [lon, lat] for origin
                    [destination[1], destination[0]]  # [lon, lat] for destination
                ],
                'units': 'km'
            }
            
            response = requests.post(
                self.ORS_BASE_URL,
                headers=headers,
                json=body,
                timeout=15
            )
            response.raise_for_status()
            
            data = response.json()
            routes = data.get('routes', [])
            
            if routes:
                distance = routes[0]['summary']['distance'] / 1000  # Convert to km
                logger.info(f"Distance: {distance:.1f}km")
                return distance
            else:
                logger.warning("No route found")
                return None
                
        except Exception as e:
            logger.error(f"Distance calculation error: {e}")
            return None
    
    def filter_companies_by_proximity(
        self,
        companies: Dict[str, Dict],
        target_city: str,
        target_country: Optional[str],
        range_km: float
    ) -> Dict[str, Dict]:
        """
        Filter companies by proximity to target city
        
        Args:
            companies: Dictionary of company profiles
            target_city: Target city name
            target_country: Target country (optional)
            range_km: Maximum distance in km
            
        Returns:
            Filtered dictionary of companies within range
        """
        if not self.api_key:
            logger.warning("ORS API key not available - skipping geo filter")
            return companies
        
        logger.info(f"Filtering {len(companies)} companies within {range_km}km of {target_city}")
        
        # Geocode target city
        target_coords = self.geocode_city(target_city, target_country)
        if not target_coords:
            logger.warning(f"Could not geocode target city: {target_city}")
            return companies  # Return all if we can't geocode target
        
        filtered = {}
        
        for company_key, profile in companies.items():
            company_name = profile.get('name', company_key)
            company_city = profile.get('city', '')
            company_country = profile.get('country', '')
            
            # If company has no city info, keep it (conservative approach)
            if not company_city:
                logger.info(f"{company_name}: No city info - keeping")
                filtered[company_key] = profile
                continue
            
            # Geocode company city
            company_coords = self.geocode_city(company_city, company_country)
            if not company_coords:
                logger.info(f"{company_name}: Could not geocode - keeping")
                filtered[company_key] = profile
                continue
            
            # Calculate distance
            distance = self.get_distance_km(target_coords, company_coords)
            
            if distance is None:
                logger.info(f"{company_name}: Distance error - keeping")
                filtered[company_key] = profile
                continue
            
            # Check if within range
            if distance <= range_km:
                logger.info(f"{company_name}: {distance:.1f}km from {target_city} - KEEPING")
                profile['geo_info'] = {
                    'distance_to_target_km': round(distance, 1),
                    'target_city': target_city,
                    'company_city': company_city,
                    'within_range': True
                }
                filtered[company_key] = profile
            else:
                logger.info(f"{company_name}: {distance:.1f}km from {target_city} - DROPPING (outside {range_km}km)")
        
        logger.info(f"Geo filter: {len(companies)} -> {len(filtered)} companies")
        return filtered

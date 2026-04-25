#!/usr/bin/env python3
"""
Test Geo Agent - Demonstrates city-based proximity filtering
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add brain to path
sys.path.insert(0, str(Path(__file__).parent))

from brain import GeoAgent


def test_geo_parsing():
    """Test ICP parsing for city and range"""
    print("=" * 80)
    print("TEST 1: ICP Parsing")
    print("=" * 80)
    
    geo_agent = GeoAgent()
    
    test_cases = [
        {
            "name": "Paris with explicit range",
            "icp": "Looking for IT companies in Paris within 150km radius",
            "attributes": {"target_cities": ["Paris"], "target_countries": ["France"]}
        },
        {
            "name": "Berlin with default range",
            "icp": "Target companies in Berlin, Germany",
            "attributes": {"target_cities": ["Berlin"], "target_countries": ["Germany"]}
        },
        {
            "name": "No city mentioned",
            "icp": "Looking for IT companies in Germany and France",
            "attributes": {"target_countries": ["Germany", "France"]}
        },
        {
            "name": "Versailles example",
            "icp": "Companies near Paris within 150km",
            "attributes": {"target_cities": ["Paris"], "target_countries": ["France"]}
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"ICP: {test['icp'][:50]}...")
        
        config = geo_agent.parse_icp_location(test['icp'], test['attributes'])
        
        if config['enabled']:
            print(f"  City: {config['city']}")
            print(f"  Country: {config['country']}")
            print(f"  Range: {config['range_km']}km")
            print(f"  Status: ENABLED")
        else:
            print(f"  Status: DISABLED (no city or range found)")


def test_geocoding():
    """Test geocoding cities"""
    print("\n" + "=" * 80)
    print("TEST 2: Geocoding")
    print("=" * 80)
    
    geo_agent = GeoAgent()
    
    if not geo_agent.is_enabled():
        print("[SKIP] ORS_API_KEY not found in .env")
        return
    
    cities = [
        ("Paris", "France"),
        ("Versailles", "France"),
        ("Nice", "France"),
        ("Lyon", "France"),
        ("Berlin", "Germany"),
    ]
    
    print("\nGeocoding test cities...")
    coords = {}
    
    for city, country in cities:
        result = geo_agent.geocode_city(city, country)
        if result:
            lat, lon = result
            coords[city] = (lat, lon)
            print(f"  {city}: {lat:.4f}, {lon:.4f}")
        else:
            print(f"  {city}: FAILED")
    
    return coords


def test_distance_calculation(coords):
    """Test distance calculation between cities"""
    print("\n" + "=" * 80)
    print("TEST 3: Distance Calculation")
    print("=" * 80)
    
    geo_agent = GeoAgent()
    
    if not geo_agent.is_enabled():
        print("[SKIP] ORS_API_KEY not found in .env")
        return
    
    if not coords or 'Paris' not in coords:
        print("[SKIP] Paris not geocoded")
        return
    
    paris_coords = coords['Paris']
    
    test_routes = [
        ("Paris", "Versailles"),
        ("Paris", "Nice"),
        ("Paris", "Lyon"),
    ]
    
    if 'Berlin' in coords:
        test_routes.append(("Paris", "Berlin"))
    
    print(f"\nDistances from Paris:")
    for origin_name, dest_name in test_routes:
        if origin_name in coords and dest_name in coords:
            distance = geo_agent.get_distance_km(coords[origin_name], coords[dest_name])
            if distance:
                status = "KEEP" if distance <= 150 else "DROP"
                print(f"  {origin_name} -> {dest_name}: {distance:.1f}km [{status}]")
            else:
                print(f"  {origin_name} -> {dest_name}: ERROR")


def test_mock_company_filtering():
    """Test filtering with mock company data"""
    print("\n" + "=" * 80)
    print("TEST 4: Mock Company Filtering")
    print("=" * 80)
    
    geo_agent = GeoAgent()
    
    if not geo_agent.is_enabled():
        print("[SKIP] ORS_API_KEY not found in .env")
        return
    
    # Mock companies
    mock_companies = {
        "versailles_tech": {
            "name": "Versailles Tech Solutions",
            "city": "Versailles",
            "country": "France",
            "industry": "IT Services"
        },
        "nice_software": {
            "name": "Nice Software Group",
            "city": "Nice",
            "country": "France",
            "industry": "Software"
        },
        "lyon_systems": {
            "name": "Lyon Systems",
            "city": "Lyon",
            "country": "France",
            "industry": "IT Consulting"
        },
        "paris_digital": {
            "name": "Paris Digital Agency",
            "city": "Paris",
            "country": "France",
            "industry": "Digital Marketing"
        }
    }
    
    print(f"\nMock ICP: Companies in Paris within 150km")
    print(f"Total companies: {len(mock_companies)}")
    print("\nBefore filtering:")
    for key, company in mock_companies.items():
        print(f"  - {company['name']} ({company['city']})")
    
    # Filter
    filtered = geo_agent.filter_companies_by_proximity(
        mock_companies,
        target_city="Paris",
        target_country="France",
        range_km=150
    )
    
    print(f"\nAfter filtering (150km from Paris):")
    print(f"Kept: {len(filtered)}/{len(mock_companies)} companies")
    for key, company in filtered.items():
        geo_info = company.get('geo_info', {})
        distance = geo_info.get('distance_to_target_km', 'N/A')
        print(f"  - {company['name']} ({company['city']}) - {distance}km")
    
    dropped = set(mock_companies.keys()) - set(filtered.keys())
    if dropped:
        print(f"\nDropped (outside 150km):")
        for key in dropped:
            company = mock_companies[key]
            print(f"  - {company['name']} ({company['city']})")


def main():
    """Run all geo agent tests"""
    print("\n" + "=" * 80)
    print("GEO AGENT TEST SUITE")
    print("=" * 80)
    
    # Check if API key exists
    api_key = os.getenv('ORS_API_KEY')
    if not api_key:
        print("\n[WARNING] ORS_API_KEY not found in .env")
        print("Geo tests will be skipped. Add this to your .env:")
        print("ORS_API_KEY=your_api_key_here")
    else:
        print(f"\n[OK] ORS_API_KEY found ({api_key[:20]}...)")
    
    # Run tests
    test_geo_parsing()
    coords = test_geocoding()
    test_distance_calculation(coords)
    test_mock_company_filtering()
    
    print("\n" + "=" * 80)
    print("GEO AGENT TEST COMPLETED")
    print("=" * 80)
    print("\nTo use in main flow:")
    print("1. Add ORS_API_KEY to .env")
    print("2. Mention a city + range in your ICP")
    print("3. Run: python main.py")


if __name__ == "__main__":
    main()

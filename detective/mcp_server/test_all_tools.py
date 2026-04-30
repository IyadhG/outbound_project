#!/usr/bin/env python3
"""
Comprehensive test for ALL 9 Detective MCP Tools

Tests each tool with real examples and shows the actual output.
Run: python test_all_tools.py
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"[OK] Loaded .env from {env_path}")
else:
    print(f"[WARN] .env file not found at {env_path}")

print(f"   ORS_API_KEY present: {bool(os.getenv('ORS_API_KEY'))}")
print(f"   GEMINI_API_KEY present: {bool(os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'))}")
print()

# Import all tool functions
from mcp_server.mcp_server import (
    extract_icp,
    match_companies,
    filter_by_employees,
    filter_by_country,
    rank_by_similarity,
    calculate_final_scores,
    rank_personas,
    geo_filter,
    analyze_company_profile
)

async def test_extract_icp():
    """Test 1: Extract ICP from natural language text"""
    print("\n" + "="*60)
    print("[1] TEST 1: extract_icp")
    print("="*60)
    
    icp_text = """I want to target IT and SaaS companies with 50-500 employees 
    in United States, Canada, United Kingdom, and Germany. 
    We want to connect with Sales Managers, CTOs, and Heads of Product.
    Companies should use modern tech stack (React, Python, AWS) 
    and be in growth stage with Series B or C funding."""
    
    print(f"\n[IN] Input (ICP text):")
    print(f"   {icp_text[:100]}...")
    
    result = await extract_icp(icp_text)
    data = json.loads(result[0].text)
    
    print(f"\n[OUT] Output:")
    if data.get("success"):
        attrs = data["icp_attributes"]
        print(f"   [OK] Industries: {attrs.get('industry', [])}")
        print(f"   [OK] Company Size: {attrs.get('company_size', {})}")
        print(f"   [OK] Target Countries: {attrs.get('target_countries', [])}")
        print(f"   [OK] Target Roles: {attrs.get('target_roles', [])}")
        print(f"   [OK] Tech Stack: {attrs.get('tech_stack', [])}")
    else:
        print(f"   [ERR] Error: {data.get('error')}")
    
    return data.get("icp_attributes", {}) if data.get("success") else {}


async def test_filter_by_employees():
    """Test 2: Filter companies by employee count"""
    print("\n" + "="*60)
    print("[2] TEST 2: filter_by_employees")
    print("="*60)
    
    sample_companies = {
        "small_tech": {
            "basic_info": {"name": "Small Tech Startup", "employees": 25, "country": "USA"}
        },
        "mid_tech": {
            "basic_info": {"name": "Mid-Size Tech", "employees": 150, "country": "Germany"}
        },
        "large_tech": {
            "basic_info": {"name": "Large Tech Corp", "employees": 5000, "country": "UK"}
        },
        "tiny_startup": {
            "basic_info": {"name": "Tiny Startup", "employees": 10, "country": "USA"}
        },
        "growing_saas": {
            "basic_info": {"name": "Growing SaaS", "employees": 300, "country": "Canada"}
        }
    }
    
    print(f"\n[IN] Input: {len(sample_companies)} companies")
    for key, company in sample_companies.items():
        print(f"   - {company['basic_info']['name']}: {company['basic_info']['employees']} employees")
    
    min_emp, max_emp = 50, 500
    print(f"\n[5] Filter: {min_emp} - {max_emp} employees")
    
    result = await filter_by_employees(sample_companies, min_emp, max_emp)
    data = json.loads(result[0].text)
    
    print(f"\n[OUT] Output:")
    if data.get("success"):
        print(f"   [OK] Kept: {data['filtered_count']} companies")
        print(f"   [OK] Companies: {', '.join(data['kept_companies'])}")
    else:
        print(f"   [ERR] Error: {data.get('error')}")


async def test_filter_by_country():
    """Test 3: Filter companies by country"""
    print("\n" + "="*60)
    print("[3] TEST 3: filter_by_country")
    print("="*60)
    
    sample_companies = {
        "us_company": {
            "basic_info": {"name": "US Tech", "country": "United States", "city": "New York"}
        },
        "de_company": {
            "basic_info": {"name": "DE Tech", "country": "Germany", "city": "Berlin"}
        },
        "fr_company": {
            "basic_info": {"name": "FR Tech", "country": "France", "city": "Paris"}
        },
        "uk_company": {
            "basic_info": {"name": "UK Tech", "country": "United Kingdom", "city": "London"}
        },
        "ca_company": {
            "basic_info": {"name": "CA Tech", "country": "Canada", "city": "Toronto"}
        }
    }
    
    target_countries = ["United States", "Germany", "United Kingdom"]
    
    print(f"\n[IN] Input: {len(sample_companies)} companies")
    for key, company in sample_companies.items():
        print(f"   - {company['basic_info']['name']}: {company['basic_info']['country']}")
    
    print(f"\n[5] Filter: {', '.join(target_countries)}")
    
    result = await filter_by_country(sample_companies, target_countries)
    data = json.loads(result[0].text)
    
    print(f"\n[OUT] Output:")
    if data.get("success"):
        print(f"   [OK] Kept: {data['filtered_count']} companies")
        for company in data.get('companies', []):
            print(f"      - {company['name']} ({company['country']})")
    else:
        print(f"   [ERR] Error: {data.get('error')}")


async def test_calculate_final_scores():
    """Test 4: Calculate final scores with intent boost"""
    print("\n" + "="*60)
    print("[4] TEST 4: calculate_final_scores")
    print("="*60)
    
    ranked_companies = [
        {
            "company_key": "bosch_us",
            "company_name": "Bosch in the USA",
            "similarity_score": 0.75
        },
        {
            "company_key": "alten_de",
            "company_name": "ALTEN Deutschland",
            "similarity_score": 0.68
        },
        {
            "company_key": "nhs_uk",
            "company_name": "NHS Digital",
            "similarity_score": 0.72
        }
    ]
    
    # Simulate intent signals (from agentic_intent)
    intent_signals = {
        "bosch_us": {
            "funding": {"confidence": 0.8, "has_recent_funding": True},
            "news": {"confidence": 0.6, "has_news": True}
        },
        "alten_de": {
            "funding": {"confidence": 0.3, "has_recent_funding": False}
        },
        "nhs_uk": {
            "news": {"confidence": 0.9, "has_news": True}
        }
    }
    
    print(f"\n[IN] Input: {len(ranked_companies)} companies with similarity scores")
    for c in ranked_companies:
        print(f"   - {c['company_name']}: {c['similarity_score']}")
    
    print(f"\n[5] Intent Signals:")
    print(f"   - Bosch: High funding confidence (0.8) + news (0.6)")
    print(f"   - ALTEN: Low funding confidence (0.3)")
    print(f"   - NHS: High news confidence (0.9)")
    
    intent_boost = 0.05
    print(f"\n[CFG] Intent Boost Factor: {intent_boost}")
    
    result = await calculate_final_scores(ranked_companies, intent_signals, intent_boost)
    data = json.loads(result[0].text)
    
    print(f"\n[OUT] Output:")
    if data.get("success"):
        print(f"   [OK] Scored {data['total_scored']} companies\n")
        for c in data.get('ranked_companies', []):
            print(f"   Rank {c['rank']}: {c['company_name']}")
            print(f"      Similarity: {c['similarity_score']}")
            print(f"      Intent Boost: +{c['intent_boost']}")
            print(f"      Final Score: {c['final_score']}")
            print()
    else:
        print(f"   [ERR] Error: {data.get('error')}")


async def test_analyze_company_profile():
    """Test 5: Analyze company profile with LLM"""
    print("\n" + "="*60)
    print("[5] TEST 5: analyze_company_profile")
    print("="*60)
    
    # Sample company profile (Bosch-like)
    company_profile = {
        "basic_info": {
            "name": "TechCorp Solutions",
            "description": "Leading provider of cloud-based enterprise software solutions. Specializes in SaaS products for mid-market companies. Over 10 years in business with 500+ employees worldwide.",
            "website": "https://techcorp.example.com",
            "country": "United States",
            "city": "San Francisco",
            "employees": 500
        },
        "classification": {
            "industries": ["Software", "SaaS", "Enterprise Technology"],
            "specialties": ["Cloud Computing", "Data Analytics", "AI/ML", "Enterprise Software"]
        },
        "products_services": {
            "products": ["Cloud Platform", "Analytics Suite", "AI Tools"],
            "services": ["Consulting", "Implementation", "Support"]
        }
    }
    
    print(f"\n[IN] Input: {company_profile['basic_info']['name']}")
    print(f"   Description: {company_profile['basic_info']['description'][:80]}...")
    print(f"   Industries: {', '.join(company_profile['classification']['industries'])}")
    
    print(f"\n[5] Analysis Type: summary")
    
    result = await analyze_company_profile(company_profile, "summary")
    data = json.loads(result[0].text)
    
    print(f"\n[OUT] Output:")
    if data.get("success"):
        print(f"   [OK] Analysis:")
        print(f"      {data['analysis'][:200]}...")
    else:
        print(f"   [WARN]  Note: {data.get('error', 'Requires Groq API key')}")
        print(f"   [TIP] The tool works but needs GROQ_API_KEY for LLM analysis")


async def test_match_companies():
    """Test 6: Match companies to ICP industries (requires API key)"""
    print("\n" + "="*60)
    print("[6] TEST 6: match_companies")
    print("="*60)
    
    icp_attributes = {
        "industry": ["IT", "SaaS", "Software"]
    }
    
    companies_folder = "../inject_collect_project/merged_profiles"
    
    print(f"\n[IN] Input:")
    print(f"   Target Industries: {', '.join(icp_attributes['industry'])}")
    print(f"   Companies Folder: {companies_folder}")
    
    print(f"\n[WARN]  This tool requires:")
    print(f"   - GROQ_API_KEY configured in .env")
    print(f"   - Companies in {companies_folder}")
    
    try:
        result = await match_companies(icp_attributes, companies_folder)
        data = json.loads(result[0].text)
        
        print(f"\n[OUT] Output:")
        if data.get("success"):
            print(f"   [OK] Matched: {data['total_matched']} companies")
            print(f"   [OK] Companies: {', '.join(data['matched_companies'][:5])}...")
        else:
            print(f"   [WARN]  {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"\n[OUT] Output:")
        print(f"   [WARN]  Tool needs GROQ_API_KEY: {e}")


async def test_rank_by_similarity():
    """Test 7: Rank companies by similarity (requires API key)"""
    print("\n" + "="*60)
    print("[7] TEST 7: rank_by_similarity")
    print("="*60)
    
    icp_text = "IT and SaaS companies with cloud solutions for enterprise"
    
    sample_companies = {
        "techcorp": {
            "basic_info": {
                "name": "TechCorp Solutions",
                "description": "Enterprise cloud software and SaaS solutions",
                "country": "USA"
            },
            "classification": {
                "industries": ["Software", "SaaS"],
                "specialties": ["Cloud", "Enterprise"]
            }
        },
        "manufacturing_co": {
            "basic_info": {
                "name": "Industrial Manufacturing Co",
                "description": "Heavy machinery and manufacturing equipment",
                "country": "Germany"
            },
            "classification": {
                "industries": ["Manufacturing"],
                "specialties": ["Machinery"]
            }
        }
    }
    
    print(f"\n[IN] Input:")
    print(f"   ICP Text: {icp_text}")
    print(f"   Companies: {len(sample_companies)}")
    
    print(f"\n[WARN]  This tool requires:")
    print(f"   - GOOGLE_API_KEY for Gemini embeddings")
    
    try:
        result = await rank_by_similarity(icp_text, sample_companies)
        data = json.loads(result[0].text)
        
        print(f"\n[OUT] Output:")
        if data.get("success"):
            print(f"   [OK] Ranked: {data['total_ranked']} companies")
            for c in data.get('ranked_companies', []):
                print(f"      - {c['company_name']}: {c['similarity_score']}")
        else:
            print(f"   [WARN]  {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"\n[OUT] Output:")
        print(f"   [WARN]  Tool needs GOOGLE_API_KEY: {e}")


async def test_rank_personas():
    """Test 8: Rank personas (requires API key)"""
    print("\n" + "="*60)
    print("[8] TEST 8: rank_personas")
    print("="*60)
    
    companies = [
        {"company_key": "bosch_us", "company_name": "Bosch in the USA"}
    ]
    target_roles = ["Sales Manager", "CTO", "Head of Product"]
    
    print(f"\n[IN] Input:")
    print(f"   Company: {companies[0]['company_name']}")
    print(f"   Target Roles: {', '.join(target_roles)}")
    
    print(f"\n[WARN]  This tool requires:")
    print(f"   - GROQ_API_KEY for LLM persona scoring")
    print(f"   - Persona files in ../inject_collect_project/personas_discovered")
    
    try:
        result = await rank_personas(companies, target_roles, "../inject_collect_project/personas_discovered")
        data = json.loads(result[0].text)
        
        print(f"\n[OUT] Output:")
        if data.get("success"):
            print(f"   [OK] Ranked personas for {data['total_companies']} companies")
            for r in data.get('persona_selections', []):
                persona = r.get('selected_persona', {})
                print(f"      - {persona.get('full_name', 'N/A')}: {persona.get('job_title', 'N/A')}")
        else:
            print(f"   [WARN]  {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"\n[OUT] Output:")
        print(f"   [WARN]  Tool needs GROQ_API_KEY and persona files: {e}")


async def test_geo_filter():
    """Test 9: Geo filter by city proximity (requires API key)"""
    print("\n" + "="*60)
    print("[9] TEST 9: geo_filter")
    print("="*60)
    
    sample_companies = {
        "berlin_tech": {
            "basic_info": {"name": "Berlin Tech", "city": "Berlin", "country": "Germany"}
        },
        "munich_tech": {
            "basic_info": {"name": "Munich Tech", "city": "Munich", "country": "Germany"}
        },
        "hamburg_tech": {
            "basic_info": {"name": "Hamburg Tech", "city": "Hamburg", "country": "Germany"}
        },
        "london_tech": {
            "basic_info": {"name": "London Tech", "city": "London", "country": "UK"}
        }
    }
    
    target_city = "Berlin"
    range_km = 300
    
    print(f"\n[IN] Input:")
    print(f"   Companies in: Berlin, Munich, Hamburg, London")
    print(f"   Target City: {target_city}")
    print(f"   Range: {range_km} km")
    
    print(f"\n[WARN]  This tool requires:")
    print(f"   - ORS_API_KEY for OpenRouteService geocoding")
    
    try:
        result = await geo_filter(sample_companies, target_city, "Germany", range_km)
        data = json.loads(result[0].text)
        
        print(f"\n[OUT] Output:")
        if data.get("success"):
            print(f"   [OK] Kept: {data['filtered_count']} companies within {range_km}km of {target_city}")
            for key in data.get('kept_companies', []):
                print(f"      - {key}")
        else:
            print(f"   [WARN]  {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"\n[OUT] Output:")
        print(f"   [WARN]  Tool needs ORS_API_KEY: {e}")


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("[D]  DETECTIVE MCP SERVER - COMPREHENSIVE TOOL TESTS")
    print("="*60)
    print("\nTesting all 9 detective tools with real examples...")
    print("Some tools require API keys (Groq, Google, ORS)")
    
    # Run all tests
    await test_extract_icp()
    await test_filter_by_employees()
    await test_filter_by_country()
    await test_calculate_final_scores()
    await test_analyze_company_profile()
    await test_match_companies()
    await test_rank_by_similarity()
    await test_rank_personas()
    await test_geo_filter()
    
    print("\n" + "="*60)
    print("[OK] ALL TESTS COMPLETED!")
    print("="*60)
    print("\n[7] Summary:")
    print("   Tools working without API keys:")
    print("      [OK] filter_by_employees")
    print("      [OK] filter_by_country")
    print("      [OK] calculate_final_scores")
    print("\n   Tools requiring API keys:")
    print("      [KEY] extract_icp (Groq)")
    print("      [KEY] match_companies (Groq)")
    print("      [KEY] rank_by_similarity (Google)")
    print("      [KEY] rank_personas (Groq)")
    print("      [KEY] analyze_company_profile (Groq)")
    print("      [KEY] geo_filter (ORS)")
    print("\n[TIP] To use all tools, configure .env with:")
    print("   GROQ_API_KEY=your_key")
    print("   GOOGLE_API_KEY=your_key")
    print("   ORS_API_KEY=your_key")


if __name__ == "__main__":
    asyncio.run(main())

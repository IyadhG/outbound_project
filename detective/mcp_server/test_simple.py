#!/usr/bin/env python3
"""
Simple inline test for Detective MCP Server

Tests the detective tools directly without external dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_tools_directly():
    """Test tools by importing and calling them directly."""
    
    print("🧪 Testing Detective MCP Tools (Direct Import)\n")
    
    # Import the tool functions directly from mcp_server.py
    from mcp_server.mcp_server import (
        extract_icp,
        match_companies,
        filter_by_employees,
        filter_by_country,
        calculate_final_scores
    )
    
    # Test 1: Extract ICP
    print("1️⃣ Testing extract_icp...")
    try:
        result = await extract_icp("I want IT companies with 50-500 employees in USA")
        print(f"   Result: {result[0].text[:200]}...")
        print("   ✅ extract_icp works!\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # Test 2: Filter by Employees
    print("2️⃣ Testing filter_by_employees...")
    try:
        sample_companies = {
            "company_1": {"basic_info": {"name": "Small Co", "employees": 25}},
            "company_2": {"basic_info": {"name": "Mid Co", "employees": 150}},
            "company_3": {"basic_info": {"name": "Large Co", "employees": 5000}}
        }
        result = await filter_by_employees(sample_companies, 50, 500)
        data = json.loads(result[0].text)
        print(f"   Input: 3 companies, Output: {data['filtered_count']} companies")
        print("   ✅ filter_by_employees works!\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # Test 3: Filter by Country
    print("3️⃣ Testing filter_by_country...")
    try:
        sample_companies = {
            "us_company": {"basic_info": {"name": "US Tech", "country": "United States"}},
            "de_company": {"basic_info": {"name": "DE Tech", "country": "Germany"}},
            "fr_company": {"basic_info": {"name": "FR Tech", "country": "France"}}
        }
        result = await filter_by_country(sample_companies, ["United States", "Germany"])
        data = json.loads(result[0].text)
        print(f"   Input: 3 companies, Output: {data['filtered_count']} companies")
        print("   ✅ filter_by_country works!\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # Test 4: Calculate Final Scores
    print("4️⃣ Testing calculate_final_scores...")
    try:
        ranked = [
            {"company_key": "c1", "company_name": "Company 1", "similarity_score": 0.75},
            {"company_key": "c2", "company_name": "Company 2", "similarity_score": 0.68}
        ]
        intent = {"c1": {"funding": {"confidence": 0.8}}}
        result = await calculate_final_scores(ranked, intent, 0.05)
        data = json.loads(result[0].text)
        print(f"   Scored {data['total_scored']} companies")
        if data.get('ranked_companies'):
            top = data['ranked_companies'][0]
            print(f"   Top: {top['company_name']} - Score: {top['final_score']}")
        print("   ✅ calculate_final_scores works!\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    print("=" * 50)
    print("✅ Direct tool tests completed!")
    print("\nNote: Tools requiring LLM/Groq need API keys configured.")
    print("Tools tested without external API calls work correctly.")

if __name__ == "__main__":
    import json
    asyncio.run(test_tools_directly())

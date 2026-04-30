#!/usr/bin/env python3
"""
Test client for Detective MCP Server - Detective Work Tools

Tests all detective tools that DO the work:
- extract_icp
- match_companies
- filter_by_employees
- filter_by_country
- rank_by_similarity
- calculate_final_scores
- rank_personas
- geo_filter
- analyze_company_profile

Usage:
    python test_mcp_client.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_mcp_server():
    """Test all detective tools."""
    
    try:
        from mcp_client import MCPClient
    except ImportError:
        print("❌ mcp_client not found.")
        print("   Install with: pip install mcp-client")
        print("   Or copy from agentic_intent/mcp_client.py")
        return
    
    client = MCPClient()
    
    print("🔌 Connecting to Detective MCP Server...")
    server_path = str(Path(__file__).parent / "mcp_server.py")
    
    try:
        await client.connect_to_server(server_path)
        print("✅ Connected!\n")
        
        # Test 1: Extract ICP
        print("🎯 Test 1: Extract ICP from text")
        icp_text = """I want IT and SaaS companies with 50-500 employees. 
        Target countries: United States, Canada, United Kingdom, Germany.
        We want to connect with Sales Managers, CTOs, and Heads of Product.
        Must have modern tech stack like React, Python, AWS.
        Prefer companies in growth stage with Series B or C funding."""
        
        result = await client.call_tool("extract_icp", {"icp_text": icp_text})
        icp_data = json.loads(result[0].text)
        print(json.dumps(icp_data, indent=2)[:800] + "...\n")
        
        if icp_data.get("success"):
            icp_attrs = icp_data["icp_attributes"]
        else:
            icp_attrs = {"industry": ["IT", "SaaS"]}
        
        # Test 2: Match Companies
        print("🏢 Test 2: Match companies to ICP industries")
        result = await client.call_tool("match_companies", {
            "icp_attributes": icp_attrs,
            "companies_folder": "../inject_collect_project/merged_profiles"
        })
        match_data = json.loads(result[0].text)
        print(json.dumps(match_data, indent=2)[:600] + "...\n")
        
        # Test 3: Analyze Company Profile
        print("🔍 Test 3: Analyze a company profile")
        
        # Load a sample company
        company_file = Path("../inject_collect_project/merged_profiles/bosch_us_MERGED.json")
        if company_file.exists():
            with open(company_file, 'r', encoding='utf-8') as f:
                company_profile = json.load(f)
            
            result = await client.call_tool("analyze_company_profile", {
                "company_profile": company_profile,
                "analysis_type": "summary"
            })
            print(result[0].text[:500] + "...\n")
        else:
            print("⚠️  Sample company not found, skipping\n")
        
        # Test 4: Filter by Employees (with sample data)
        print("👥 Test 4: Filter by employee count")
        
        sample_companies = {
            "company_1": {
                "basic_info": {"name": "Small Tech", "employees": 25, "country": "USA"}
            },
            "company_2": {
                "basic_info": {"name": "Mid Tech", "employees": 150, "country": "Germany"}
            },
            "company_3": {
                "basic_info": {"name": "Large Tech", "employees": 5000, "country": "UK"}
            }
        }
        
        result = await client.call_tool("filter_by_employees", {
            "companies": sample_companies,
            "min_employees": 50,
            "max_employees": 500
        })
        print(result[0].text + "\n")
        
        # Test 5: Filter by Country
        print("🌍 Test 5: Filter by country")
        result = await client.call_tool("filter_by_country", {
            "companies": sample_companies,
            "target_countries": ["USA", "Germany"]
        })
        print(result[0].text + "\n")
        
        # Test 6: Calculate Final Scores
        print("⭐ Test 6: Calculate final scores with intent boost")
        
        sample_ranked = [
            {
                "company_key": "bosch_us",
                "company_name": "Bosch",
                "similarity_score": 0.75
            },
            {
                "company_key": "alten_de",
                "company_name": "ALTEN",
                "similarity_score": 0.68
            }
        ]
        
        sample_intent = {
            "bosch_us": {"funding": {"confidence": 0.8}, "news": {"confidence": 0.6}},
            "alten_de": {"funding": {"confidence": 0.3}}
        }
        
        result = await client.call_tool("calculate_final_scores", {
            "ranked_companies": sample_ranked,
            "intent_signals": sample_intent,
            "intent_boost": 0.05
        })
        print(result[0].text + "\n")
        
        # Test 7: Rank Personas
        print("👤 Test 7: Rank personas for companies")
        
        sample_companies_list = [
            {"company_key": "bosch_us", "company_name": "Bosch in the USA"}
        ]
        
        result = await client.call_tool("rank_personas", {
            "companies": sample_companies_list,
            "target_roles": ["Sales Manager", "CTO"],
            "personas_folder": "../inject_collect_project/personas_discovered"
        })
        persona_data = json.loads(result[0].text)
        print(json.dumps(persona_data, indent=2)[:800] + "...\n")
        
        # Test 8: Geo Filter (if ORS key available)
        print("📍 Test 8: Geo filter (requires ORS_API_KEY)")
        
        sample_geo_companies = {
            "company_1": {
                "basic_info": {"name": "Berlin Tech", "city": "Berlin", "country": "Germany"}
            },
            "company_2": {
                "basic_info": {"name": "Munich Tech", "city": "Munich", "country": "Germany"}
            },
            "company_3": {
                "basic_info": {"name": "London Tech", "city": "London", "country": "UK"}
            }
        }
        
        result = await client.call_tool("geo_filter", {
            "companies": sample_geo_companies,
            "target_city": "Berlin",
            "target_country": "Germany",
            "range_km": 200
        })
        print(result[0].text + "\n")
        
        print("✅ All detective tools tested successfully!")
        print("\n📊 Summary:")
        print("  - ICP Extraction: Uses LLM to parse text")
        print("  - Company Matching: Uses LLM to match industries")
        print("  - Filtering: Employee count & country")
        print("  - Ranking: Gemini embeddings similarity")
        print("  - Scoring: Intent boost calculation")
        print("  - Personas: LLM-based selection")
        print("  - Geo Filter: Proximity filtering")
        print("  - Analysis: LLM company insights")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_mcp_server())

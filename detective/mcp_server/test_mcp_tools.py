#!/usr/bin/env python3
"""
Test MCP Tools - Similar to main.py but for MCP Server

This tests the 2 high-level orchestrator tools:
1. rank_lead - Analyze single company against ICP
2. detect_top_leads - Full pipeline with dynamic graph
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server import rank_lead, detect_top_leads

def test_rank_lead():
    """Test rank_lead tool with a sample company."""
    print("\n" + "="*80)
    print("🎯 TEST 1: rank_lead")
    print("="*80)
    
    # Sample company data (Bosch US)
    company_data = json.dumps({
        "basic_info": {
            "name": "Bosch in the USA",
            "description": "Leading supplier of technology and services in automotive and industrial sectors",
            "employees": 3200,
            "country": "United States",
            "city": "Farmington Hills"
        },
        "classification": {
            "industries": ["Automotive", "Technology", "Engineering"]
        },
        "intent_signals": [
            {"confidence": 0.8, "relevance": 0.9}
        ],
        "personas": [
            {
                "full_name": "John Smith",
                "job_title": "Sales Manager",
                "is_likely_to_engage": 0.8,
                "intent_strength": 8
            }
        ]
    })
    
    # Sample ICP
    icp_data = json.dumps({
        "industry": ["Automotive", "Technology"],
        "company_size": {"min": 50, "max": 5000},
        "target_countries": ["United States", "Germany"],
        "target_roles": ["Sales Manager", "CTO"]
    })
    
    user_offering = "AI-powered lead generation platform"
    raw_query = "Looking for automotive and tech companies in US with 50-5000 employees"
    
    print("\n📥 Input:")
    print(f"   Company: Bosch in the USA")
    print(f"   ICP: Automotive/Tech, 50-5000 employees, US/Germany")
    
    result = rank_lead(company_data, icp_data, user_offering, raw_query)
    data = json.loads(result)
    
    print("\n📤 Output:")
    if "error" in data:
        print(f"   ❌ Error: {data['error']}")
    else:
        print(f"   ✅ Total Score: {data.get('total_score', 0)}")
        print(f"   ✅ Industry Match: {data.get('industry_match', 0)}")
        print(f"   ✅ Size Match: {data.get('size_match', 0)}")
        print(f"   ✅ Location Match: {data.get('location_match', 0)}")
        print(f"   ✅ Intent Boost: {data.get('intent_boost', 0)}")
        print(f"   ✅ Reason: {data.get('reason', 'N/A')}")
    
    return data


def test_detect_top_leads():
    """Test detect_top_leads full pipeline."""
    print("\n" + "="*80)
    print("🕵️ TEST 2: detect_top_leads (Full Pipeline)")
    print("="*80)
    
    # Example ICP query
    raw_icp_query = """
    I want to target IT and SaaS companies with 50-500 employees 
    in United States, Canada, United Kingdom, Germany.
    Target roles: Sales Manager, CTO, Head of Product.
    They should use modern tech stack like React, Python.
    """
    
    user_offering = "AI-powered lead generation and sales automation platform"
    
    print("\n📥 Input:")
    print(f"   ICP: {raw_icp_query[:60]}...")
    print(f"   User Offering: {user_offering}")
    print(f"   Max Distance: 150km")
    print(f"   Limit: 3 top leads")
    
    # Test with explicit absolute path for debugging
    import os
    absolute_path = r"c:\Users\Dell\Desktop\outbound_project\inject_collect_project\merged_profiles"
    print(f"\n📁 Companies Folder:")
    print(f"   Absolute: {absolute_path}")
    print(f"   Exists: {Path(absolute_path).exists()}")
    
    if Path(absolute_path).exists():
        json_count = len(list(Path(absolute_path).glob("*.json")))
        print(f"   JSON Files: {json_count}")
    
    result = detect_top_leads(raw_icp_query, user_offering, absolute_path, 150, 3)
    data = json.loads(result)
    
    print("\n📤 Output:")
    if data.get("status") == "error":
        print(f"   ❌ Error: {data.get('error', 'Unknown error')}")
        if "traceback" in data:
            print(f"   Traceback: {data['traceback'][:200]}...")
    else:
        summary = data.get("summary", {})
        leads = data.get("leads", [])
        graph = data.get("dynamic_graph", {})
        
        print(f"   ✅ Status: SUCCESS")
        print(f"   ✅ Total Found: {summary.get('total_found', 0)}")
        print(f"   ✅ Top Leads: {len(leads)}")
        
        if leads:
            print("\n   🏆 Top Companies:")
            for i, lead in enumerate(leads, 1):
                print(f"      {i}. {lead['company']} (Score: {lead['total_score']})")
                print(f"         Reason: {lead['reasoning']}")
                if lead.get('top_contacts'):
                    print(f"         Contacts: {len(lead['top_contacts'])}")
                    for c in lead['top_contacts'][:2]:
                        print(f"           - {c['name']} ({c['title']}) - Score: {c['score']}")
        
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        print(f"\n   📊 Dynamic Graph:")
        print(f"      Nodes: {len(nodes)}")
        print(f"      Edges: {len(edges)}")
        
        if nodes:
            print("\n      Graph Nodes:")
            for node in nodes[:5]:
                print(f"        - {node['id']} ({node['type']}): {node.get('label', 'N/A')}")
    
    return data


def test_load_companies():
    """Test company loading by checking folder paths."""
    print("\n" + "="*80)
    print("📁 TEST 3: Check Companies Folder")
    print("="*80)
    
    # Try different paths
    paths_to_try = [
        "../../inject_collect_project/merged_profiles",
        "../inject_collect_project/merged_profiles", 
        "inject_collect_project/merged_profiles",
        r"c:\Users\Dell\Desktop\outbound_project\inject_collect_project\merged_profiles"
    ]
    
    for path in paths_to_try:
        p = Path(path)
        print(f"\n   Path: {path}")
        print(f"   Resolved: {p.absolute()}")
        print(f"   Exists: {p.exists()}")
        if p.exists():
            json_files = list(p.glob("*.json"))
            print(f"   JSON Files: {len(json_files)}")
            if json_files:
                print(f"   First 3: {[f.name for f in json_files[:3]]}")
                return True
    
    return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("🕵️ MCP SERVER TOOL TESTS")
    print("="*80)
    print("\nThis tests the 2 high-level orchestrator tools:")
    print("  1. rank_lead - Analyze single company against ICP")
    print("  2. detect_top_leads - Full pipeline with dynamic graph")
    
    # Test 1: rank_lead
    result1 = test_rank_lead()
    
    # Test 2: Check companies folder
    folder_ok = test_load_companies()
    
    # Test 3: Full pipeline
    if folder_ok:
        print("\n   ✅ Companies folder found, running full pipeline...")
        result2 = test_detect_top_leads()
    else:
        print("\n   ⚠️ Companies folder not found - checking folder paths...")
        # Still try detect_top_leads - it has hardcoded fallback
        print("   Trying detect_top_leads with absolute path...")
        result2 = test_detect_top_leads()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()

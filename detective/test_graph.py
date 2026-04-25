#!/usr/bin/env python3
"""
Test LangGraph-based detective pipeline
"""

from detective_graph import run_detective_pipeline


def test_graph_pipeline():
    """Test the LangGraph pipeline"""
    
    example_icp = """
    I want IT companies with 50-500 employees and annual revenue between $10M - $100M.
    Target companies should be in United States, Canada, and Germany.
    
    We want to connect with Sales Managers, VPs of Engineering, and CTOs.
    
    Must-have traits:
    - Using modern tech stack (React, Python, AWS)
    - In growth stage with Series B or C funding
    """
    
    print("=" * 80)
    print("TESTING LANGGRAPH PIPELINE")
    print("=" * 80)
    
    # Run pipeline
    result = run_detective_pipeline(example_icp, "graph_test")
    
    # Check results
    print("\n" + "=" * 80)
    print("RESULT SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Pipeline Steps:")
    print(f"   Step completed: {result.get('step_completed', 'none')}")
    
    print(f"\n📈 Data Flow:")
    print(f"   ICP extracted: {result.get('icp_attributes') is not None}")
    print(f"   Companies matched: {len(result.get('matched_companies', {}))}")
    print(f"   After geo-filter: {len(result.get('geo_filtered_companies', {}))}")
    print(f"   After criteria filter: {len(result.get('filtered_companies', {}))}")
    print(f"   Ranked: {len(result.get('ranking_results', []))}")
    print(f"   Final ranked: {len(result.get('final_rankings', []))}")
    print(f"   Persona results: {len(result.get('persona_results', []))}")
    
    print(f"\n📁 Output Files:")
    for name, path in result.get('output_files', {}).items():
        print(f"   • {name}: {path}")
    
    if result.get('errors'):
        print(f"\n⚠️ Errors ({len(result['errors'])}):")
        for err in result['errors']:
            print(f"   • {err}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_graph_pipeline()

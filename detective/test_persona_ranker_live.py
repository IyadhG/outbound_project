#!/usr/bin/env python3
"""
Test Persona Ranker with REAL companies from final ranking
"""

import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from ranking import PersonaRanker


def test_persona_ranker_live():
    """Test persona ranking on real companies from final ranking"""
    
    print("=" * 80)
    print("LIVE TEST: Persona Ranker (LLM-Based)")
    print("=" * 80)
    
    # Load final ranking
    ranking_file = Path('ranking/example_icp_final_final_ranking.json')
    
    if not ranking_file.exists():
        print(f"[ERROR] Ranking file not found: {ranking_file}")
        print("Run main.py first to generate final ranking.")
        return
    
    with open(ranking_file, 'r', encoding='utf-8') as f:
        ranking_data = json.load(f)
    
    companies = ranking_data.get('rankings', [])
    
    print(f"\n[LOADED] {len(companies)} companies from final ranking")
    
    # Initialize persona ranker
    personas_folder = Path('../inject_collect_project/personas_discovered')
    if not personas_folder.exists():
        personas_folder = Path('personas_discovered')
    
    print(f"[FOLDER] Looking for personas in: {personas_folder}")
    
    # Check which companies have personas
    available_personas = list(personas_folder.glob('*_personas.json'))
    print(f"[FOUND] {len(available_personas)} persona files")
    
    for company in companies:
        key = company.get('company_key', '')
        persona_file = personas_folder / f"{key}_personas.json"
        status = "✓" if persona_file.exists() else "✗"
        print(f"  {status} {company['name']}")
    
    # Initialize ranker
    print("\n[INIT] Initializing LLM-based PersonaRanker...")
    ranker = PersonaRanker(personas_folder)
    
    # Target roles from ICP
    target_roles = ["VPs of Engineering", "CTOs", "Heads of Product"]
    print(f"\n[ICP] Target roles: {', '.join(target_roles)}")
    
    # Rank personas
    print("\n" + "=" * 80)
    print("RANKING PERSONAS WITH LLM...")
    print("=" * 80)
    
    results = ranker.rank_personas_for_companies(companies, target_roles)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    if results:
        ranker.print_persona_rankings(results)
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        sales_count = sum(1 for p in results if p['persona_score']['is_sales'])
        ceo_count = sum(1 for p in results if p['persona_score']['is_ceo'])
        other_count = len(results) - sales_count - ceo_count
        
        print(f"Companies processed: {len(results)}/{len(companies)}")
        print(f"  - Sales personas: {sales_count}")
        print(f"  - CEO/Founder (fallback): {ceo_count}")
        print(f"  - Other high-seniority: {other_count}")
        
        # Save results
        output_file = ranker.save_persona_rankings(
            results,
            "example_icp_test",
            ranking_data.get('icp_text_preview', '')
        )
        print(f"\n[SAVED] Results to: {output_file}")
        
    else:
        print("\n[WARNING] No persona results generated")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_persona_ranker_live()

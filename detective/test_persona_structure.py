#!/usr/bin/env python3
"""
Quick test - Verify persona files structure without LLM calls
"""

import json
from pathlib import Path


def test_persona_structure():
    """Test loading persona files"""
    
    print("=" * 80)
    print("TEST: Persona File Structure")
    print("=" * 80)
    
    # Load final ranking
    ranking_file = Path('ranking/example_icp_final_final_ranking.json')
    
    with open(ranking_file, 'r', encoding='utf-8') as f:
        ranking_data = json.load(f)
    
    companies = ranking_data.get('rankings', [])
    
    # Check personas folder
    personas_folder = Path('../inject_collect_project/personas_discovered')
    
    print(f"\nCompanies in final ranking: {len(companies)}")
    print(f"Personas folder: {personas_folder}")
    print(f"Folder exists: {personas_folder.exists()}")
    
    for company in companies:
        key = company.get('company_key', '')
        name = company.get('name', '')
        persona_file = personas_folder / f"{key}_personas.json"
        
        print(f"\n{'='*60}")
        print(f"Company: {name}")
        print(f"Key: {key}")
        print(f"Persona file: {persona_file.name}")
        print(f"File exists: {persona_file.exists()}")
        
        if persona_file.exists():
            with open(persona_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both dict with 'personas' key or direct list
            if isinstance(data, list):
                personas = data
            elif isinstance(data, dict):
                personas = data.get('personas', [])
            else:
                personas = []
            
            print(f"Personas count: {len(personas)}")
            
            if personas:
                print("\nSample personas:")
                for i, p in enumerate(personas[:3], 1):
                    print(f"  {i}. {p.get('full_name', 'N/A')}")
                    print(f"     Title: {p.get('job_title', 'N/A')}")
                    print(f"     Role: {p.get('job_title_role', 'N/A')}")
                    print(f"     Level: {p.get('job_title_level', 'N/A')}")
                    if p.get('emails'):
                        print(f"     Email: {p['emails'][0]}")
    
    print("\n" + "=" * 80)
    print("Structure test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_persona_structure()

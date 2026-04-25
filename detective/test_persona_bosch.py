#!/usr/bin/env python3
"""
Test Persona Ranker with actual Bosch personas structure
"""

import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from ranking import PersonaRanker


def test_bosch_personas():
    """Test LLM persona ranking on Bosch with real data"""
    
    print("=" * 80)
    print("TEST: Bosch Personas with LLM Ranking")
    print("=" * 80)
    
    # Company info
    company = {
        "company_key": "bosch_us",
        "name": "Bosch in the USA (Robert Bosch LLC)",
        "rank": 1,
        "final_score": 0.7155
    }
    
    target_roles = ["VPs of Engineering", "CTOs", "Heads of Product"]
    
    print(f"\n[COMPANY] {company['name']}")
    print(f"[TARGET ROLES] {', '.join(target_roles)}")
    print(f"[PRIORITY] Sales > CEO > Other High Seniority")
    
    # Initialize ranker
    personas_folder = Path('../inject_collect_project/personas_discovered')
    print(f"\n[INIT] Loading PersonaRanker...")
    ranker = PersonaRanker(personas_folder)
    
    # Load and show personas
    print("\n" + "=" * 80)
    print("PERSONAS FOUND:")
    print("=" * 80)
    
    personas = ranker.load_personas_for_company(company['company_key'])
    
    for i, p in enumerate(personas, 1):
        current_title = ranker._extract_current_title(p)
        print(f"\n{i}. {p.get('full_name', 'N/A')}")
        print(f"   Current Title: {current_title}")
        print(f"   Role Category: {p.get('job_title_role', 'N/A')}")
        print(f"   Level: {p.get('job_title_level', 'N/A')}")
        email = ranker._extract_primary_email(p)
        if email:
            print(f"   Email: {email}")
    
    # Rank with LLM
    print("\n" + "=" * 80)
    print("LLM SCORING...")
    print("=" * 80)
    
    result = ranker.rank_personas_for_company(
        company['company_key'],
        company['name'],
        target_roles
    )
    
    if result:
        print("\n" + "=" * 80)
        print("🎯 SELECTED TARGET PERSONA")
        print("=" * 80)
        
        persona = result['selected_persona']
        score = result['persona_score']
        
        print(f"\nName: {persona['full_name']}")
        print(f"Title: {persona['job_title']}")
        print(f"Role: {persona['job_title_role']} | Level: {persona['job_title_level']}")
        print(f"Location: {persona.get('city', 'N/A')}, {persona.get('country', 'N/A')}")
        
        print(f"\n📊 Scores:")
        print(f"   Final: {score['final_score']:.3f}")
        print(f"   ├─ Seniority: {score['seniority_score']:.2f}")
        print(f"   ├─ Department: {score['department_score']:.2f}")
        print(f"   └─ Role Match: {score['role_match_score']:.2f}")
        
        if score['is_sales']:
            print(f"\n✅ SALES DEPARTMENT - IDEAL TARGET")
        elif score['is_ceo']:
            print(f"\n⚠️ CEO/FOUNDER - FALLBACK TARGET")
        else:
            print(f"\n📌 HIGH SENIORITY - BEST AVAILABLE")
        
        print(f"\n🧠 LLM Reasoning: {result['selection_reason']}")
        
        if persona.get('email'):
            print(f"\n📧 Contact: {persona['email']}")
        
        if persona['linkedin_url']:
            print(f"🔗 LinkedIn: {persona['linkedin_url']}")
        
        # Save
        output_file = ranker.save_persona_rankings([result], "bosch_live", "Bosch USA targeting")
        print(f"\n💾 Saved to: {output_file}")
        
    else:
        print("\n❌ No personas found or scoring failed")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_bosch_personas()

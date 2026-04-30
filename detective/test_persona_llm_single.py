#!/usr/bin/env python3
"""
Test LLM Persona Ranker on single company (saves tokens)
"""

import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from ranking import PersonaRanker


def test_single_company():
    """Test LLM persona ranking on Bosch only"""
    
    print("=" * 80)
    print("LLM PERSONA RANKER TEST - Single Company (Token Saver)")
    print("=" * 80)
    
    # Test just Bosch (has good persona data)
    company = {
        "company_key": "bosch_us",
        "name": "Bosch in the USA (Robert Bosch LLC)",
        "rank": 1,
        "final_score": 0.7155
    }
    
    target_roles = ["VPs of Engineering", "CTOs", "Heads of Product"]
    
    print(f"\n[COMPANY] {company['name']}")
    print(f"[TARGET ROLES] {', '.join(target_roles)}")
    
    # Initialize ranker
    personas_folder = Path('../inject_collect_project/personas_discovered')
    print(f"\n[INIT] Loading PersonaRanker with LLM...")
    ranker = PersonaRanker(personas_folder)
    
    # Rank personas
    print("\n" + "=" * 80)
    print("CALLING GROQ LLM TO SCORE PERSONAS...")
    print("=" * 80)
    
    result = ranker.rank_personas_for_company(
        company['company_key'],
        company['name'],
        target_roles
    )
    
    if result:
        print("\n" + "=" * 80)
        print("LLM SCORING RESULTS")
        print("=" * 80)
        
        persona = result['selected_persona']
        score = result['persona_score']
        
        print(f"\n🎯 SELECTED TARGET:")
        print(f"   Name: {persona['full_name']}")
        print(f"   Title: {persona['job_title']}")
        print(f"   Role: {persona['job_title_role']} | Level: {persona['job_title_level']}")
        print(f"\n📊 LLM SCORES:")
        print(f"   Final Score: {score['final_score']:.3f}")
        print(f"   ├─ Seniority: {score['seniority_score']:.2f}")
        print(f"   ├─ Department: {score['department_score']:.2f}")
        print(f"   └─ Role Match: {score['role_match_score']:.2f}")
        
        if score['is_sales']:
            print(f"\n✅ Department: SALES (Ideal)")
        elif score['is_ceo']:
            print(f"\n⚠️ Department: CEO/Founder (Fallback)")
        
        print(f"\n🧠 LLM Reasoning:")
        print(f"   {result['selection_reason']}")
        
        if persona['emails']:
            print(f"\n📧 Email: {persona['emails'][0]}")
        
        if persona['linkedin_url']:
            print(f"🔗 LinkedIn: {persona['linkedin_url']}")
        
        # Save result
        output_file = ranker.save_persona_rankings(
            [result],
            "bosch_test",
            "IT companies test"
        )
        print(f"\n💾 Saved to: {output_file}")
        
    else:
        print("\n❌ No personas found or LLM scoring failed")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_single_company()

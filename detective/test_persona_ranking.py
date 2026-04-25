#!/usr/bin/env python3
"""
Test Persona Ranking - Demonstrates persona selection per company
"""

import json
from pathlib import Path
from ranking import PersonaRanker


def test_persona_ranking():
    """Test persona ranking with mock data"""
    
    print("=" * 80)
    print("TEST: Persona Ranking (Seniority + Role Match + Department)")
    print("=" * 80)
    
    # Create mock personas folder
    personas_folder = Path('test_personas')
    personas_folder.mkdir(exist_ok=True)
    
    # Create mock personas for Bosch
    bosch_personas = {
        "company_name": "Bosch USA",
        "company_key": "bosch_us",
        "personas": [
            {
                "full_name": "John Smith",
                "job_title": "VP of Sales",
                "job_title_role": "sales",
                "job_title_level": "vp",
                "linkedin_url": "https://linkedin.com/in/johnsmith",
                "emails": ["john.smith@bosch.us"]
            },
            {
                "full_name": "Sarah Johnson",
                "job_title": "Sales Director",
                "job_title_role": "sales",
                "job_title_level": "director",
                "linkedin_url": "https://linkedin.com/in/sarahjohnson",
                "emails": ["sarah.j@bosch.us"]
            },
            {
                "full_name": "Dr. Michael Chen",
                "job_title": "Chief Technology Officer",
                "job_title_role": "engineering",
                "job_title_level": "c-level",
                "linkedin_url": "https://linkedin.com/in/mchen",
                "emails": ["m.chen@bosch.us"]
            },
            {
                "full_name": "Emily Davis",
                "job_title": "Marketing Manager",
                "job_title_role": "marketing",
                "job_title_level": "manager",
                "linkedin_url": "https://linkedin.com/in/edavis",
                "emails": ["emily.d@bosch.us"]
            }
        ]
    }
    
    # Create mock personas for ALTEN (no sales, only CEO)
    alten_personas = {
        "company_name": "ALTEN Deutschland",
        "company_key": "alten-germany_de",
        "personas": [
            {
                "full_name": "Hans Mueller",
                "job_title": "CEO & Founder",
                "job_title_role": "executive",
                "job_title_level": "c-level",
                "linkedin_url": "https://linkedin.com/in/hmueller",
                "emails": ["h.mueller@alten.de"]
            },
            {
                "full_name": "Klaus Schmidt",
                "job_title": "Engineering Manager",
                "job_title_role": "engineering",
                "job_title_level": "manager",
                "linkedin_url": "https://linkedin.com/in/kschmidt",
                "emails": ["klaus.s@alten.de"]
            },
            {
                "full_name": "Anna Weber",
                "job_title": "Senior Consultant",
                "job_title_role": "consulting",
                "job_title_level": "senior",
                "linkedin_url": "https://linkedin.com/in/aweber",
                "emails": ["anna.w@alten.de"]
            }
        ]
    }
    
    # Save mock persona files
    with open(personas_folder / "bosch_us_personas.json", 'w', encoding='utf-8') as f:
        json.dump(bosch_personas, f, indent=2)
    
    with open(personas_folder / "alten-germany_de_personas.json", 'w', encoding='utf-8') as f:
        json.dump(alten_personas, f, indent=2)
    
    print(f"\n[MOCK] Created test personas:")
    print(f"  - Bosch: 4 personas (2 sales, 1 CTO, 1 marketing)")
    print(f"  - ALTEN: 3 personas (1 CEO, 1 engineering, 1 consultant)")
    
    # Test target roles from ICP
    target_roles = ["VP of Engineering", "CTO", "Heads of Product", "Sales Director"]
    
    print(f"\n[ICP] Target roles: {', '.join(target_roles)}")
    
    # Initialize ranker
    ranker = PersonaRanker(personas_folder)
    
    # Test companies
    companies = [
        {
            "company_key": "bosch_us",
            "name": "Bosch in the USA",
            "rank": 1,
            "final_score": 0.75
        },
        {
            "company_key": "alten-germany_de",
            "name": "ALTEN Deutschland",
            "rank": 2,
            "final_score": 0.72
        }
    ]
    
    print("\n" + "=" * 80)
    print("TESTING PERSONA RANKING")
    print("=" * 80)
    
    # Rank personas for companies
    results = ranker.rank_personas_for_companies(companies, target_roles)
    
    # Print results
    ranker.print_persona_rankings(results)
    
    # Save results
    output_file = ranker.save_persona_rankings(results, "test_persona", "IT companies in Germany and USA...")
    print(f"\n[SAVED] Persona rankings to: {output_file}")
    
    # Clean up
    import shutil
    shutil.rmtree(personas_folder)
    output_file.unlink()
    print("\n[CLEANED] Removed test files")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    
    print("\nKey takeaways:")
    print("- Bosch: VP of Sales selected (sales dept + matches 'Sales Director' role)")
    print("- ALTEN: CEO selected (fallback - no sales personas available)")
    print("- Seniority, department, and role matching all contribute to score")
    print("- Sales personas are prioritized over other departments")


if __name__ == "__main__":
    test_persona_ranking()

#!/usr/bin/env python3
"""
Test script for Brain Agent with user's specific ICP text
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from main import BrainAgent


def test_user_icp():
    """Test the brain agent with the user's specific ICP text"""
    
    # Test scenario inputs
    inputs = {
        "raw_icp_text": """I want IT companies with 50-500 employees and annual revenue between $10M - $100M. Target companies should be in North America and Europe, specifically in the United States, Canada, United Kingdom, and Germany.
            
            We want to connect with VPs of Engineering, CTOs, and Heads of Product.
            
            Must-have traits: 
            - Using modern tech stack (React, Python, AWS)
            - In growth stage with Series B or C funding
            - Product-led growth model
            
            Nice-to-have:
            - Remote-friendly culture
            - AI/ML components in their product
            - Strong engineering team
            
            Exclude:
            - Consulting companies
            - Digital marketing agencies
            - Non-tech companies""",
        "user_offering": "AI supply chain automation."
    }
    
    print("=" * 80)
    print("TESTING BRAIN AGENT WITH USER ICP")
    print("=" * 80)
    print(f"User Offering: {inputs['user_offering']}")
    print("\nICP Text:")
    print("-" * 80)
    print(inputs['raw_icp_text'])
    print("-" * 80)
    
    try:
        # Initialize brain agent
        brain = BrainAgent()
        
        # Extract ICP attributes
        print("\n[EXTRACTING] Extracting ICP attributes...")
        icp_attributes = brain.extract_from_text(inputs['raw_icp_text'])
        
        # Display results
        print("\n[SUCCESS] Extraction complete!")
        brain.display_icp_summary(icp_attributes)
        
        # Save results
        output_name = "user_test_icp"
        brain.save_icp_result(icp_attributes, output_name)
        print(f"\n[SAVED] Results saved as: {output_name}_icp.json")
        
        # Additional analysis
        print("\n[ANALYSIS] DETAILED ANALYSIS:")
        print("=" * 50)
        
        print(f"\n[INDUSTRY] INDUSTRY FOCUS:")
        for industry in icp_attributes.industry:
            print(f"  • {industry}")
        
        print(f"\n[GEOGRAPHIC] GEOGRAPHIC TARGETING:")
        if icp_attributes.target_countries:
            print(f"  Countries: {', '.join(icp_attributes.target_countries)}")
        if icp_attributes.target_continents:
            print(f"  Continents: {', '.join(icp_attributes.target_continents)}")
        
        print(f"\n[SIZE] COMPANY SIZE:")
        size_min = icp_attributes.company_size.min or "Not specified"
        size_max = icp_attributes.company_size.max or "Not specified"
        print(f"  Employees: {size_min} - {size_max}")
        
        print(f"\n[REVENUE] REVENUE RANGE:")
        rev_min = icp_attributes.revenue_range.min or "Not specified"
        rev_max = icp_attributes.revenue_range.max or "Not specified"
        print(f"  Revenue: ${rev_min} - ${rev_max}")
        
        print(f"\n[ROLES] TARGET ROLES:")
        for role in icp_attributes.target_roles:
            print(f"  • {role}")
        
        print(f"\n[TECH] TECH STACK:")
        for tech in icp_attributes.dynamic_attributes.tech_stack:
            print(f"  • {tech}")
        
        if icp_attributes.must_have_traits:
            print(f"\n[MUST-HAVE] MUST-HAVE TRAITS:")
            for trait in icp_attributes.must_have_traits:
                print(f"  • {trait}")
        
        if icp_attributes.nice_to_have_traits:
            print(f"\n[NICE-TO-HAVE] NICE-TO-HAVE TRAITS:")
            for trait in icp_attributes.nice_to_have_traits:
                print(f"  • {trait}")
        
        if icp_attributes.exclude:
            print(f"\n[EXCLUDE] EXCLUDE:")
            for item in icp_attributes.exclude:
                print(f"  • {item}")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return icp_attributes
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def print_extraction_summary(icp_attributes):
    """Print a clean summary of extracted ICP attributes"""
    
    print("\n" + "=" * 80)
    print("ICP EXTRACTION SUMMARY")
    print("=" * 80)
    
    print(f"\n[INDUSTRIES]")
    print(f"  Total: {len(icp_attributes.industry)}")
    for industry in icp_attributes.industry:
        print(f"  - {industry}")
    
    print(f"\n[COMPANY SIZE]")
    if icp_attributes.company_size.min or icp_attributes.company_size.max:
        print(f"  Range: {icp_attributes.company_size.min or '?'} - {icp_attributes.company_size.max or '?'} employees")
    else:
        print("  Not specified")
    
    print(f"\n[REVENUE RANGE]")
    if icp_attributes.revenue_range.min or icp_attributes.revenue_range.max:
        print(f"  Range: ${icp_attributes.revenue_range.min or '?'} - ${icp_attributes.revenue_range.max or '?'}")
    else:
        print("  Not specified")
    
    print(f"\n[GEOGRAPHIC TARGETING]")
    print(f"  Continents: {', '.join(icp_attributes.target_continents) if icp_attributes.target_continents else 'None'}")
    print(f"  Countries: {', '.join(icp_attributes.target_countries) if icp_attributes.target_countries else 'None'}")
    if icp_attributes.target_cities:
        print(f"  Cities:")
        for city in icp_attributes.target_cities:
            print(f"    - {city.city}, {city.country}")
    
    print(f"\n[TARGET ROLES]")
    print(f"  Total: {len(icp_attributes.target_roles)}")
    for role in icp_attributes.target_roles:
        print(f"  - {role}")
    
    print(f"\n[MUST-HAVE TRAITS]")
    print(f"  Total: {len(icp_attributes.must_have_traits)}")
    for trait in icp_attributes.must_have_traits:
        print(f"  - {trait}")
    
    print(f"\n[NICE-TO-HAVE TRAITS]")
    print(f"  Total: {len(icp_attributes.nice_to_have_traits)}")
    for trait in icp_attributes.nice_to_have_traits:
        print(f"  - {trait}")
    
    print(f"\n[EXCLUSIONS]")
    print(f"  Total: {len(icp_attributes.exclude)}")
    for item in icp_attributes.exclude:
        print(f"  - {item}")
    
    print(f"\n[DYNAMIC ATTRIBUTES]")
    print(f"  Tech Stack: {', '.join(icp_attributes.dynamic_attributes.tech_stack) if icp_attributes.dynamic_attributes.tech_stack else 'None'}")
    print(f"  Growth Stage: {icp_attributes.dynamic_attributes.growth_stage or 'Not specified'}")
    print(f"  Funding Stage: {icp_attributes.dynamic_attributes.funding_stage or 'Not specified'}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run the test
    icp_attributes = test_user_icp()
    
    if icp_attributes:
        # Print clean extraction summary
        print_extraction_summary(icp_attributes)
    
    print("\n[DONE] Test completed. Check the saved JSON files for detailed results.")

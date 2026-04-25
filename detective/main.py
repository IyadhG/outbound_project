#!/usr/bin/env python3
"""
Main Orchestrator - LangGraph-based ICP extraction and company targeting
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Import the LangGraph pipeline
from detective_graph import run_detective_pipeline

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def display_icp_summary(icp_attributes):
    """Display a summary of extracted ICP attributes"""
    print("\n" + "=" * 60)
    print("EXTRACTED ICP ATTRIBUTES")
    print("=" * 60)
    
    print(f"Industries: {', '.join(icp_attributes.industry) if icp_attributes.industry else 'None'}")
    
    if icp_attributes.company_size.min or icp_attributes.company_size.max:
        size_range = f"{icp_attributes.company_size.min or '?'} - {icp_attributes.company_size.max or '?' }"
        print(f"Company Size: {size_range} employees")
    
    if icp_attributes.revenue_range.min or icp_attributes.revenue_range.max:
        revenue_range = f"${icp_attributes.revenue_range.min or '?'} - ${icp_attributes.revenue_range.max or '?' }"
        print(f"Revenue Range: {revenue_range}")
    
    print(f"Target Countries: {', '.join(icp_attributes.target_countries) if icp_attributes.target_countries else 'None'}")
    print(f"Target Roles: {', '.join(icp_attributes.target_roles) if icp_attributes.target_roles else 'None'}")
    
    if icp_attributes.dynamic_attributes.tech_stack:
        print(f"Tech Stack: {', '.join(icp_attributes.dynamic_attributes.tech_stack)}")
    
    print("=" * 60)


def main():
    """Main entry point - uses LangGraph pipeline for orchestration"""
    
    # Load environment variables
    load_dotenv()
    
    if not os.getenv('GROQ_API_KEY'):
        print("[ERROR] GROQ_API_KEY not found in environment variables")
        return
    
    # Example ICP text - MODIFY THIS WITH YOUR ACTUAL ICP
    example_icp = """
    I want IT companies with 50 -5000 employees and annual revenue between $10M - $100M. 
    Target companies should be in North America and Europe, specifically in the United States, 
    Canada, United Kingdom, and Germany.
    
    We want to connect with Sales Manager, CTOs, and Heads of Product.
    
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
    - Non-tech companies
    """
    
    output_name = "example_icp"
    
    print("=" * 80)
    print("🕵️ DETECTIVE PIPELINE - LangGraph Orchestration")
    print("=" * 80)
    print("\nUsing example ICP text (modify in main.py for your needs)")
    
    # Run the LangGraph pipeline
    try:
        result = run_detective_pipeline(example_icp, output_name)
        
        # Show summary
        print("\n" + "=" * 80)
        print("📊 PIPELINE SUMMARY")
        print("=" * 80)
        print(f"\nStep completed: {result.get('step_completed', 'unknown')}")
        print(f"Companies matched: {len(result.get('matched_companies', {}))}")
        print(f"Final ranked: {len(result.get('final_rankings', []))}")
        print(f"Persona targets: {len(result.get('persona_results', []))}")
        
        if result.get('errors'):
            print(f"\n⚠️ Errors: {len(result['errors'])}")
            for err in result['errors'][:3]:  # Show first 3 errors
                print(f"  • {err}")
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        print(f"\n[ERROR] Pipeline failed: {e}")


if __name__ == "__main__":
    main()

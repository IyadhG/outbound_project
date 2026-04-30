#!/usr/bin/env python3
"""
Test Intent Collection Only - Skip ICP extraction and matching
Uses already saved matched_companies_for_intent.json to save tokens
"""

import sys
import json
import asyncio
import importlib.util
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables for Groq API key
load_dotenv()

# Add agentic_intent to path
agentic_intent_folder = str(Path(__file__).parent.parent / 'agentic_intent')
if agentic_intent_folder not in sys.path:
    sys.path.insert(0, agentic_intent_folder)


def load_matched_companies():
    """Load company list from previously saved file"""
    json_file = Path('matched_companies_for_intent.json')
    
    if not json_file.exists():
        print("[ERROR] matched_companies_for_intent.json not found!")
        print("Run python main.py first to generate matched companies.")
        return None
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def run_intent_collection(company_names):
    """Run agentic_intent on the provided companies"""
    print("=" * 80)
    print("INTENT COLLECTION TEST")
    print("=" * 80)
    print(f"\nCompanies to analyze: {', '.join(company_names)}")
    print(f"Total companies: {len(company_names)}")
    
    try:
        # Import agentic_intent main function
        if 'agentic_intent_module' in sys.modules:
            del sys.modules['agentic_intent_module']
        
        spec = importlib.util.spec_from_file_location(
            "agentic_intent_module", 
            Path(agentic_intent_folder) / 'main.py'
        )
        agentic_module = importlib.util.module_from_spec(spec)
        sys.modules['agentic_intent_module'] = agentic_module
        spec.loader.exec_module(agentic_module)
        
        print("\n[STEP 3] Collecting intent...")
        
        # Run agentic_intent with matched companies
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        intent_result = loop.run_until_complete(agentic_module.main(
            companies=company_names,
            save_to_file=True
        ))
        
        if intent_result:
            print(f"\n[SUCCESS] Intent collection completed for {len(company_names)} companies")
            return intent_result
        else:
            print("\n[WARNING] Intent collection returned no results")
            return None
            
    except Exception as e:
        print(f"\n[ERROR] Intent collection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point"""
    # Load matched companies
    data = load_matched_companies()
    if not data:
        return
    
    company_names = data['companies']
    icp_name = data.get('icp_name', 'unknown')
    
    print(f"\nLoaded {len(company_names)} companies from ICP: {icp_name}")
    
    # Run intent collection
    result = run_intent_collection(company_names)
    
    # Summary
    print("\n" + "=" * 80)
    if result:
        print("INTENT COLLECTION: SUCCESS")
        print(f"Output saved to: company_intel_*.json")
    else:
        print("INTENT COLLECTION: FAILED")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    main()

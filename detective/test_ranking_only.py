#!/usr/bin/env python3
"""
Test Ranking Only - Embeds ICP and companies, calculates cosine similarity
Uses already saved matched_companies_example_icp folder to save tokens
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import ranking module
from ranking import CompanyRanker, CompanyFilter


def load_icp_text():
    """Load ICP text from saved JSON or use default"""
    icp_file = Path('example_icp_icp.json')
    
    if icp_file.exists():
        with open(icp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Try to get raw text
            raw_text = data.get('raw_text')
            if raw_text:
                return raw_text
    
    # Default example ICP if file not found or no raw_text
    return """We are looking for IT companies with 50-500 employees and annual revenue between $10M - $100M. 
    Target companies should be in North America and Europe, specifically in the United States, Canada, United Kingdom, and Germany.
    
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
    - Non-tech companies"""


def main():
    """Main entry point"""
    print("=" * 80)
    print("RANKING TEST - ICP Similarity Scoring")
    print("=" * 80)
    
    # Find matched companies folder
    matched_folder = Path('matched_companies_example_icp')
    
    if not matched_folder.exists():
        print(f"[ERROR] Folder not found: {matched_folder}")
        print("Run main.py first to generate matched companies.")
        return
    
    # Get list of company files
    company_files = list(matched_folder.glob('*_MATCHED.json'))
    
    if not company_files:
        print(f"[ERROR] No company files found in {matched_folder}")
        return
    
    print(f"\nFound {len(company_files)} company profiles to rank")
    
    # Load ICP text
    icp_text = load_icp_text()
    print(f"\nICP text loaded ({len(icp_text)} characters)")
    print(f"Preview: {icp_text[:100]}...")
    
    try:
        # Initialize ranker
        print("\n[STEP 1] Initializing Gemini embedder...")
        ranker = CompanyRanker()
        
        # Embed ICP
        print("[STEP 2] Embedding ICP text...")
        ranker.embed_icp(icp_text)
        print(f"ICP embedded successfully! (dim: {len(ranker.icp_embedding)})")
        
        # Process each company
        print(f"\n[STEP 3] Processing {len(company_files)} companies...")
        results = []
        
        for i, json_file in enumerate(company_files, 1):
            try:
                # Load company profile
                with open(json_file, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                
                company_name = profile.get('name', json_file.stem)
                print(f"\n{i}. {company_name}")
                
                # Create description
                description = ranker.create_company_description(profile)
                print(f"   Description: {description[:80]}...")
                
                # Embed description
                company_embedding = ranker.embedder.embed_text(description)
                
                # Calculate similarity
                similarity = ranker.cosine_similarity(ranker.icp_embedding, company_embedding)
                
                result = {
                    'rank': 0,  # Will be set after sorting
                    'company_key': json_file.stem.replace('_MATCHED', ''),
                    'name': company_name,
                    'description': description,
                    'similarity_score': round(similarity, 4),
                    'embedding_dim': len(company_embedding)
                }
                results.append(result)
                
                print(f"   Similarity Score: {similarity:.4f}")
                
            except Exception as e:
                print(f"   [ERROR] Failed to process: {e}")
                continue
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Assign ranks
        for i, r in enumerate(results, 1):
            r['rank'] = i
        
        # Print final rankings
        print("\n" + "=" * 80)
        print("FINAL RANKINGS (by Cosine Similarity to ICP)")
        print("=" * 80)
        
        for r in results[:10]:  # Top 10
            print(f"\n{r['rank']}. {r['name']}")
            print(f"   Score: {r['similarity_score']:.4f}")
            print(f"   Desc: {r['description'][:60]}...")
        
        # Save results
        output_file = Path('ranking') / 'test_ranking_results.json'
        output_file.parent.mkdir(exist_ok=True)
        
        output_data = {
            'icp_text_preview': icp_text[:200] + '...',
            'total_companies': len(results),
            'rankings': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUCCESS] Results saved to: {output_file}")
        print(f"Top match: {results[0]['name']} (score: {results[0]['similarity_score']:.4f})")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("RANKING TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()

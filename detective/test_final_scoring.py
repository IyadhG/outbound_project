#!/usr/bin/env python3
"""
Test Final Scoring - Demonstrates similarity + intent combination
"""

import json
from pathlib import Path
from ranking import FinalScorer


def test_final_scoring():
    """Test final scoring with mock data"""
    
    print("=" * 80)
    print("TEST: Final Scoring (Similarity + Intent)")
    print("=" * 80)
    
    # Create mock similarity ranking
    mock_ranking = {
        "icp_text_preview": "IT companies in Germany...",
        "total_companies": 2,
        "rankings": [
            {
                "rank": 1,
                "company_key": "alten-germany_de",
                "name": "ALTEN Deutschland",
                "similarity_score": 0.7133,
                "description": "IT consulting company in Germany"
            },
            {
                "rank": 2,
                "company_key": "bosch_us",
                "name": "Bosch in the USA (Robert Bosch LLC)",
                "similarity_score": 0.7171,
                "description": "Technology company in USA"
            }
        ]
    }
    
    # Create mock intent results (with meaningful intent for ALTEN)
    mock_intent = {
        "metadata": {"companies_analyzed": ["ALTEN Deutschland", "Bosch in the USA (Robert Bosch LLC)"]},
        "companies": {
            "ALTEN Deutschland": {
                "funding_events": [
                    {"financial_details": {"amount": "50M EUR", "amount_confidence": 0.8}},
                    {"financial_details": {"amount": "Unknown", "amount_confidence": 0.2}}
                ],
                "news_events": [
                    {"event": {"title": "ALTEN Deutschland expands operations in Munich", "confidence": 0.85}},
                    {"event": {"title": "New AI division launched", "confidence": 0.75}}
                ],
                "summary": {
                    "total_funding_events": 2,
                    "high_confidence_funding": 1,
                    "total_news_events": 2,
                    "high_confidence_news": 2
                }
            },
            "Bosch in the USA (Robert Bosch LLC)": {
                "funding_events": [],
                "news_events": [],
                "summary": {
                    "total_funding_events": 0,
                    "high_confidence_funding": 0,
                    "total_news_events": 0,
                    "high_confidence_news": 0
                }
            }
        }
    }
    
    # Save mock files
    ranking_file = Path('ranking/test_ranking.json')
    intent_file = Path('ranking/test_intent.json')
    ranking_file.parent.mkdir(exist_ok=True)
    
    with open(ranking_file, 'w', encoding='utf-8') as f:
        json.dump(mock_ranking, f, indent=2)
    
    with open(intent_file, 'w', encoding='utf-8') as f:
        json.dump(mock_intent, f, indent=2)
    
    print(f"\n[MOCK] Created test files:")
    print(f"  - {ranking_file}")
    print(f"  - {intent_file}")
    
    # Test final scoring
    print("\n[TEST] Running FinalScorer...")
    final_scorer = FinalScorer(intent_boost=0.05)
    
    final_rankings = final_scorer.calculate_final_scores(ranking_file, intent_file)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for company in final_rankings:
        print(f"\n{company['rank']}. {company['name']}")
        print(f"   Similarity: {company['similarity_score']:.4f}")
        print(f"   Intent Score: {company['intent_score']:.2f}")
        print(f"   Intent Boost: +{company['intent_boost']:.4f}")
        print(f"   FINAL SCORE: {company['final_score']:.4f}")
        
        if company['has_meaningful_intent']:
            print(f"   ✓ Meaningful intent detected!")
            print(f"   Funding: {company['funding_events']} events")
            print(f"   News: {company['news_events']} events")
    
    # Save final ranking
    final_file = final_scorer.save_final_ranking(final_rankings, "test_final", mock_ranking['icp_text_preview'])
    print(f"\n[SAVED] Final ranking to: {final_file}")
    
    # Clean up
    ranking_file.unlink()
    intent_file.unlink()
    final_file.unlink()
    print("\n[CLEANED] Removed test files")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    
    print("\nKey takeaways:")
    print("- ALTEN Deutschland had meaningful intent (funding + news)")
    print("- Got boost: +0.05 to similarity score")
    print("- Bosch had no intent signals = no boost")
    print("- Final ranking may change based on intent strength!")


if __name__ == "__main__":
    test_final_scoring()

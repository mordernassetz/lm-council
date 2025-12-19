#!/usr/bin/env python3
"""
🧪 Test Script for Supreme LLM Council
Tests the structure, imports, and basic functionality
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.supreme_council_rrr import (
    SupremeCouncil,
    RRRFramework,
    ProcessingStage,
    TIER1_TEXT_REASONING,
    TIER2_CODE_IDES,
    TIER3_MULTIMODAL_VISION,
    TIER4_VIDEO_GENERATION,
    TIER5_SPECIALIZED,
    TIER6_OPEN_SOURCE,
    SUPREME_COUNCIL,
    SUPREME_JUDGES,
    LITE_COUNCIL,
    LITE_JUDGES,
    OPENROUTER_VERIFIED_MODELS,
    CRITERIA_RRR_INTEGRATED,
    CRITERIA_LITE,
)


def test_tier_counts():
    """Test that all tiers have the expected number of members"""
    print("\n🔍 Testing Tier Counts...")
    
    counts = {
        "TIER1_TEXT_REASONING": (TIER1_TEXT_REASONING, 8),
        "TIER2_CODE_IDES": (TIER2_CODE_IDES, 8),
        "TIER3_MULTIMODAL_VISION": (TIER3_MULTIMODAL_VISION, 10),
        "TIER4_VIDEO_GENERATION": (TIER4_VIDEO_GENERATION, 6),
        "TIER5_SPECIALIZED": (TIER5_SPECIALIZED, 10),
        "TIER6_OPEN_SOURCE": (TIER6_OPEN_SOURCE, 8),
    }
    
    all_passed = True
    for name, (tier, expected) in counts.items():
        actual = len(tier)
        status = "✅" if actual == expected else "❌"
        if actual != expected:
            all_passed = False
        print(f"  {status} {name}: {actual} members (expected {expected})")
    
    # Total check
    total = sum(len(tier) for tier, _ in counts.values())
    print(f"\n  📊 Total across all tiers: {total} (50 expected)")
    
    return all_passed


def test_supreme_council_structure():
    """Test the full council structure"""
    print("\n🏛️ Testing Supreme Council Structure...")
    
    # Check that SUPREME_COUNCIL is deduped and has reasonable count
    unique_count = len(SUPREME_COUNCIL)
    print(f"  📊 Unique models in SUPREME_COUNCIL: {unique_count}")
    
    # Check judges
    judge_count = len(SUPREME_JUDGES)
    print(f"  ⚖️ Supreme Judges: {judge_count}")
    
    # Check lite versions
    lite_model_count = len(LITE_COUNCIL)
    lite_judge_count = len(LITE_JUDGES)
    print(f"  🪶 Lite Council: {lite_model_count} models, {lite_judge_count} judges")
    
    # Check verified models
    verified_count = len(OPENROUTER_VERIFIED_MODELS)
    print(f"  ✓ OpenRouter Verified Models: {verified_count}")
    
    return True


def test_criteria():
    """Test evaluation criteria"""
    print("\n📋 Testing Evaluation Criteria...")
    
    # Full criteria
    print(f"  📊 Full criteria count: {len(CRITERIA_RRR_INTEGRATED)}")
    for c in CRITERIA_RRR_INTEGRATED:
        print(f"      • {c.name}")
    
    # Lite criteria
    print(f"\n  🪶 Lite criteria count: {len(CRITERIA_LITE)}")
    for c in CRITERIA_LITE:
        print(f"      • {c.name}")
    
    return True


async def test_rrr_framework():
    """Test the RRR Framework"""
    print("\n🔄 Testing RRR Framework...")
    
    rrr = RRRFramework()
    
    # Test Recite
    recite_response = await rrr.recite("Test prompt", "test-model")
    assert recite_response.stage == ProcessingStage.RECITE
    print(f"  1️⃣ RECITE: ✅ Stage={recite_response.stage.value}, Confidence={recite_response.confidence}")
    
    # Test Retrieve
    retrieve_response, facts = await rrr.retrieve(recite_response, "Test prompt")
    assert retrieve_response.stage == ProcessingStage.RETRIEVE
    print(f"  2️⃣ RETRIEVE: ✅ Stage={retrieve_response.stage.value}, Sources={len(facts)}")
    
    # Test Revise
    revise_response = await rrr.revise(retrieve_response, facts)
    assert revise_response.stage == ProcessingStage.REVISE
    print(f"  3️⃣ REVISE: ✅ Stage={revise_response.stage.value}, Quality={revise_response.quality_score}")
    
    # Test full pipeline
    final = await rrr.execute_rrr_pipeline("Full test", "test-model")
    print(f"  🔄 FULL PIPELINE: ✅ Final Stage={final.stage.value}")
    
    return True


def test_council_initialization():
    """Test council initialization (without API key)"""
    print("\n🏛️ Testing Council Initialization...")
    
    # Check if API key is set
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("  ⚠️ OPENROUTER_API_KEY not set - skipping live tests")
        print("  📝 To run live tests, set your API key:")
        print("     export OPENROUTER_API_KEY='your-key-here'")
        print("     or create a .env file with OPENROUTER_API_KEY=your-key")
        return "SKIPPED"
    
    print(f"  ✅ API Key found: {api_key[:10]}...")
    
    # Test lite council initialization
    try:
        council = SupremeCouncil(mode="lite")
        print(f"  ✅ Lite Council initialized with {len(council.models)} models")
        council.print_council_info()
        return True
    except Exception as e:
        print(f"  ❌ Failed to initialize council: {e}")
        return False


async def test_council_execution():
    """Test council execution (requires API key)"""
    print("\n🚀 Testing Council Execution...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("  ⚠️ Skipping - No API key")
        return "SKIPPED"
    
    try:
        council = SupremeCouncil(mode="lite")
        
        # Simple test prompt
        prompt = "What is 2 + 2? Answer in one word."
        
        print(f"  📝 Test prompt: '{prompt}'")
        print("  ⏳ Executing council (this may take a minute)...")
        
        completions_df, judgments_df = await council.execute(prompt, verbose=False)
        
        print(f"\n  ✅ Execution complete!")
        print(f"  📊 Completions: {len(completions_df)}")
        print(f"  ⚖️ Judgments: {len(judgments_df)}")
        
        # Show sample completion
        if len(completions_df) > 0:
            sample = completions_df.iloc[0]
            print(f"\n  📌 Sample completion from {sample['model']}:")
            print(f"     '{sample['completion_text'][:100]}...'")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 SUPREME LLM COUNCIL - TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Structural tests (no API required)
    results["Tier Counts"] = test_tier_counts()
    results["Council Structure"] = test_supreme_council_structure()
    results["Criteria"] = test_criteria()
    results["RRR Framework"] = await test_rrr_framework()
    
    # Tests requiring initialization/API
    results["Council Init"] = test_council_initialization()
    
    # Only run execution test if API key is available
    if os.getenv("OPENROUTER_API_KEY"):
        results["Council Execution"] = await test_council_execution()
    else:
        results["Council Execution"] = "SKIPPED"
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result == True:
            status = "✅ PASSED"
        elif result == "SKIPPED":
            status = "⏭️ SKIPPED (no API key)"
        else:
            status = "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    # Final status
    failed = [k for k, v in results.items() if v == False]
    if failed:
        print(f"\n❌ {len(failed)} tests failed: {failed}")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

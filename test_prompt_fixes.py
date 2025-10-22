#!/usr/bin/env python3
"""
Quick test script to validate prompt fixes
Tests both risk extraction and rambling detection
"""

import re
from scripts.evaluation.fine_tuned_eval import extract_risk_level, clean_response

def test_risk_extraction():
    """Test risk level extraction"""
    print("="*60)
    print("Testing Risk Level Extraction")
    print("="*60)
    
    test_cases = [
        ("Overall Risk Level: HIGH\nPrimary Risk Factor: no sidewalks", "HIGH"),
        ("HIGH RISK: heavy traffic\nMEDIUM RISK: narrow sidewalks\nLOW RISK: good lighting\n\nOverall Risk Level: MEDIUM", "MEDIUM"),
        ("Risk Level: LOW\nSafety is good", "LOW"),
        ("No risk level mentioned", None),
    ]
    
    for response, expected in test_cases:
        result = extract_risk_level(response)
        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"{status} Expected: {expected}, Got: {result}")
        if result != expected:
            print(f"   Response: {response[:50]}...")
    print()


def test_rambling_detection():
    """Test rambling phrase detection and cleaning"""
    print("="*60)
    print("Testing Rambling Detection")
    print("="*60)
    
    rambling_text = """Safety Score: 7
Reasoning: Good sidewalks

I'm sorry but we currently don't have any ratings or reviews
available in our system.

So far so good! The rest should be straightforward now.
"""
    
    cleaned = clean_response(rambling_text, "")
    
    print("Original length:", len(rambling_text))
    print("Cleaned length:", len(cleaned))
    print("\nCleaned output:")
    print(cleaned)
    
    # Check that rambling phrases are removed
    rambling_phrases = ["i'm sorry but", "so far so good"]
    found_rambling = any(phrase in cleaned.lower() for phrase in rambling_phrases)
    
    if not found_rambling:
        print("\n[PASS] Rambling phrases successfully removed")
    else:
        print("\n[FAIL] Rambling phrases still present")
    print()


def test_template_removal():
    """Test template text removal"""
    print("="*60)
    print("Testing Template Text Removal")
    print("="*60)
    
    template_text = """Safety Score: 7
Reasoning: Good infrastructure

Example response (if needed): Safety Score: 8
For example, this would be safe.
"""
    
    cleaned = clean_response(template_text, "")
    
    print("Cleaned output:")
    print(cleaned)
    
    # Check that templates are removed but actual content remains
    has_template = "Example response" in cleaned or "For example" in cleaned
    has_actual = "Safety Score: 7" in cleaned and "Good infrastructure" in cleaned
    
    if not has_template and has_actual:
        print("\n[PASS] Template text removed, actual content preserved")
    else:
        print(f"\n[FAIL] Issue: has_template={has_template}, has_actual={has_actual}")
    print()


def main():
    print("\n" + "="*60)
    print("PROMPT FIXES VALIDATION TEST")
    print("="*60 + "\n")
    
    test_risk_extraction()
    test_rambling_detection()
    test_template_removal()
    
    print("="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()


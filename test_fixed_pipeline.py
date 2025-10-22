#!/usr/bin/env python3
"""
Test script for the fixed Qwen-VL processing pipeline
Tests both single image assessment and pairwise comparison
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from qwen_safety_inference_fixed import FixedSafetyAssessmentModel, create_safety_prompts


def test_single_image_assessment():
    """Test single image safety assessment"""
    print("="*60)
    print("TESTING SINGLE IMAGE ASSESSMENT")
    print("="*60)
    
    # Find a sample image
    sample_images_dir = Path("data/samples/Sample SVI")
    if not sample_images_dir.exists():
        print(f"Sample images directory not found: {sample_images_dir}")
        return False
    
    image_files = list(sample_images_dir.glob("*.jpg"))
    if not image_files:
        print("No sample images found")
        return False
    
    test_image = str(image_files[0])
    print(f"Testing with image: {Path(test_image).name}")
    
    # Initialize model (using baseline model for testing)
    try:
        model = FixedSafetyAssessmentModel("Qwen/Qwen2-VL-2B-Instruct", device="cpu")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Test different prompt types
    prompts = create_safety_prompts()
    
    for prompt_type, prompt in prompts.items():
        if prompt_type == "comparison":  # Skip comparison for single image test
            continue
            
        print(f"\nTesting {prompt_type} prompt:")
        print("-" * 40)
        
        try:
            response = model.assess_safety(test_image, prompt)
            print(f"Response: {response}")
            
            # Check if response contains expected elements
            if "Error" in response:
                print(f"[ERROR] Error in {prompt_type} assessment")
                return False
            else:
                print(f"[OK] {prompt_type} assessment successful")
                
        except Exception as e:
            print(f"[ERROR] Exception in {prompt_type} assessment: {e}")
            return False
    
    return True


def test_pairwise_comparison():
    """Test pairwise image comparison"""
    print("\n" + "="*60)
    print("TESTING PAIRWISE COMPARISON")
    print("="*60)
    
    # Find sample images
    sample_images_dir = Path("data/samples/Sample SVI")
    if not sample_images_dir.exists():
        print(f"Sample images directory not found: {sample_images_dir}")
        return False
    
    image_files = list(sample_images_dir.glob("*.jpg"))
    if len(image_files) < 2:
        print("Need at least 2 sample images for comparison")
        return False
    
    test_images = [str(img) for img in image_files[:2]]
    print(f"Testing comparison between: {Path(test_images[0]).name} vs {Path(test_images[1]).name}")
    
    # Initialize model
    try:
        model = FixedSafetyAssessmentModel("Qwen/Qwen2-VL-2B-Instruct", device="cpu")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Test comparison
    comparison_prompt = """Compare these two street view images and determine which one is SAFER for pedestrians.

Consider these factors:
- Sidewalk condition and width
- Traffic density and speed  
- Street lighting quality
- Road maintenance
- Pedestrian-vehicle separation
- Visibility and sight lines
- Overall infrastructure safety

Provide your response as:
SAFER IMAGE: [1 or 2]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation of why one is safer than the other]"""
    
    try:
        response = model.compare_two_images(test_images[0], test_images[1], comparison_prompt)
        print(f"Comparison response: {response}")
        
        # Check if response contains expected elements
        if "Error" in response:
            print("[ERROR] Error in pairwise comparison")
            return False
        else:
            print("[OK] Pairwise comparison successful")
            return True
            
    except Exception as e:
        print(f"[ERROR] Exception in pairwise comparison: {e}")
        return False


def test_processing_pipeline():
    """Test the core processing pipeline components"""
    print("\n" + "="*60)
    print("TESTING PROCESSING PIPELINE COMPONENTS")
    print("="*60)
    
    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        
        # Load processor
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
        print("[OK] Processor loaded successfully")
        
        # Test image processing
        sample_images_dir = Path("data/samples/Sample SVI")
        if sample_images_dir.exists():
            image_files = list(sample_images_dir.glob("*.jpg"))
            if image_files:
                test_image = Image.open(image_files[0]).convert('RGB')
                
                # Test single image processing
                text_with_image = "<|image_pad|>Test prompt"
                inputs = processor(
                    text=text_with_image,
                    images=test_image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                print("[OK] Single image processing successful")
                print(f"   Input keys: {list(inputs.keys())}")
                print(f"   Input IDs shape: {inputs['input_ids'].shape}")
                print(f"   Pixel values shape: {inputs['pixel_values'].shape}")
                
                # Test dual image processing
                if len(image_files) >= 2:
                    test_image2 = Image.open(image_files[1]).convert('RGB')
                    text_with_images = "<|image_pad|><|image_pad|>Test comparison prompt"
                    inputs2 = processor(
                        text=text_with_images,
                        images=[test_image, test_image2],
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    
                    print("[OK] Dual image processing successful")
                    print(f"   Input keys: {list(inputs2.keys())}")
                    print(f"   Input IDs shape: {inputs2['input_ids'].shape}")
                    print(f"   Pixel values shape: {inputs2['pixel_values'].shape}")
                
                return True
            else:
                print("[ERROR] No sample images found for testing")
                return False
        else:
            print("[ERROR] Sample images directory not found")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error testing processing pipeline: {e}")
        return False


def main():
    """Run all tests"""
    print("Testing Fixed Qwen-VL Processing Pipeline")
    print("="*60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Processing pipeline components
    if test_processing_pipeline():
        tests_passed += 1
    
    # Test 2: Single image assessment
    if test_single_image_assessment():
        tests_passed += 1
    
    # Test 3: Pairwise comparison
    if test_pairwise_comparison():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("[SUCCESS] All tests passed! The fixed processing pipeline is working correctly.")
        print("\nYou can now use the fixed scripts:")
        print("- qwen_safety_inference_fixed.py")
        print("- scripts/evaluation/vlm_baseline_evaluation_fixed.py")
        print("- scripts/evaluation/anchor_based_safety_scoring_fixed.py")
    else:
        print("[ERROR] Some tests failed. Please check the error messages above.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

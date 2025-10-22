#!/usr/bin/env python3
"""
Fixed Qwen-VL Baseline Evaluation for Street View Safety Assessment
Run evaluation on unmodified Qwen-VL model without fine-tuning using correct processing pipeline
"""

import os
import json
import torch
import argparse
import itertools
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
import time
from collections import defaultdict, Counter

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2VLForConditionalGeneration
)


class FixedBaselineSafetyEvaluator:
    """Fixed evaluator using unmodified Qwen-VL model with correct processing"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "auto"):
        """
        Initialize the baseline evaluator
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading baseline model: {model_name} on {self.device}")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        self.model.eval()
        print("Baseline model loaded successfully!")
    
    def evaluate_single_image(
        self, 
        image_path: str, 
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Evaluate safety of a single street view image using correct processing
        
        Args:
            image_path: Path to street view image
            prompt: Safety assessment prompt
            max_length: Maximum response length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Safety assessment response
        """
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Create text with image token placeholder
            text_with_image = f"<|image_pad|>{prompt}"
            
            # Process inputs using the correct method
            inputs = self.processor(
                text=text_with_image,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            input_length = len(inputs['input_ids'][0])
            response = self.processor.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    def pairwise_comparison(
        self,
        image_paths: List[str],
        prompt: str = None,
        max_comparisons: int = 40
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform pairwise comparisons by comparing individual numeric safety scores
        """
        # Evaluate each image individually to get a numeric safety score
        scores = {}
        for img in image_paths:
            # Prompt for a numeric score only
            individual_prompt = "Rate this image safety 1-10 (just return the number)."
            resp = self.evaluate_single_image(img, individual_prompt)
            try:
                scores[img] = float(resp)
            except:
                scores[img] = 0.0
        # Now perform pairwise win counts based on numeric scores
        wins = {img: 0 for img in image_paths}
        comparisons = {}
        pairs = list(itertools.combinations(image_paths, 2))
        # Limit number of comparisons
        if len(pairs) > max_comparisons:
            indices = np.random.choice(len(pairs), max_comparisons, replace=False)
            pairs = [pairs[i] for i in indices]
        print(f"Performing {len(pairs)} pairwise comparisons... (via numeric scores)")
        # Count wins
        for img1, img2 in pairs:
            s1, s2 = scores.get(img1, 0.0), scores.get(img2, 0.0)
            winner = img1 if s1 >= s2 else img2
            if winner:
                wins[winner] += 1
            comparisons[f"{Path(img1).stem}_vs_{Path(img2).stem}"] = {"score1": s1, "score2": s2, "winner": Path(winner).stem}
        # Calculate final safety_scores
        total = len(pairs)
        safety_scores = {}
        for img in image_paths:
            win_count = wins[img]
            safety_scores[Path(img).stem] = {
                "image_path": img,
                "wins": win_count,
                "total_comparisons": total,
                "win_percentage": win_count / total if total>0 else 0,
                "safety_score": (win_count/total)*10 if total>0 else 0
            }
        return {"safety_scores": safety_scores, "comparisons": comparisons, "total_pairs": total}
    
    def compare_two_images(self, img1_path: str, img2_path: str, prompt: str) -> str:
        """
        Compare two images side by side using correct processing
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image  
            prompt: Comparison prompt
            
        Returns:
            Comparison response
        """
        try:
            # Load images
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

            # Simplest possible prompt construction for multi-image
            text = f"<|image_pad|><|image_pad|>{prompt}"

            inputs = self.processor(
                text=text,
                images=[img1, img2],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                # Use greedy beam search for concrete safety decisions
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            input_length = len(inputs['input_ids'][0])
            response = self.processor.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Error comparing images: {str(e)}"
    
    def parse_comparison_response(self, response: str, img1_name: str, img2_name: str) -> Optional[str]:
        """
        Parse the comparison response to determine which image won
        
        Args:
            response: Model response text
            img1_name: Name of first image
            img2_name: Name of second image
            
        Returns:
            Name of winning image, or None if unclear
        """
        response_lower = response.lower()
        
        # Look for explicit indicators
        if "safer image: 1" in response_lower or "image 1" in response_lower or "first image" in response_lower:
            return img1_name
        elif "safer image: 2" in response_lower or "image 2" in response_lower or "second image" in response_lower:
            return img2_name
        
        # Look for safer keywords associated with image numbers
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if "safer" in line_lower:
                if any(term in line_lower for term in ["1", "first", img1_name]):
                    return img1_name
                elif any(term in line_lower for term in ["2", "second", img2_name]):
                    return img2_name
        
        # Fallback: pattern matching
        if ("1" in response_lower and "2" in response_lower) or ("one" in response_lower and "two" in response_lower):
            # Both mentioned, need to determine precedence
            # Simple heuristic: whichever appears first after safety-related keywords
            safety_keywords = ["safer", "better", "improved", "preferred"]
            for keyword in safety_keywords:
                if keyword in response_lower:
                    keyword_pos = response_lower.find(keyword)
                    text_after = response_lower[keyword_pos:]
                    
                    if ("1" in text_after and "2" not in text_after[:50]) or ("one" in text_after and "two" not in text_after[:50]):
                        return img1_name
                    elif ("2" in text_after[:-50] and "1" not in text_after[:50]) or ("two" in text_after[:-50] and "one" not in text_after[:50]):
                        return img2_name
        
        return None  # Unable to determine winner


def load_training_data(data_path: str) -> List[Dict]:
    """Load the training data for comparison"""
    try:
        with open(data_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading training data: {e}")
        return []


def compare_with_training_data(
    baseline_results: Dict,
    training_data: List[Dict],
    output_path: str
) -> Dict[str, Any]:
    """
    Compare baseline VLM results with training data annotations
    
    Args:
        baseline_results: Results from baseline evaluation
        training_data: Training data with expected responses
        output_path: Path to save comparison results
        
    Returns:
        Comparison analysis results
    """
    print("Comparing baseline results with training data...")
    
    # Extract safety scores from training data
    training_scores = {}
    
    for item in training_data:
        image_id = Path(item['image_path']).stem
        
        # Parse expected response to extract safety score
        if item['task_type'] == 'safety_score':
            expected_response = item['expected_response']
            try:
                # Extract score from "Safety Score: X"
                if "Safety Score:" in expected_response:
                    score_line = [line for line in expected_response.split('\n') 
                                if "Safety Score:" in line]
                    if score_line:
                        score_text = score_line[0].split("Safety Score:")[1].strip()
                        score = int(score_text.split()[0])
                        training_scores[image_id] = score
            except:
                continue
        elif item['task_type'] == 'detailed_analysis':
            # Extract overall score from detailed analysis
            expected_response = item['expected_response']
            try:
                if "Overall Score:" in expected_response:
                    score_line = [line for line in expected_response.split('\n') 
                                if "Overall Score:" in line]
                    if score_line:
                        score_part = score_line[0].split("Overall Score:")[1].strip()
                        # Extract numerator (e.g., "35/50" -> 35)
                        numerator = int(score_part.split('/')[0])
                        # Convert to 0-10 scale (50 total -> 10 max)
                        score = (numerator / 50) * 10
                        training_scores[image_id] = score
            except:
                continue
    
    # Match with baseline results
    comparison_results = {
        "matched_pairs": [],
        "baseline_only": [],
        "training_only": [],
        "statistics": {}
    }
    
    # Find matches
    baseline_images = set(baseline_results["safety_scores"].keys())
    training_images = set(training_scores.keys())
    matched_images = baseline_images.intersection(training_images)
    
    matched_data = []
    for image_id in matched_images:
        baseline_score = baseline_results["safety_scores"][image_id]["safety_score"]
        training_score = training_scores[image_id]
        
        matched_data.append({
            "image_id": image_id,
            "baseline_score": baseline_score,
            "training_score": training_score,
            "difference": abs(baseline_score - training_score),
            "relative_error": abs(baseline_score - training_score) / max(training_score, 0.1)
        })
        
        comparison_results["matched_pairs"].append({
            "image_id": image_id,
            "baseline_score": baseline_score,
            "training_score": training_score,
            "difference": abs(baseline_score - training_score),
            "relative_error": abs(baseline_score - training_score) / max(training_score, 0.1)
        })
    
    # Calculate statistics
    if matched_data:
        differences = [item["difference"] for item in matched_data]
        relative_errors = [item["relative_error"] for item in matched_data]
        
        comparison_results["statistics"] = {
            "num_matched": len(matched_data),
            "mean_absolute_error": np.mean(differences),
            "median_absolute_error": np.median(differences),
            "max_difference": np.max(differences),
            "mean_relative_error": np.mean(relative_errors),
            "correlation_coefficient": np.corrcoef(
                [item["baseline_score"] for item in matched_data],
                [item["training_score"] for item in matched_data]
            )[0, 1] if len(matched_data) > 1 else 0
        }
    
    # Baseline-only images
    comparison_results["baseline_only"] = list(baseline_images - training_images)
    
    # Training-only images  
    comparison_results["training_only"] = list(training_images - baseline_images)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Fixed Qwen-VL Baseline Safety Evaluation")
    parser.add_argument("--image_folder", type=str, required=True, 
                       help="Folder containing street view images")
    parser.add_argument("--training_data", type=str, 
                       default="vlm_safety_training_data.json",
                       help="Path to training data JSON file")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Directory to save results")
    parser.add_argument("--model_name", type=str, 
                       default="Qwen/Qwen2-VL-2B-Instruct",
                       help="Qwen-VL model name")
    parser.add_argument("--max_comparisons", type=int, default=40,
                       help="Maximum pairwise comparisons")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get image paths
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(Path(args.image_folder).glob(ext))
        image_paths.extend(Path(args.image_folder).glob(ext.upper()))
    image_paths = [str(p) for p in sorted(image_paths)]
    
    if not image_paths:
        print(f"No images found in {args.image_folder}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Initialize evaluator
    evaluator = FixedBaselineSafetyEvaluator(args.model_name, args.device)
    
    # Run pairwise comparison evaluation
    print("Starting pairwise comparison evaluation...")
    baseline_results = evaluator.pairwise_comparison(
        image_paths, 
        max_comparisons=args.max_comparisons
    )
    
    # Save baseline results
    baseline_output_path = os.path.join(args.output_dir, "baseline_pairwise_results.json")
    with open(baseline_output_path, 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    print(f"Baseline results saved to {baseline_output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE EVALUATION SUMMARY")
    print("="*60)
    
    safety_scores = baseline_results["safety_scores"]
    scores = [data["safety_score"] for data in safety_scores.values()]
    
    print(f"Images evaluated: {len(safety_scores)}")
    print(f"Pairwise comparisons: {baseline_results['total_pairs']}")
    print(f"Average safety score: {np.mean(scores):.2f}")
    print(f"Score range: {np.min(scores):.2f} - {np.max(scores):.2f}")
    print(f"Standard deviation: {np.std(scores):.2f}")
    
    # Sort by safety score
    sorted_scores = sorted(safety_scores.items(), 
                          key=lambda x: x[1]["safety_score"], 
                          reverse=True)
    
    print("\nTop 5 safest images:")
    for i, (image_id, data) in enumerate(sorted_scores[:5]):
        print(f"{i+1}. {image_id}: {data['safety_score']:.2f} (wins: {data['wins']}/{data['total_comparisons']})")
    
    print("\nBottom 5 least safe images:")
    for i, (image_id, data) in enumerate(sorted_scores[-5:]):
        print(f"{len(sorted_scores)-4+i}. {image_id}: {data['safety_score']:.2f} (wins: {data['wins']}/{data['total_comparisons']})")
    
    # Compare with training data if available
    if os.path.exists(args.training_data):
        training_data = load_training_data(args.training_data)
        comparison_path = os.path.join(args.output_dir, "baseline_vs_training_comparison.json")
        
        print(f"\nComparing with training data from {args.training_data}...")
        comparison_results = compare_with_training_data(
            baseline_results, 
            training_data, 
            comparison_path
        )
        
        if comparison_results["statistics"]:
            stats = comparison_results["statistics"]
            print(f"\nComparison with training data:")
            print(f"Matched images: {stats['num_matched']}")
            print(f"Mean absolute error: {stats['mean_absolute_error']:.2f}")
            print(f"Correlation coefficient: {stats['correlation_coefficient']:.3f}")
            print(f"Comparison results saved to {comparison_path}")
    else:
        print(f"\nTraining data file {args.training_data} not found - skipping comparison")


if __name__ == "__main__":
    main()

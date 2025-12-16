#!/usr/bin/env python3
"""
Comprehensive comparison pipeline for GPT-4, GPT-4o-mini (fine-tuned), and Qwen2-VL (fine-tuned)
Evaluates all models on the same test set and compares results
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm

from unified_model_inference import ModelFactory


class SafetyScoreExtractor:
    """Extract safety scores from model responses"""
    
    @staticmethod
    def extract_score(response: str, task_type: str = "safety_score") -> Optional[float]:
        """
        Extract safety score from response
        
        Args:
            response: Model response text
            task_type: Type of task (safety_score, binary_classification, detailed_analysis)
            
        Returns:
            Extracted score or None
        """
        if task_type == "safety_score":
            # Look for the new format: "Score: X/10"
            patterns = [
                r'Score:\s*(\d+(?:\.\d+)?)\s*/\s*10',  # New format: Score: X/10
                r'Score:\s*(\d+(?:\.\d+)?)',            # Score: X
                r'Safety Score:\s*(\d+(?:\.\d+)?)',     # Safety Score: X
                r'(\d+(?:\.\d+)?)\s*/\s*10',            # X/10
                r'(\d+(?:\.\d+)?)\s+out of 10',         # X out of 10
                r'rate.*?(\d+(?:\.\d+)?)\s*/\s*10',
                r'score.*?(\d+(?:\.\d+)?)\s*/\s*10',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    # Ensure score is in valid range
                    if 0 <= score <= 10:
                        return score
            
            # Fallback: look for any number followed by /10 or "out of 10"
            numbers = re.findall(r'(\d+(?:\.\d+)?)', response)
            for num in numbers:
                score = float(num)
                if 0 <= score <= 10:
                    return score
            
            return None
        
        elif task_type == "binary_classification":
            # Look for SAFE or UNSAFE
            response_upper = response.upper()
            if "UNSAFE" in response_upper:
                return 0  # Unsafe
            elif "SAFE" in response_upper:
                return 1  # Safe
            return None
        
        elif task_type == "detailed_analysis":
            # Extract overall score from detailed analysis
            match = re.search(r'Overall Score:\s*(\d+(?:\.\d+)?)\s*/\s*50', response, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                if 0 <= score <= 50:
                    return score / 5  # Normalize to 0-10 scale
            return None
        
        return None


class ModelComparator:
    """Compare multiple models on safety assessment task"""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize comparator
        
        Args:
            output_dir: Directory to save results
        """
        self.project_root = Path(__file__).parent.parent.parent
        
        if output_dir is None:
            output_dir = self.project_root / "results" / "model_comparison"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.score_extractor = SafetyScoreExtractor()
    
    def load_test_data(self) -> List[Dict]:
        """
        Load test data (ground truth) - ONLY safety_score tasks
        
        Returns:
            List of test examples
        """
        test_file = self.project_root / "configs" / "vlm_safety_training_data.json"
        
        print(f"Loading test data from: {test_file}")
        
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        # Filter to ONLY safety_score tasks (exclude binary classification)
        data = [item for item in data if item.get('task_type') == 'safety_score']
        
        # Use a subset for testing (e.g., every 3rd item for test set)
        test_data = [item for i, item in enumerate(data) if i % 3 == 0]
        
        print(f"Loaded {len(test_data)} test examples (safety_score only)")
        
        return test_data
    
    def evaluate_model(
        self, 
        model_type: str, 
        test_data: List[Dict],
        model_kwargs: Dict = None
    ) -> List[Dict]:
        """
        Evaluate a single model on test data
        
        Args:
            model_type: Type of model to evaluate
            test_data: List of test examples
            model_kwargs: Additional arguments for model initialization
            
        Returns:
            List of results
        """
        print(f"\n{'=' * 60}")
        print(f"EVALUATING: {model_type}")
        print(f"{'=' * 60}")
        
        if model_kwargs is None:
            model_kwargs = {}
        
        # Initialize model
        try:
            model = ModelFactory.create_model(model_type, **model_kwargs)
        except Exception as e:
            print(f"[ERROR] Failed to initialize {model_type}: {e}")
            return []
        
        results = []
        errors = 0
        
        # Evaluate each example
        for item in tqdm(test_data, desc=f"Evaluating {model_type}"):
            try:
                image_path = item.get('image_path')
                prompt = item.get('prompt')
                expected_response = item.get('expected_response')
                task_type = item.get('task_type', 'safety_score')
                
                if not image_path or not Path(image_path).exists():
                    print(f"[WARNING] Skipping: image not found - {image_path}")
                    continue
                
                # Get model prediction
                response = model.predict(image_path, prompt)
                
                # Extract score from response
                predicted_score = self.score_extractor.extract_score(response, task_type)
                expected_score = self.score_extractor.extract_score(expected_response, task_type)
                
                result = {
                    'image_path': image_path,
                    'prompt': prompt,
                    'task_type': task_type,
                    'expected_response': expected_response,
                    'expected_score': expected_score,
                    'model_response': response,
                    'predicted_score': predicted_score,
                    'model_name': model.get_model_name()
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"[ERROR] Error evaluating image {image_path}: {e}")
                errors += 1
                continue
        
        print(f"\n[OK] Evaluation complete!")
        print(f"   - Processed: {len(results)} examples")
        print(f"   - Errors: {errors}")
        
        return results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary of metrics
        """
        # Filter results with valid scores
        valid_results = [
            r for r in results 
            if r.get('expected_score') is not None and r.get('predicted_score') is not None
        ]
        
        if not valid_results:
            return {
                'total_examples': len(results),
                'valid_examples': 0,
                'mae': None,
                'rmse': None,
                'accuracy_1point': None,
                'accuracy_2point': None
            }
        
        # Calculate errors
        errors = [
            abs(r['predicted_score'] - r['expected_score']) 
            for r in valid_results
        ]
        
        # Mean Absolute Error
        mae = sum(errors) / len(errors)
        
        # Root Mean Square Error
        rmse = (sum([e**2 for e in errors]) / len(errors)) ** 0.5
        
        # Accuracy within 1 point
        accuracy_1point = sum([1 for e in errors if e <= 1]) / len(errors)
        
        # Accuracy within 2 points
        accuracy_2point = sum([1 for e in errors if e <= 2]) / len(errors)
        
        metrics = {
            'total_examples': len(results),
            'valid_examples': len(valid_results),
            'mae': mae,
            'rmse': rmse,
            'accuracy_1point': accuracy_1point,
            'accuracy_2point': accuracy_2point,
        }
        
        return metrics
    
    def compare_models(
        self,
        models_to_compare: List[Tuple[str, Dict]] = None
    ) -> Dict:
        """
        Compare multiple models
        
        Args:
            models_to_compare: List of (model_type, model_kwargs) tuples
            
        Returns:
            Dictionary with all comparison results
        """
        if models_to_compare is None:
            models_to_compare = [
                ("gpt4o-mini-ft", {}),
                ("qwen2vl-ft", {}),
            ]
        
        print("=" * 60)
        print("MODEL COMPARISON PIPELINE")
        print("=" * 60)
        print(f"\nModels to compare: {[m[0] for m in models_to_compare]}")
        
        # Load test data
        test_data = self.load_test_data()
        
        # Evaluate each model
        all_results = {}
        all_metrics = {}
        
        for model_type, model_kwargs in models_to_compare:
            try:
                results = self.evaluate_model(model_type, test_data, model_kwargs)
                metrics = self.calculate_metrics(results)
                
                all_results[model_type] = results
                all_metrics[model_type] = metrics
                
                # Save individual model results
                self.save_model_results(model_type, results, metrics)
                
            except Exception as e:
                print(f"[ERROR] Failed to evaluate {model_type}: {e}")
                continue
        
        # Create comparison report
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'models': list(all_results.keys()),
            'test_size': len(test_data),
            'results': all_results,
            'metrics': all_metrics
        }
        
        # Save comparison results
        self.save_comparison_results(comparison)
        
        # Print summary
        self.print_comparison_summary(all_metrics)
        
        return comparison
    
    def save_model_results(self, model_type: str, results: List[Dict], metrics: Dict):
        """Save results for a single model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results (JSON)
        json_file = self.output_dir / f"{model_type}_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'model': model_type,
                'metrics': metrics,
                'results': results
            }, f, indent=2)
        
        print(f"[SAVED] {model_type} results to: {json_file}")
        
        # Save as Excel for easy viewing
        try:
            df = pd.DataFrame(results)
            excel_file = self.output_dir / f"{model_type}_results_{timestamp}.xlsx"
            df.to_excel(excel_file, index=False)
            print(f"[SAVED] {model_type} results to: {excel_file}")
        except Exception as e:
            print(f"[WARNING] Could not save Excel file: {e}")
    
    def save_comparison_results(self, comparison: Dict):
        """Save comparison results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save full comparison (JSON)
        json_file = self.output_dir / f"model_comparison_{timestamp}.json"
        
        # Remove full results to avoid huge file (save separately)
        comparison_summary = comparison.copy()
        comparison_summary['results'] = {
            model: f"See {model}_results_{timestamp}.json"
            for model in comparison['results'].keys()
        }
        
        with open(json_file, 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        
        print(f"\n[SAVED] Comparison summary to: {json_file}")
        
        # Save metrics comparison as Excel
        try:
            metrics_df = pd.DataFrame(comparison['metrics']).T
            excel_file = self.output_dir / f"metrics_comparison_{timestamp}.xlsx"
            metrics_df.to_excel(excel_file)
            print(f"[SAVED] Metrics comparison to: {excel_file}")
        except Exception as e:
            print(f"[WARNING] Could not save metrics Excel: {e}")
    
    def print_comparison_summary(self, all_metrics: Dict):
        """Print comparison summary to console"""
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        # Create comparison table
        print(f"\n{'Model':<25} {'MAE':<10} {'RMSE':<10} {'Acc@1':<10} {'Acc@2':<10}")
        print("-" * 65)
        
        for model, metrics in all_metrics.items():
            mae = f"{metrics['mae']:.3f}" if metrics['mae'] is not None else "N/A"
            rmse = f"{metrics['rmse']:.3f}" if metrics['rmse'] is not None else "N/A"
            acc1 = f"{metrics['accuracy_1point']:.1%}" if metrics['accuracy_1point'] is not None else "N/A"
            acc2 = f"{metrics['accuracy_2point']:.1%}" if metrics['accuracy_2point'] is not None else "N/A"
            
            print(f"{model:<25} {mae:<10} {rmse:<10} {acc1:<10} {acc2:<10}")
        
        print("\n" + "=" * 60)
        print("Legend:")
        print("  MAE: Mean Absolute Error (lower is better)")
        print("  RMSE: Root Mean Square Error (lower is better)")
        print("  Acc@1: Accuracy within 1 point (higher is better)")
        print("  Acc@2: Accuracy within 2 points (higher is better)")
        print("=" * 60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Compare all models for safety assessment")
    parser.add_argument("--models", nargs="+", default=["gpt4o-mini-ft", "qwen2vl-ft"],
                        help="Models to compare")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--gpt4-model", type=str, default="gpt-4",
                        help="GPT-4 model variant to use (optional, for baseline comparison)")
    
    args = parser.parse_args()
    
    # Prepare model configurations
    models_to_compare = []
    
    for model in args.models:
        if model == "gpt4":
            models_to_compare.append(("gpt4", {"model": args.gpt4_model}))
        elif model == "gpt4o-mini-ft":
            models_to_compare.append(("gpt4o-mini-ft", {}))
        elif model == "qwen2vl-ft":
            models_to_compare.append(("qwen2vl-ft", {}))
        else:
            print(f"[WARNING] Unknown model: {model}")
    
    # Run comparison
    comparator = ModelComparator(output_dir=args.output_dir)
    comparator.compare_models(models_to_compare)
    
    print("\n[OK] Comparison complete!")


if __name__ == "__main__":
    main()



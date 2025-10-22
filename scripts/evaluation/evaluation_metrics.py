#!/usr/bin/env python3
"""
Evaluation Metrics for Visual Safety Assessment
Comprehensive evaluation comparing VLM results with ground truth annotations
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
from collections import defaultdict
from dataclasses import dataclass
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    cohen_kappa_score
)
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import re


@dataclass
class EvaluationResult:
    """Container for evaluation metrics"""
    image_id: str
    predicted_score: Optional[float]
    ground_truth_score: Optional[float]
    predicted_classification: Optional[str]
    ground_truth_classification: Optional[str]
    confidence: Optional[str]
    task_type: str


class SafetyEvaluatorMetrics:
    """
    Comprehensive evaluation metrics for safety assessment
    
    Supports evaluation of:
    - Continuous safety scores (0-10)
    - Binary safety classifications (SAFE/UNSAFE)
    - Detailed multi-dimensional analyses
    - Risk assessments
    """
    
    def __init__(self):
        self.evaluation_results = []
        self.task_types = {
            'safety_score': 'continuous',
            'binary_classification': 'categorical',
            'detailed_analysis': 'multi_dimensional',
            'risk_assessment': 'categorical'
        }
    
    def parse_ground_truth_data(self, training_data: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """
        Parse training data annotations into structured format
        
        Args:
            training_data: List of training examples
            
        Returns:
            Dictionary mapping image_id to parsed annotations
        """
        parsed_data = defaultdict(lambda: {
            'safety_score': None,
            'binary_classification': None,
            'detailed_scores': {},
            'risk_level': None,
            'task_types': []
        })
        
        for item in training_data:
            image_id = Path(item['image_path']).stem
            task_type = item['task_type']
            expected_response = item['expected_response']
            
            parsed_data[image_id]['task_types'].append(task_type)
            
            try:
                if task_type == 'safety_score':
                    # Extract safety score from "Safety Score: X"
                    score_match = re.search(r'Safety Score:\s*(\d+)', expected_response)
                    if score_match:
                        score = int(score_match.group(1))
                        parsed_data[image_id]['safety_score'] = score
                
                elif task_type == 'binary_classification':
                    # Extract SAFE/UNSAFE classification
                    if 'Classification: SAFE' in expected_response:
                        parsed_data[image_id]['binary_classification'] = 'SAFE'
                    elif 'Classification: UNSAFE' in expected_response:
                        parsed_data[image_id]['binary_classification'] = 'UNSAFE'
                
                elif task_type == 'detailed_analysis':
                    # Extract individual component scores
                    scores = {}
                    areas = ['Pedestrian Safety', 'Traffic Safety', 'Lighting Safety', 
                            'Infrastructure Safety', 'Crime Safety']
                    
                    for area in areas:
                        pattern = f'{area}:\s*(\d+)'
                        match = re.search(pattern, expected_response)
                        if match:
                            scores[area.lower().replace(' ', '_')] = int(match.group(1))
                    
                    parsed_data[image_id]['detailed_scores'] = scores
                    
                    # Extract overall score
                    overall_match = re.search(r'Overall Score:\s*(\d+)/50', expected_response)
                    if overall_match:
                        overall_score = int(overall_match.group(1))
                        # Convert to 0-10 scale
                        parsed_data[image_id]['safety_score'] = (overall_score / 50) * 10
                
                elif task_type == 'risk_assessment':
                    # Extract risk level (LOW/MEDIUM/HIGH)
                    risk_match = re.search(r'Overall Risk Level:\s*(LOW|MEDIUM|HIGH)', 
                                        expected_response)
                    if risk_match:
                        parsed_data[image_id]['risk_level'] = risk_match.group(1)
                        
                        # Convert risk level to approximate safety score
                        risk_mapping = {'LOW': 7.5, 'MEDIUM': 5.0, 'HIGH': 2.5}
                        parsed_data[image_id]['safety_score'] = risk_mapping.get(risk_match.group(1))
            
            except Exception as e:
                print(f"Error parsing ground truth for {image_id}, {task_type}: {e}")
                continue
        
        return dict(parsed_data)
    
    def parse_vlm_predictions(self, vlm_results: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """
        Parse VLM model predictions into structured format
        
        Args:
            vlm_results: Results from VLM inference
            
        Returns:
            Dictionary mapping image_id to parsed predictions
        """
        parsed_predictions = defaultdict(lambda: {
            'safety_score': None,
            'binary_classification': None,
            'detailed_scores': {},
            'risk_level': None,
            'confidence': None,
            'raw_response': None
        })
        
        for result in vlm_results:
            image_id = Path(result['image_path']).stem if isinstance(result['image_path'], str) else Path(result['image_path']).stem
            response = result.get('response', '')
            
            parsed_predictions[image_id]['raw_response'] = response
            
            try:
                # Extract safety score from various formats
                score_patterns = [
                    r'Safety Score:\s*(\d+(?:\.\d+)?)',
                    r'score.*?(\d+(?:\.\d+)?)',
                    r'(\d+(?:\.\d+)?)/10'
                ]
                
                for pattern in score_patterns:
                    score_match = re.search(pattern, response, re.IGNORECASE)
                    if score_match:
                        score = float(score_match.group(1))
                        # Ensure score is in 0-10 range
                        score = max(0, min(10, score))
                        parsed_predictions[image_id]['safety_score'] = score
                        break
                
                # Extract binary classification
                if any(word in response.upper() for word in ['SAFE', 'UNSAFE']):
                    if 'SAFE' in response.upper() and 'UNSAFE' not in response.upper():
                        parsed_predictions[image_id]['binary_classification'] = 'SAFE'
                    elif 'UNSAFE' in response.upper():
                        parsed_predictions[image_id]['binary_classification'] = 'UNSAFE'
                
                # Extract confidence
                confidence_pattern = r'Confidence:\s*(HIGH|MEDIUM|LOW)'
                conf_match = re.search(confidence_pattern, response, re.IGNORECASE)
                if conf_match:
                    parsed_predictions[image_id]['confidence'] = conf_match.group(1).upper()
                
                # Extract detailed scores if present
                detailed_areas = ['pedestrian safety', 'traffic safety', 'lighting safety',
                               'infrastructure safety', 'crime safety']
                
                for area in detailed_areas:
                    pattern = f'{area}:\\s*(\\d+(?:\\.\\d+)?)'
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        parsed_predictions[image_id]['detailed_scores'][area] = float(match.group(1))
            
            except Exception as e:
                print(f"Error parsing VLM prediction for {image_id}: {e}")
                continue
        
        return dict(parsed_predictions)
    
    def match_comparison_data(self, ground_truth_data: Dict, vlm_predictions: Dict) -> List[EvaluationResult]:
        """
        Match ground truth with VLM predictions
        
        Args:
            ground_truth_data: Parsed ground truth annotations
            vlm_predictions: Parsed VLM predictions
            
        Returns:
            List of matched evaluation results
        """
        matched_results = []
        
        # Find common image IDs
        common_images = set(ground_truth_data.keys()) & set(vlm_predictions.keys())
        
        for image_id in common_images:
            gt_data = ground_truth_data[image_id]
            pred_data = vlm_predictions[image_id]
            
            # Create evaluation result for each task type present in ground truth
            for task_type in gt_data['task_types']:
                eval_result = EvaluationResult(
                    image_id=image_id,
                    predicted_score=pred_data.get('safety_score'),
                    ground_truth_score=gt_data.get('safety_score'),
                    predicted_classification=self._get_classification(pred_data, task_type),
                    ground_truth_classification=self._get_classification(gt_data, task_type),
                    confidence=pred_data.get('confidence'),
                    task_type=task_type
                )
                
                matched_results.append(eval_result)
        
        self.evaluation_results = matched_results
        return matched_results
    
    def _get_classification(self, data: Dict, task_type: str) -> Optional[str]:
        """Extract classification based on task type"""
        if task_type == 'binary_classification':
            return data.get('binary_classification')
        elif task_type == 'risk_assessment':
            return data.get('risk_level')
        return None
    
    def calculate_regression_metrics(self) -> Dict[str, float]:
        """
        Calculate regression metrics for continuous safety scores
        
        Returns:
            Dictionary of regression metrics
        """
        # Filter for continuous score predictions
        continuous_results = [
            r for r in self.evaluation_results 
            if r.task_type in ['safety_score', 'detailed_analysis'] 
            and r.predicted_score is not None 
            and r.ground_truth_score is not None
        ]
        
        if len(continuous_results) < 2:
            return {"error": "Insufficient continuous score data for regression metrics"}
        
        predicted_scores = [r.predicted_score for r in continuous_results]
        true_scores = [r.ground_truth_score for r in continuous_results]
        
        metrics = {
            'mean_absolute_error': mean_absolute_error(true_scores, predicted_scores),
            'mean_squared_error': mean_squared_error(true_scores, predicted_scores),
            'root_mean_squared_error': np.sqrt(mean_squared_error(true_scores, predicted_scores)),
            'r2_score': r2_score(true_scores, predicted_scores),
            'pearson_correlation': pearsonr(true_scores, predicted_scores)[0],
            'spearman_correlation': spearmanr(true_scores, predicted_scores)[0],
            'kendall_tau': kendalltau(true_scores, predicted_scores)[0],
            'num_samples': len(continuous_results)
        }
        
        # Calculate percentage within different thresholds
        errors = np.abs(np.array(predicted_scores) - np.array(true_scores))
        
        thresholds = [0.5, 1.0, 1.5, 2.0]
        for threshold in thresholds:
            within_threshold = np.sum(errors <= threshold) / len(errors)
            metrics[f'within_{threshold}_points'] = within_threshold
        
        return metrics
    
    def calculate_classification_metrics(self) -> Dict[str, Any]:
        """
        Calculate classification metrics for binary and categorical predictions
        
        Returns:
            Dictionary of classification metrics
        """
        # Filter for classification predictions
        classification_results = [
            r for r in self.evaluation_results 
            if r.task_type in ['binary_classification', 'risk_assessment']
            and r.predicted_classification is not None 
            and r.ground_truth_classification is not None
        ]
        
        if len(classification_results) == 0:
            return {"error": "No classification data available"}
        
        predicted_classes = [r.predicted_classification for r in classification_results]
        true_classes = [r.ground_truth_classification for r in classification_results]
        
        # Calculate overall accuracy
        accuracy = accuracy_score(true_classes, predicted_classes)
        
        # Calculate Cohen's kappa for inter-rater reliability
        kappa = cohen_kappa_score(true_classes, predicted_classes)
        
        # Generate classification report
        try:
            report = classification_report(true_classes, predicted_classes, 
                                        output_dict=True, zero_division=0)
        except Exception as e:
            report = {"error": f"Could not generate classification report: {e}"}
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        metrics = {
            'accuracy': accuracy,
            'cohen_kappa': kappa,
            'num_samples': len(classification_results),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'classes': sorted(set(true_classes + predicted_classes))
        }
        
        # Calculate per-class metrics for binary classification
        if len(set(true_classes)) == 2:  # Binary classification
            from sklearn.metrics import precision_recall_fscore_support
            
            precision, recall, f1, support = precision_recall_fscore_support(
                true_classes, predicted_classes, average=None, zero_division=0
            )
            
            class_names = sorted(set(true_classes))
            metrics['per_class_metrics'] = {
                class_name: {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1_score': f1[i],
                    'support': support[i]
                } for i, class_name in enumerate(class_names)
            }
        
        return metrics
    
    def calculate_detailed_analysis_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics for detailed multi-dimensional analysis
        
        Returns:
            Dictionary of detailed analysis metrics
        """
        # For now, treat detailed analysis similar to regression
        return self.calculate_regression_metrics()
    
    def generate_evaluation_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Complete evaluation report
        """
        if not self.evaluation_results:
            report = {"error": "No evaluation results available"}
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
            return report
        
        # Calculate overall statistics
        total_results = len(self.evaluation_results)
        task_type_counts = defaultdict(int)
        confidence_distribution = defaultdict(int)
        
        for result in self.evaluation_results:
            task_type_counts[result.task_type] += 1
            if result.confidence:
                confidence_distribution[result.confidence] += 1
        
        # Calculate metrics by task type
        regression_metrics = self.calculate_regression_metrics()
        classification_metrics = self.calculate_classification_metrics()
        
        # Check for errors
        regression_error = isinstance(regression_metrics, dict) and "error" in regression_metrics
        classification_error = isinstance(classification_metrics, dict) and "error" in classification_metrics
        
        report = {
            "evaluation_summary": {
                "total_evaluations": total_results,
                "task_type_breakdown": dict(task_type_counts),
                "confidence_distribution": dict(confidence_distribution)
            },
            "regression_metrics": regression_metrics if not regression_error else None,
            "classification_metrics": classification_metrics if not classification_error else None,
            "detailed_analysis": [
                {
                    "image_id": r.image_id,
                    "task_type": r.task_type,
                    "predicted_score": r.predicted_score,
                    "ground_truth_score": r.ground_truth_score,
                    "predicted_classification": r.predicted_classification,
                    "ground_truth_classification": r.ground_truth_classification,
                    "confidence": r.confidence
                } for r in self.evaluation_results
            ]
        }
        
        # Add overall performance summary
        if not regression_error and regression_metrics:
            report["performance_summary"] = {
                "mean_absolute_error": regression_metrics.get('mean_absolute_error'),
                "pearson_correlation": regression_metrics.get('pearson_correlation'),
                "r2_score": regression_metrics.get('r2_score')
            }
        
        if not classification_error and classification_metrics:
            report["performance_summary"] = report.get("performance_summary", {})
            report["performance_summary"].update({
                "classification_accuracy": classification_metrics.get('accuracy'),
                "cohen_kappa": classification_metrics.get('cohen_kappa')
            })
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def plot_evaluation_results(self, output_dir: str = "./plots") -> List[str]:
        """
        Generate visualization plots for evaluation results
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            List of generated plot file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        plot_files = []
        
        # Plot 1: Regression scatter plot
        continuous_results = [
            r for r in self.evaluation_results 
            if r.predicted_score is not None and r.ground_truth_score is not None
        ]
        
        if len(continuous_results) >= 2:
            plt.figure(figsize=(8, 6))
            predicted_scores = [r.predicted_score for r in continuous_results]
            true_scores = [r.ground_truth_score for r in continuous_results]
            
            plt.scatter(true_scores, predicted_scores, alpha=0.7)
            plt.plot([0, 10], [0, 10], 'r--', label='Perfect Agreement')
            
            # Add regression line
            z = np.polyfit(true_scores, predicted_scores, 1)
            p = np.poly1d(z)
            plt.plot(true_scores, p(true_scores), 'g-', alpha=0.8, label='Best Fit')
            
            plt.xlabel('Ground Truth Safety Score')
            plt.ylabel('Predicted Safety Score')
            plt.title('Safety Score Prediction vs Ground Truth')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            scatter_path = os.path.join(output_dir, "safety_score_scatter.png")
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(scatter_path)
        
        # Plot 2: Error distribution
        if len(continuous_results) >= 2:
            predicted_scores = [r.predicted_score for r in continuous_results]
            true_scores = [r.ground_truth_score for r in continuous_results]
            errors = np.array(predicted_scores) - np.array(true_scores)
            
            plt.figure(figsize=(8, 6))
            plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Prediction Error (Predicted - Ground Truth)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Safety Score Prediction Errors')
            plt.axvline(x=0, color='r', linestyle='--', label='Perfect Prediction')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            error_path = os.path.join(output_dir, "error_distribution.png")
            plt.savefig(error_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(error_path)
        
        return plot_files


def main():
    """Example usage of the evaluation metrics"""
    parser = argparse.ArgumentParser(description="Evaluate VLM Safety Assessment")
    parser.add_argument("--training_data", type=str, required=True,
                        help="Path to training data JSON")
    parser.add_argument("--vlm_results", type=str, required=True,
                        help="Path to VLM evaluation results JSON")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Load data
    with open(args.training_data, 'r') as f:
        training_data = json.load(f)
    
    with open(args.vlm_results, 'r') as f:
        vlm_results = json.load(f)
    
    # Initialize evaluator
    evaluator = SafetyEvaluatorMetrics()
    
    # Parse data
    ground_truth = evaluator.parse_ground_truth_data(training_data)
    vlm_predictions = evaluator.parse_vlm_predictions(vlm_results)
    
    # Match and evaluate
    matched_results = evaluator.match_comparison_data(ground_truth, vlm_predictions)
    
    # Generate report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    report = evaluator.generate_evaluation_report(report_path)
    
    # Generate plots
    plot_files = evaluator.plot_evaluation_results(args.output_dir)
    
    # Print summary
    print("Evaluation completed!")
    print(f"Results saved to {report_path}")
    print(f"Plots saved to {len(plot_files)} files in {args.output_dir}")
    
    if "performance_summary" in report:
        print("\nPerformance Summary:")
        for metric, value in report["performance_summary"].items():
            print(f"{metric}: {value:.3f}" if isinstance(value, float) else f"{metric}: {value}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze and visualize model comparison results
Creates plots and detailed analysis reports
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ComparisonAnalyzer:
    """Analyze and visualize model comparison results"""
    
    def __init__(self, results_dir: Path = None):
        """
        Initialize analyzer
        
        Args:
            results_dir: Directory containing comparison results
        """
        self.project_root = Path(__file__).parent.parent.parent
        
        if results_dir is None:
            results_dir = self.project_root / "results" / "model_comparison"
        
        self.results_dir = Path(results_dir)
        
        if not self.results_dir.exists():
            raise ValueError(f"Results directory not found: {self.results_dir}")
    
    def load_latest_comparison(self) -> Dict:
        """
        Load the most recent comparison results
        
        Returns:
            Comparison results dictionary
        """
        # Find most recent comparison file
        comparison_files = sorted(self.results_dir.glob("model_comparison_*.json"))
        
        if not comparison_files:
            raise ValueError("No comparison results found")
        
        latest_file = comparison_files[-1]
        print(f"Loading results from: {latest_file}")
        
        with open(latest_file, 'r') as f:
            comparison = json.load(f)
        
        return comparison
    
    def load_model_results(self, model_type: str) -> List[Dict]:
        """
        Load detailed results for a specific model
        
        Args:
            model_type: Type of model
            
        Returns:
            List of result dictionaries
        """
        # Find most recent results file for this model
        result_files = sorted(self.results_dir.glob(f"{model_type}_results_*.json"))
        
        if not result_files:
            print(f"⚠️  No results found for {model_type}")
            return []
        
        latest_file = result_files[-1]
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        return data.get('results', [])
    
    def create_metrics_comparison_plot(self, metrics: Dict, output_file: Path):
        """
        Create bar plot comparing metrics across models
        
        Args:
            metrics: Dictionary of metrics per model
            output_file: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models = list(metrics.keys())
        
        # MAE comparison
        mae_values = [metrics[m]['mae'] for m in models if metrics[m]['mae'] is not None]
        mae_models = [m for m in models if metrics[m]['mae'] is not None]
        
        if mae_values:
            axes[0, 0].bar(mae_models, mae_values, color='skyblue', edgecolor='navy')
            axes[0, 0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
            axes[0, 0].set_ylabel('MAE (lower is better)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(mae_values):
                axes[0, 0].text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')
        
        # RMSE comparison
        rmse_values = [metrics[m]['rmse'] for m in models if metrics[m]['rmse'] is not None]
        rmse_models = [m for m in models if metrics[m]['rmse'] is not None]
        
        if rmse_values:
            axes[0, 1].bar(rmse_models, rmse_values, color='lightcoral', edgecolor='darkred')
            axes[0, 1].set_title('Root Mean Square Error (RMSE)', fontweight='bold')
            axes[0, 1].set_ylabel('RMSE (lower is better)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            for i, v in enumerate(rmse_values):
                axes[0, 1].text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Accuracy @ 1 point
        acc1_values = [metrics[m]['accuracy_1point'] * 100 for m in models if metrics[m]['accuracy_1point'] is not None]
        acc1_models = [m for m in models if metrics[m]['accuracy_1point'] is not None]
        
        if acc1_values:
            axes[1, 0].bar(acc1_models, acc1_values, color='lightgreen', edgecolor='darkgreen')
            axes[1, 0].set_title('Accuracy within 1 Point', fontweight='bold')
            axes[1, 0].set_ylabel('Accuracy (%) - higher is better')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].set_ylim(0, 100)
            
            for i, v in enumerate(acc1_values):
                axes[1, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # Accuracy @ 2 points
        acc2_values = [metrics[m]['accuracy_2point'] * 100 for m in models if metrics[m]['accuracy_2point'] is not None]
        acc2_models = [m for m in models if metrics[m]['accuracy_2point'] is not None]
        
        if acc2_values:
            axes[1, 1].bar(acc2_models, acc2_values, color='plum', edgecolor='purple')
            axes[1, 1].set_title('Accuracy within 2 Points', fontweight='bold')
            axes[1, 1].set_ylabel('Accuracy (%) - higher is better')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_ylim(0, 100)
            
            for i, v in enumerate(acc2_values):
                axes[1, 1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Saved metrics comparison plot to: {output_file}")
        plt.close()
    
    def create_score_distribution_plot(self, all_results: Dict[str, List[Dict]], output_file: Path):
        """
        Create distribution plot of predicted vs expected scores
        
        Args:
            all_results: Dictionary of results per model
            output_file: Path to save plot
        """
        fig, axes = plt.subplots(1, len(all_results), figsize=(6*len(all_results), 5))
        
        if len(all_results) == 1:
            axes = [axes]
        
        fig.suptitle('Predicted vs Expected Safety Scores', fontsize=16, fontweight='bold')
        
        for idx, (model, results) in enumerate(all_results.items()):
            # Extract valid scores
            valid_results = [
                r for r in results 
                if r.get('expected_score') is not None and r.get('predicted_score') is not None
            ]
            
            if not valid_results:
                axes[idx].text(0.5, 0.5, 'No valid data', ha='center', va='center')
                axes[idx].set_title(model)
                continue
            
            expected = [r['expected_score'] for r in valid_results]
            predicted = [r['predicted_score'] for r in valid_results]
            
            # Scatter plot
            axes[idx].scatter(expected, predicted, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(min(expected), min(predicted))
            max_val = max(max(expected), max(predicted))
            axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
            
            axes[idx].set_xlabel('Expected Score', fontweight='bold')
            axes[idx].set_ylabel('Predicted Score', fontweight='bold')
            axes[idx].set_title(model, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            
            # Set equal aspect ratio
            axes[idx].set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Saved score distribution plot to: {output_file}")
        plt.close()
    
    def create_error_distribution_plot(self, all_results: Dict[str, List[Dict]], output_file: Path):
        """
        Create histogram of prediction errors
        
        Args:
            all_results: Dictionary of results per model
            output_file: Path to save plot
        """
        fig, axes = plt.subplots(1, len(all_results), figsize=(6*len(all_results), 5))
        
        if len(all_results) == 1:
            axes = [axes]
        
        fig.suptitle('Prediction Error Distribution', fontsize=16, fontweight='bold')
        
        for idx, (model, results) in enumerate(all_results.items()):
            # Calculate errors
            valid_results = [
                r for r in results 
                if r.get('expected_score') is not None and r.get('predicted_score') is not None
            ]
            
            if not valid_results:
                axes[idx].text(0.5, 0.5, 'No valid data', ha='center', va='center')
                axes[idx].set_title(model)
                continue
            
            errors = [r['predicted_score'] - r['expected_score'] for r in valid_results]
            
            # Histogram
            axes[idx].hist(errors, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            axes[idx].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
            
            axes[idx].set_xlabel('Prediction Error (Predicted - Expected)', fontweight='bold')
            axes[idx].set_ylabel('Frequency', fontweight='bold')
            axes[idx].set_title(model, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Saved error distribution plot to: {output_file}")
        plt.close()
    
    def create_detailed_report(self, comparison: Dict, all_results: Dict, output_file: Path):
        """
        Create detailed text report
        
        Args:
            comparison: Comparison summary
            all_results: All model results
            output_file: Path to save report
        """
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL COMPARISON DETAILED REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test set size: {comparison.get('test_size', 'N/A')}\n")
            f.write(f"Models compared: {', '.join(comparison.get('models', []))}\n\n")
            
            # Overall metrics
            f.write("=" * 80 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("=" * 80 + "\n\n")
            
            metrics = comparison.get('metrics', {})
            
            f.write(f"{'Model':<30} {'MAE':<12} {'RMSE':<12} {'Acc@1':<12} {'Acc@2':<12}\n")
            f.write("-" * 78 + "\n")
            
            for model, m in metrics.items():
                mae = f"{m['mae']:.4f}" if m['mae'] is not None else "N/A"
                rmse = f"{m['rmse']:.4f}" if m['rmse'] is not None else "N/A"
                acc1 = f"{m['accuracy_1point']:.2%}" if m['accuracy_1point'] is not None else "N/A"
                acc2 = f"{m['accuracy_2point']:.2%}" if m['accuracy_2point'] is not None else "N/A"
                
                f.write(f"{model:<30} {mae:<12} {rmse:<12} {acc1:<12} {acc2:<12}\n")
            
            f.write("\n\n")
            
            # Per-model detailed analysis
            for model, results in all_results.items():
                f.write("=" * 80 + "\n")
                f.write(f"DETAILED ANALYSIS: {model}\n")
                f.write("=" * 80 + "\n\n")
                
                valid_results = [
                    r for r in results 
                    if r.get('expected_score') is not None and r.get('predicted_score') is not None
                ]
                
                if not valid_results:
                    f.write("No valid results for this model.\n\n")
                    continue
                
                errors = [abs(r['predicted_score'] - r['expected_score']) for r in valid_results]
                
                f.write(f"Valid predictions: {len(valid_results)}/{len(results)}\n")
                f.write(f"Mean error: {sum(errors)/len(errors):.4f}\n")
                f.write(f"Median error: {sorted(errors)[len(errors)//2]:.4f}\n")
                f.write(f"Min error: {min(errors):.4f}\n")
                f.write(f"Max error: {max(errors):.4f}\n\n")
                
                # Best and worst predictions
                sorted_results = sorted(valid_results, key=lambda r: abs(r['predicted_score'] - r['expected_score']))
                
                f.write("Best predictions (lowest error):\n")
                f.write("-" * 78 + "\n")
                for r in sorted_results[:5]:
                    error = abs(r['predicted_score'] - r['expected_score'])
                    f.write(f"  Image: {Path(r['image_path']).name}\n")
                    f.write(f"  Expected: {r['expected_score']:.2f}, Predicted: {r['predicted_score']:.2f}, Error: {error:.2f}\n\n")
                
                f.write("\nWorst predictions (highest error):\n")
                f.write("-" * 78 + "\n")
                for r in sorted_results[-5:]:
                    error = abs(r['predicted_score'] - r['expected_score'])
                    f.write(f"  Image: {Path(r['image_path']).name}\n")
                    f.write(f"  Expected: {r['expected_score']:.2f}, Predicted: {r['predicted_score']:.2f}, Error: {error:.2f}\n\n")
                
                f.write("\n")
        
        print(f"📄 Saved detailed report to: {output_file}")
    
    def analyze(self):
        """Run full analysis and generate all visualizations"""
        print("=" * 60)
        print("ANALYZING COMPARISON RESULTS")
        print("=" * 60)
        
        # Load comparison
        comparison = self.load_latest_comparison()
        
        # Load detailed results for each model
        all_results = {}
        for model in comparison.get('models', []):
            results = self.load_model_results(model)
            if results:
                all_results[model] = results
        
        if not all_results:
            print("❌ No detailed results found")
            return
        
        # Create output directory for plots
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate visualizations
        print("\n📊 Generating visualizations...")
        
        # 1. Metrics comparison
        self.create_metrics_comparison_plot(
            comparison['metrics'],
            plots_dir / f"metrics_comparison_{timestamp}.png"
        )
        
        # 2. Score distribution
        self.create_score_distribution_plot(
            all_results,
            plots_dir / f"score_distribution_{timestamp}.png"
        )
        
        # 3. Error distribution
        self.create_error_distribution_plot(
            all_results,
            plots_dir / f"error_distribution_{timestamp}.png"
        )
        
        # 4. Detailed report
        self.create_detailed_report(
            comparison,
            all_results,
            self.results_dir / f"detailed_report_{timestamp}.txt"
        )
        
        print("\n✅ Analysis complete!")
        print(f"   Results saved to: {self.results_dir}")
        print(f"   Plots saved to: {plots_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze model comparison results")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing comparison results")
    
    args = parser.parse_args()
    
    try:
        analyzer = ComparisonAnalyzer(results_dir=args.results_dir)
        analyzer.analyze()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



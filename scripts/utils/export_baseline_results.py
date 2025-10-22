"""
Export Baseline Evaluation Results to Excel
Converts JSON baseline evaluation results into organized Excel spreadsheets
"""

import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows


def load_json_file(filepath):
    """Load JSON file safely"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def export_pairwise_results(data, writer):
    """Export pairwise comparison results to Excel"""
    if not data or 'safety_scores' not in data:
        return
    
    # Create DataFrame from safety scores
    scores_data = []
    for img_id, score_info in data['safety_scores'].items():
        scores_data.append({
            'Image ID': img_id,
            'Image Path': score_info.get('image_path', ''),
            'Wins': score_info.get('wins', 0),
            'Total Comparisons': score_info.get('total_comparisons', 0),
            'Win Percentage': score_info.get('win_percentage', 0),
            'Safety Score': score_info.get('safety_score', 0)
        })
    
    df = pd.DataFrame(scores_data)
    df = df.sort_values('Safety Score', ascending=False)
    df.to_excel(writer, sheet_name='Safety Scores', index=False)
    
    # Export comparison statistics if available
    if 'total_pairs' in data:
        stats_df = pd.DataFrame([{
            'Metric': 'Total Pairs Compared',
            'Value': data.get('total_pairs', 0)
        }])
        stats_df.to_excel(writer, sheet_name='Comparison Stats', index=False)


def export_training_comparison(data, writer):
    """Export baseline vs training comparison results"""
    if not data or 'matched_pairs' not in data:
        return
    
    # Convert matched pairs to DataFrame
    df = pd.DataFrame(data['matched_pairs'])
    df = df.sort_values('difference', ascending=False)
    df.to_excel(writer, sheet_name='Baseline vs Training', index=False)
    
    # Export statistics
    if 'statistics' in data:
        stats = data['statistics']
        stats_df = pd.DataFrame([{
            'Metric': k,
            'Value': v
        } for k, v in stats.items()])
        stats_df.to_excel(writer, sheet_name='Comparison Statistics', index=False)


def export_anchor_evaluation(data, writer):
    """Export anchor-based evaluation report"""
    if not data:
        return
    
    # Evaluation Summary
    if 'evaluation_summary' in data:
        summary_df = pd.DataFrame([{
            'Metric': k,
            'Value': v
        } for k, v in data['evaluation_summary'].items()])
        summary_df.to_excel(writer, sheet_name='Evaluation Summary', index=False)
    
    # Score Statistics
    if 'score_statistics' in data:
        stats_df = pd.DataFrame([{
            'Metric': k,
            'Value': v
        } for k, v in data['score_statistics'].items()])
        stats_df.to_excel(writer, sheet_name='Score Statistics', index=False)
    
    # Rankings - Safest Images
    if 'rankings' in data and 'safest_images' in data['rankings']:
        safest_data = []
        for img_id, info in data['rankings']['safest_images']:
            row = {'Image ID': img_id}
            row.update(info)
            safest_data.append(row)
        safest_df = pd.DataFrame(safest_data)
        safest_df.to_excel(writer, sheet_name='Safest Images', index=False)
    
    # Rankings - Least Safe Images
    if 'rankings' in data and 'least_safe_images' in data['rankings']:
        least_safe_data = []
        for img_id, info in data['rankings']['least_safe_images']:
            row = {'Image ID': img_id}
            row.update(info)
            least_safe_data.append(row)
        least_safe_df = pd.DataFrame(least_safe_data)
        least_safe_df.to_excel(writer, sheet_name='Least Safe Images', index=False)
    
    # Detailed Scores
    if 'detailed_scores' in data:
        detailed_data = []
        for img_id, scores in data['detailed_scores'].items():
            row = {'Image ID': img_id}
            row.update(scores)
            detailed_data.append(row)
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df = detailed_df.sort_values('safety_score', ascending=False)
        detailed_df.to_excel(writer, sheet_name='Detailed Scores', index=False)


def create_summary_sheet(writer, baseline_dirs, export_info):
    """Create an overview summary sheet"""
    summary_data = []
    
    # Add metadata
    summary_data.append({
        'Item': 'Export Date',
        'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    summary_data.append({
        'Item': 'Baseline Directories Processed',
        'Value': len(baseline_dirs)
    })
    summary_data.append({
        'Item': 'Total Files Exported',
        'Value': len(export_info)
    })
    
    # Add section divider
    summary_data.append({'Item': '', 'Value': ''})
    summary_data.append({
        'Item': '=== Exported Files ===',
        'Value': ''
    })
    
    # List all exported files
    for info in export_info:
        summary_data.append(info)
    
    df = pd.DataFrame(summary_data)
    df.to_excel(writer, sheet_name='Export Summary', index=False)


def export_baseline_results_to_excel(results_dir='results/baseline', output_dir='results'):
    """
    Main function to export all baseline results to Excel
    
    Args:
        results_dir: Directory containing baseline results
        output_dir: Directory to save Excel files
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all baseline result directories
    baseline_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(baseline_dirs)} baseline result directories")
    
    all_exports = []
    
    # Process each baseline directory
    for baseline_dir in baseline_dirs:
        dir_name = baseline_dir.name
        print(f"\nProcessing: {dir_name}")
        
        # Create Excel file for this baseline run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_filename = output_path / f"baseline_results_{dir_name}_{timestamp}.xlsx"
        
        export_info = []
        
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            
            # Look for pairwise results
            pairwise_file = baseline_dir / 'baseline_pairwise_results.json'
            if pairwise_file.exists():
                print(f"  - Exporting pairwise results...")
                data = load_json_file(pairwise_file)
                export_pairwise_results(data, writer)
                export_info.append({
                    'Item': 'Pairwise Results',
                    'Value': str(pairwise_file)
                })
            
            # Look for training comparison
            comparison_file = baseline_dir / 'baseline_vs_training_comparison.json'
            if comparison_file.exists():
                print(f"  - Exporting training comparison...")
                data = load_json_file(comparison_file)
                export_training_comparison(data, writer)
                export_info.append({
                    'Item': 'Training Comparison',
                    'Value': str(comparison_file)
                })
            
            # Look for anchor evaluation report
            anchor_file = baseline_dir / 'anchor_safety_evaluation_report.json'
            if anchor_file.exists():
                print(f"  - Exporting anchor evaluation...")
                data = load_json_file(anchor_file)
                export_anchor_evaluation(data, writer)
                export_info.append({
                    'Item': 'Anchor Evaluation',
                    'Value': str(anchor_file)
                })
            
            # Create summary sheet
            create_summary_sheet(writer, baseline_dirs, export_info)
        
        all_exports.append({
            'Directory': dir_name,
            'Excel File': excel_filename.name,
            'Sheets': len(export_info)
        })
        
        print(f"  [SAVED] {excel_filename}")
    
    # Create a master summary CSV
    if all_exports:
        master_summary = pd.DataFrame(all_exports)
        summary_csv = output_path / f"baseline_exports_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        master_summary.to_csv(summary_csv, index=False)
        print(f"\n[SAVED] Master summary: {summary_csv}")
    
    print(f"\n{'='*60}")
    print(f"Export complete! Processed {len(all_exports)} baseline result sets")
    print(f"Excel files saved in: {output_path}")
    print(f"{'='*60}")


def export_single_consolidated_excel(results_dir='results/baseline', output_file='results/baseline_results_consolidated.xlsx'):
    """
    Export all baseline results into a single consolidated Excel file
    with clearly labeled sheets
    
    Args:
        results_dir: Directory containing baseline results
        output_file: Output Excel file path
    """
    results_path = Path(results_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    # Find all baseline result directories
    baseline_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    
    print(f"Creating consolidated Excel file from {len(baseline_dirs)} baseline directories...")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        export_info = []
        sheet_counter = {}  # Track sheet names to avoid duplicates
        
        for baseline_dir in sorted(baseline_dirs):
            dir_name = baseline_dir.name
            print(f"\nProcessing: {dir_name}")
            
            # Pairwise results
            pairwise_file = baseline_dir / 'baseline_pairwise_results.json'
            if pairwise_file.exists():
                data = load_json_file(pairwise_file)
                if data and 'safety_scores' in data:
                    scores_data = []
                    for img_id, score_info in data['safety_scores'].items():
                        scores_data.append({
                            'Source': dir_name,
                            'Image ID': img_id,
                            'Image Path': score_info.get('image_path', ''),
                            'Wins': score_info.get('wins', 0),
                            'Total Comparisons': score_info.get('total_comparisons', 0),
                            'Win Percentage': score_info.get('win_percentage', 0),
                            'Safety Score': score_info.get('safety_score', 0)
                        })
                    if scores_data:
                        df = pd.DataFrame(scores_data)
                        sheet_name = f"{dir_name[:20]}_Scores"  # Limit sheet name length
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        export_info.append({'Directory': dir_name, 'Type': 'Safety Scores', 'Sheet': sheet_name})
                        print(f"  [OK] Exported safety scores")
            
            # Training comparison
            comparison_file = baseline_dir / 'baseline_vs_training_comparison.json'
            if comparison_file.exists():
                data = load_json_file(comparison_file)
                if data and 'matched_pairs' in data:
                    df = pd.DataFrame(data['matched_pairs'])
                    df['Source'] = dir_name
                    sheet_name = f"{dir_name[:20]}_Comparison"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    export_info.append({'Directory': dir_name, 'Type': 'Training Comparison', 'Sheet': sheet_name})
                    print(f"  [OK] Exported training comparison")
            
            # Anchor evaluation
            anchor_file = baseline_dir / 'anchor_safety_evaluation_report.json'
            if anchor_file.exists():
                data = load_json_file(anchor_file)
                if data and 'detailed_scores' in data:
                    detailed_data = []
                    for img_id, scores in data['detailed_scores'].items():
                        row = {'Source': dir_name, 'Image ID': img_id}
                        row.update(scores)
                        detailed_data.append(row)
                    if detailed_data:
                        df = pd.DataFrame(detailed_data)
                        sheet_name = f"{dir_name[:20]}_Detailed"
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        export_info.append({'Directory': dir_name, 'Type': 'Detailed Scores', 'Sheet': sheet_name})
                        print(f"  [OK] Exported detailed scores")
        
        # Create master index sheet
        if export_info:
            index_df = pd.DataFrame(export_info)
            index_df.to_excel(writer, sheet_name='INDEX', index=False)
            print(f"\n  [OK] Created index sheet")
    
    print(f"\n{'='*60}")
    print(f"[SUCCESS] Consolidated Excel file created: {output_path}")
    print(f"  Total sheets: {len(export_info) + 1}")
    print(f"{'='*60}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export baseline evaluation results to Excel')
    parser.add_argument('--results-dir', default='results/baseline',
                       help='Directory containing baseline results (default: results/baseline)')
    parser.add_argument('--output-dir', default='results',
                       help='Directory to save Excel files (default: results)')
    parser.add_argument('--mode', choices=['separate', 'consolidated', 'both'], default='both',
                       help='Export mode: separate Excel files, one consolidated file, or both')
    parser.add_argument('--consolidated-file', default='results/baseline_results_consolidated.xlsx',
                       help='Path for consolidated Excel file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Baseline Evaluation Results Exporter")
    print("="*60)
    
    if args.mode in ['separate', 'both']:
        print("\n[Mode: Separate Excel Files]")
        export_baseline_results_to_excel(args.results_dir, args.output_dir)
    
    if args.mode in ['consolidated', 'both']:
        print("\n[Mode: Consolidated Excel File]")
        export_single_consolidated_excel(args.results_dir, args.consolidated_file)
    
    print("\n[SUCCESS] All exports complete!")


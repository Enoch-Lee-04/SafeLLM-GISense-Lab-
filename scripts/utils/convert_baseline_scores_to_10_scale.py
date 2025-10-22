"""
Convert Baseline Evaluation Scores from 0-1 scale to 1-10 scale
Matches the manual training data format
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


def convert_scores_to_10_scale(input_file, output_file=None, backup=True):
    """
    Convert safety scores from 0-1 scale to 1-10 scale
    
    Args:
        input_file: Path to baseline results JSON
        output_file: Path to save converted results (default: overwrite input)
        backup: Whether to create a backup of original file
    """
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        return None
    
    # Create backup if requested
    if backup:
        backup_path = input_path.parent / f"{input_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{input_path.suffix}"
        shutil.copy2(input_path, backup_path)
        print(f"[BACKUP] Created backup: {backup_path}")
    
    # Load the data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*70}")
    print("Converting Safety Scores: 0-1 Scale to 1-10 Scale")
    print(f"{'='*70}\n")
    
    # Track conversion statistics
    converted_count = 0
    score_before = []
    score_after = []
    
    # Convert safety_scores
    if 'safety_scores' in data:
        for img_id, score_info in data['safety_scores'].items():
            old_score = score_info.get('safety_score', 0)
            score_before.append(old_score)
            
            # Convert: multiply by 10 to get 1-10 scale
            new_score = old_score * 10
            
            score_info['safety_score'] = new_score
            score_after.append(new_score)
            converted_count += 1
    
    # Also convert win_percentage to percentage format (optional, for clarity)
    if 'safety_scores' in data:
        for img_id, score_info in data['safety_scores'].items():
            win_pct = score_info.get('win_percentage', 0)
            # Keep win_percentage as decimal (0-1), but ensure consistency
            score_info['win_percentage'] = win_pct
    
    # Determine output path
    if output_file is None:
        output_file = input_file
    
    # Save converted data
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[SUCCESS] Converted {converted_count} safety scores")
    print(f"\nScore Statistics:")
    print(f"  Before (0-1 scale):")
    print(f"    - Min: {min(score_before):.2f}")
    print(f"    - Max: {max(score_before):.2f}")
    print(f"    - Avg: {sum(score_before)/len(score_before):.2f}")
    print(f"\n  After (1-10 scale):")
    print(f"    - Min: {min(score_after):.2f}")
    print(f"    - Max: {max(score_after):.2f}")
    print(f"    - Avg: {sum(score_after)/len(score_after):.2f}")
    
    print(f"\n[SAVED] Converted results to: {output_file}")
    print(f"{'='*70}\n")
    
    return output_file


def convert_all_baseline_results(results_dir='results/baseline'):
    """Convert all baseline result files in the directory"""
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Directory not found: {results_dir}")
        return
    
    print(f"\n{'='*70}")
    print("Batch Conversion: All Baseline Results to 1-10 Scale")
    print(f"{'='*70}\n")
    
    # Find all baseline result files
    pairwise_files = list(results_path.rglob('baseline_pairwise_results.json'))
    
    if not pairwise_files:
        print("No baseline result files found")
        return
    
    print(f"Found {len(pairwise_files)} baseline result files to convert:\n")
    
    converted_files = []
    for file in pairwise_files:
        print(f"Converting: {file.relative_to(results_path.parent)}")
        result = convert_scores_to_10_scale(file, backup=True)
        if result:
            converted_files.append(result)
        print()
    
    print(f"{'='*70}")
    print(f"[COMPLETE] Converted {len(converted_files)} files")
    print(f"{'='*70}\n")
    
    return converted_files


def verify_scale_consistency(baseline_file, training_data_file):
    """
    Verify that baseline scores and training scores are on the same scale
    """
    
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    with open(training_data_file, 'r') as f:
        training_data = json.load(f)
    
    # Extract scores from baseline
    baseline_scores = [
        info['safety_score'] 
        for info in baseline_data.get('safety_scores', {}).values()
    ]
    
    # Extract scores from training data (from expected_response)
    training_scores = []
    for item in training_data:
        if item.get('task_type') == 'safety_score':
            response = item.get('expected_response', '')
            # Parse "Safety Score: X" format
            if 'Safety Score:' in response:
                try:
                    score_str = response.split('Safety Score:')[1].split('\n')[0].strip()
                    score = float(score_str)
                    training_scores.append(score)
                except:
                    pass
    
    print(f"\n{'='*70}")
    print("Scale Consistency Verification")
    print(f"{'='*70}\n")
    
    print(f"Baseline Scores:")
    print(f"  Count: {len(baseline_scores)}")
    print(f"  Range: {min(baseline_scores):.2f} - {max(baseline_scores):.2f}")
    print(f"  Average: {sum(baseline_scores)/len(baseline_scores):.2f}")
    
    print(f"\nTraining Data Scores:")
    print(f"  Count: {len(training_scores)}")
    print(f"  Range: {min(training_scores):.0f} - {max(training_scores):.0f}")
    print(f"  Average: {sum(training_scores)/len(training_scores):.2f}")
    
    # Check if scales match
    baseline_max = max(baseline_scores)
    training_max = max(training_scores)
    
    if abs(baseline_max - training_max) < 2:
        print(f"\n[OK] GOOD: Scales are consistent (both using 1-10 range)")
    else:
        print(f"\n[WARNING] Scales may not match!")
        print(f"   Baseline max: {baseline_max:.2f}")
        print(f"   Training max: {training_max:.0f}")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert baseline scores to 1-10 scale')
    parser.add_argument('--file', help='Single file to convert')
    parser.add_argument('--all', action='store_true', 
                       help='Convert all baseline results in results/baseline/')
    parser.add_argument('--verify', action='store_true',
                       help='Verify scale consistency with training data')
    parser.add_argument('--results-dir', default='results/baseline',
                       help='Directory containing baseline results')
    parser.add_argument('--training-data', default='configs/vlm_safety_training_data.json',
                       help='Path to training data JSON')
    
    args = parser.parse_args()
    
    if args.file:
        convert_scores_to_10_scale(args.file)
        if args.verify:
            verify_scale_consistency(args.file, args.training_data)
    elif args.all:
        convert_all_baseline_results(args.results_dir)
        # Verify the fixed_baseline_gpu results
        fixed_gpu_file = Path(args.results_dir) / 'fixed_baseline_gpu' / 'baseline_pairwise_results.json'
        if fixed_gpu_file.exists() and args.verify:
            verify_scale_consistency(str(fixed_gpu_file), args.training_data)
    else:
        print("Please specify --file or --all")
        parser.print_help()


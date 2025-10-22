"""
Export VLM Safety Training Data to Excel
Converts JSON training data into organized Excel spreadsheet
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime


def export_training_data_to_excel(
    json_file='configs/vlm_safety_training_data.json',
    output_file='configs/vlm_safety_training_data.xlsx'
):
    """
    Export VLM training data to Excel with organized sheets
    
    Args:
        json_file: Path to input JSON file
        output_file: Path to output Excel file
    """
    
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*70}")
    print("VLM Safety Training Data Export")
    print(f"{'='*70}\n")
    print(f"Loading data from: {json_file}")
    print(f"Total training examples: {len(data)}\n")
    
    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # Sheet 1: All Data - Complete overview
        all_data = []
        for idx, item in enumerate(data, 1):
            all_data.append({
                'ID': idx,
                'Image Path': item.get('image_path', '').split('\\')[-1],  # Just filename
                'Task Type': item.get('task_type', ''),
                'Expected Format': item.get('expected_format', ''),
                'Prompt Preview': item.get('prompt', '')[:100] + '...' if len(item.get('prompt', '')) > 100 else item.get('prompt', ''),
                'Response Preview': item.get('expected_response', '')[:100] + '...' if len(item.get('expected_response', '')) > 100 else item.get('expected_response', '')
            })
        
        df_all = pd.DataFrame(all_data)
        df_all.to_excel(writer, sheet_name='Overview', index=False)
        print(f"[OK] Created 'Overview' sheet with {len(df_all)} entries")
        
        # Sheet 2: By Image - Group by image
        images_summary = {}
        for item in data:
            img_path = item.get('image_path', '').split('\\')[-1]
            if img_path not in images_summary:
                images_summary[img_path] = {
                    'Image': img_path,
                    'Total Examples': 0,
                    'Task Types': set()
                }
            images_summary[img_path]['Total Examples'] += 1
            images_summary[img_path]['Task Types'].add(item.get('task_type', ''))
        
        images_data = []
        for img, info in sorted(images_summary.items()):
            images_data.append({
                'Image': img,
                'Total Examples': info['Total Examples'],
                'Task Types': ', '.join(sorted(info['Task Types']))
            })
        
        df_images = pd.DataFrame(images_data)
        df_images.to_excel(writer, sheet_name='By Image', index=False)
        print(f"[OK] Created 'By Image' sheet with {len(df_images)} unique images")
        
        # Sheet 3: By Task Type
        task_summary = {}
        for item in data:
            task = item.get('task_type', 'unknown')
            if task not in task_summary:
                task_summary[task] = 0
            task_summary[task] += 1
        
        task_data = []
        for task, count in sorted(task_summary.items()):
            task_data.append({
                'Task Type': task,
                'Count': count,
                'Percentage': f"{(count/len(data)*100):.1f}%"
            })
        
        df_tasks = pd.DataFrame(task_data)
        df_tasks.to_excel(writer, sheet_name='By Task Type', index=False)
        print(f"[OK] Created 'By Task Type' sheet with {len(df_tasks)} task types")
        
        # Sheet 4-8: Detailed sheets for each task type
        task_types = {
            'binary_classification': 'Binary Classification',
            'risk_assessment': 'Risk Assessment',
            'detailed_analysis': 'Detailed Analysis',
            'safety_score': 'Safety Score'
        }
        
        for task_key, sheet_name in task_types.items():
            task_items = [item for item in data if item.get('task_type') == task_key]
            if task_items:
                task_details = []
                for idx, item in enumerate(task_items, 1):
                    task_details.append({
                        'ID': idx,
                        'Image': item.get('image_path', '').split('\\')[-1],
                        'Full Prompt': item.get('prompt', ''),
                        'Expected Response': item.get('expected_response', ''),
                        'Expected Format': item.get('expected_format', '')
                    })
                
                df_task = pd.DataFrame(task_details)
                df_task.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet name limit
                print(f"[OK] Created '{sheet_name}' sheet with {len(df_task)} examples")
        
        # Sheet 9: Full Data - Everything
        full_data = []
        for idx, item in enumerate(data, 1):
            full_data.append({
                'ID': idx,
                'Image Path': item.get('image_path', ''),
                'Task Type': item.get('task_type', ''),
                'Expected Format': item.get('expected_format', ''),
                'Full Prompt': item.get('prompt', ''),
                'Expected Response': item.get('expected_response', '')
            })
        
        df_full = pd.DataFrame(full_data)
        df_full.to_excel(writer, sheet_name='Full Data', index=False)
        print(f"[OK] Created 'Full Data' sheet with complete information")
        
        # Sheet 10: Statistics
        stats_data = [
            {'Metric': 'Total Training Examples', 'Value': len(data)},
            {'Metric': 'Unique Images', 'Value': len(images_summary)},
            {'Metric': 'Task Types', 'Value': len(task_summary)},
            {'Metric': 'Binary Classification Examples', 'Value': task_summary.get('binary_classification', 0)},
            {'Metric': 'Risk Assessment Examples', 'Value': task_summary.get('risk_assessment', 0)},
            {'Metric': 'Detailed Analysis Examples', 'Value': task_summary.get('detailed_analysis', 0)},
            {'Metric': 'Safety Score Examples', 'Value': task_summary.get('safety_score', 0)},
            {'Metric': 'Average Examples per Image', 'Value': f"{len(data)/len(images_summary):.1f}"},
            {'Metric': 'Export Date', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
            {'Metric': 'Source File', 'Value': json_file}
        ]
        
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_excel(writer, sheet_name='Statistics', index=False)
        print(f"[OK] Created 'Statistics' sheet")
    
    print(f"\n{'='*70}")
    print(f"[SUCCESS] Excel file created: {output_file}")
    print(f"{'='*70}\n")
    
    # Display summary
    print("Summary:")
    print(f"  - Total examples: {len(data)}")
    print(f"  - Unique images: {len(images_summary)}")
    print(f"  - Task types: {len(task_summary)}")
    print(f"  - Sheets created: 10")
    print(f"\nTask type breakdown:")
    for task, count in sorted(task_summary.items()):
        print(f"  - {task}: {count} examples ({count/len(data)*100:.1f}%)")
    
    return output_file


def view_training_data_summary(json_file='configs/vlm_safety_training_data.json'):
    """Display a quick summary of the training data"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*70}")
    print("VLM Safety Training Data Summary")
    print(f"{'='*70}\n")
    
    # Count by image
    images = {}
    for item in data:
        img = item.get('image_path', '').split('\\')[-1]
        images[img] = images.get(img, 0) + 1
    
    # Count by task type
    tasks = {}
    for item in data:
        task = item.get('task_type', 'unknown')
        tasks[task] = tasks.get(task, 0) + 1
    
    print(f"Total Examples: {len(data)}")
    print(f"Unique Images: {len(images)}")
    print(f"\nImages with most examples:")
    for img, count in sorted(images.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {img}: {count} examples")
    
    print(f"\nTask Type Distribution:")
    for task, count in sorted(tasks.items()):
        print(f"  {task}: {count} ({count/len(data)*100:.1f}%)")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export VLM training data to Excel')
    parser.add_argument('--input', default='configs/vlm_safety_training_data.json',
                       help='Input JSON file (default: configs/vlm_safety_training_data.json)')
    parser.add_argument('--output', default='configs/vlm_safety_training_data.xlsx',
                       help='Output Excel file (default: configs/vlm_safety_training_data.xlsx)')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary only, do not export')
    
    args = parser.parse_args()
    
    if args.summary:
        view_training_data_summary(args.input)
    else:
        export_training_data_to_excel(args.input, args.output)


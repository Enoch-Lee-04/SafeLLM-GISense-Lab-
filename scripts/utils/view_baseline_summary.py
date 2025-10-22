"""
Quick viewer for baseline evaluation results in Excel
"""

import pandas as pd
from pathlib import Path


def view_excel_summary(excel_file='results/baseline_results_consolidated.xlsx'):
    """Display a summary of the Excel file contents"""
    
    if not Path(excel_file).exists():
        print(f"Excel file not found: {excel_file}")
        print("Run 'python scripts/utils/export_baseline_results.py --mode consolidated' first")
        return
    
    print("\n" + "="*70)
    print(f"BASELINE EVALUATION RESULTS SUMMARY")
    print("="*70)
    print(f"\nFile: {excel_file}\n")
    
    xl = pd.ExcelFile(excel_file)
    
    # Show all sheets
    print("Available Sheets:")
    print("-"*70)
    for i, sheet in enumerate(xl.sheet_names, 1):
        print(f"  {i:2d}. {sheet}")
    
    print("\n" + "="*70)
    
    # Show INDEX if available
    if 'INDEX' in xl.sheet_names:
        print("\nQuick Index:")
        print("-"*70)
        index_df = pd.read_excel(excel_file, sheet_name='INDEX')
        print(index_df.to_string(index=False))
    
    # Show sample data from first data sheet
    data_sheets = [s for s in xl.sheet_names if s != 'INDEX']
    if data_sheets:
        first_sheet = data_sheets[0]
        print(f"\n\nSample Data from '{first_sheet}':")
        print("-"*70)
        df = pd.read_excel(excel_file, sheet_name=first_sheet)
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"\nColumns: {', '.join(df.columns.tolist())}")
        print(f"\nFirst 5 rows:")
        print(df.head().to_string(index=False))
    
    print("\n" + "="*70)
    print("\nTo open the full file:")
    print(f"  - Double-click: {excel_file}")
    print(f"  - Or use Excel, LibreOffice, or Google Sheets")
    print("\nFor detailed usage, see: results/BASELINE_RESULTS_GUIDE.md")
    print("="*70 + "\n")


def compare_runs(excel_file='results/baseline_results_consolidated.xlsx'):
    """Compare safety scores across different baseline runs"""
    
    if not Path(excel_file).exists():
        print(f"Excel file not found: {excel_file}")
        return
    
    xl = pd.ExcelFile(excel_file)
    score_sheets = [s for s in xl.sheet_names if '_Scores' in s]
    
    if not score_sheets:
        print("No score sheets found")
        return
    
    print("\n" + "="*70)
    print("COMPARISON ACROSS BASELINE RUNS")
    print("="*70 + "\n")
    
    for sheet in score_sheets:
        df = pd.read_excel(excel_file, sheet_name=sheet)
        run_name = sheet.replace('_Scores', '')
        
        print(f"{run_name}:")
        print(f"  Images evaluated: {len(df)}")
        if 'Safety Score' in df.columns:
            print(f"  Average safety score: {df['Safety Score'].mean():.2f}")
            print(f"  Max safety score: {df['Safety Score'].max():.2f}")
            print(f"  Min safety score: {df['Safety Score'].min():.2f}")
        print()
    
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='View baseline evaluation results summary')
    parser.add_argument('--file', default='results/baseline_results_consolidated.xlsx',
                       help='Excel file to view')
    parser.add_argument('--compare', action='store_true',
                       help='Compare results across runs')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_runs(args.file)
    else:
        view_excel_summary(args.file)


#!/usr/bin/env python3
"""
Large Files Analyzer for Soundscape-to-Image Project
This script identifies large files in your project to help optimize disk usage.
"""

import os
import sys
from pathlib import Path
import json

def get_file_size_mb(file_path):
    """Get file size in MB"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except (OSError, PermissionError):
        return 0

def analyze_directory(directory_path, min_size_mb=10):
    """Analyze directory for large files"""
    large_files = []
    total_size = 0
    file_count = 0
    
    print(f"Analyzing directory: {directory_path}")
    print(f"Looking for files larger than {min_size_mb} MB")
    print("-" * 60)
    
    try:
        for root, dirs, files in os.walk(directory_path):
            # Skip certain directories to avoid system files
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_size_mb = get_file_size_mb(file_path)
                total_size += file_size_mb
                file_count += 1
                
                if file_size_mb >= min_size_mb:
                    relative_path = os.path.relpath(file_path, directory_path)
                    large_files.append({
                        'path': relative_path,
                        'size_mb': round(file_size_mb, 2),
                        'full_path': file_path
                    })
    except (OSError, PermissionError) as e:
        print(f"Error accessing directory: {e}")
        return [], 0, 0
    
    return large_files, total_size, file_count

def analyze_specific_file_types(directory_path):
    """Analyze specific file types that are commonly large"""
    file_types = {
        '.pt': 'PyTorch model files',
        '.pth': 'PyTorch checkpoint files', 
        '.safetensors': 'SafeTensors model files',
        '.bin': 'Binary model files',
        '.json': 'JSON data files',
        '.wav': 'Audio files',
        '.jpg': 'Image files',
        '.png': 'Image files',
        '.mp4': 'Video files',
        '.avi': 'Video files',
        '.zip': 'Archive files',
        '.tar': 'Archive files',
        '.gz': 'Compressed files'
    }
    
    type_stats = {}
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in file_types:
                file_size_mb = get_file_size_mb(file_path)
                
                if file_ext not in type_stats:
                    type_stats[file_ext] = {
                        'description': file_types[file_ext],
                        'count': 0,
                        'total_size_mb': 0,
                        'files': []
                    }
                
                type_stats[file_ext]['count'] += 1
                type_stats[file_ext]['total_size_mb'] += file_size_mb
                
                if file_size_mb > 1:  # Only track files > 1MB
                    relative_path = os.path.relpath(file_path, directory_path)
                    type_stats[file_ext]['files'].append({
                        'path': relative_path,
                        'size_mb': round(file_size_mb, 2)
                    })
    
    return type_stats

def main():
    """Main analysis function"""
    print("Large Files Analyzer for Soundscape-to-Image Project")
    print("=" * 60)
    
    # Get project directory
    project_dir = Path(__file__).parent.absolute()
    print(f"Project directory: {project_dir}")
    print()
    
    # Analyze large files (>10MB)
    print("1. LARGE FILES ANALYSIS (>10MB)")
    print("=" * 40)
    large_files, total_size, file_count = analyze_directory(project_dir, min_size_mb=10)
    
    if large_files:
        # Sort by size (largest first)
        large_files.sort(key=lambda x: x['size_mb'], reverse=True)
        
        print(f"Found {len(large_files)} files larger than 10MB:")
        print()
        for i, file_info in enumerate(large_files, 1):
            print(f"{i:2d}. {file_info['path']}")
            print(f"    Size: {file_info['size_mb']:.2f} MB")
            print()
    else:
        print("No files larger than 10MB found.")
    
    print()
    
    # Analyze file types
    print("2. FILE TYPE ANALYSIS")
    print("=" * 40)
    type_stats = analyze_specific_file_types(project_dir)
    
    if type_stats:
        # Sort by total size
        sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['total_size_mb'], reverse=True)
        
        for file_ext, stats in sorted_types:
            if stats['total_size_mb'] > 0:
                print(f"\n{file_ext.upper()} - {stats['description']}")
                print(f"  Files: {stats['count']}")
                print(f"  Total Size: {stats['total_size_mb']:.2f} MB")
                
                if stats['files']:
                    print(f"  Large files (>1MB):")
                    for file_info in sorted(stats['files'], key=lambda x: x['size_mb'], reverse=True)[:5]:  # Top 5
                        print(f"    - {file_info['path']} ({file_info['size_mb']:.2f} MB)")
                    if len(stats['files']) > 5:
                        print(f"    ... and {len(stats['files']) - 5} more files")
    
    print()
    
    # Summary
    print("3. SUMMARY")
    print("=" * 40)
    print(f"Total files analyzed: {file_count}")
    print(f"Total project size: {total_size:.2f} MB")
    print(f"Large files (>10MB): {len(large_files)}")
    
    if large_files:
        large_files_total = sum(f['size_mb'] for f in large_files)
        print(f"Large files total size: {large_files_total:.2f} MB")
        print(f"Percentage of project: {(large_files_total/total_size)*100:.1f}%")
    
    print()
    
    # Recommendations
    print("4. RECOMMENDATIONS")
    print("=" * 40)
    
    if large_files:
        print("Large files that might be candidates for optimization:")
        for file_info in large_files[:5]:  # Top 5 largest
            print(f"  - {file_info['path']} ({file_info['size_mb']:.2f} MB)")
            
            # Specific recommendations based on file type
            if file_info['path'].endswith('.pt') or file_info['path'].endswith('.pth'):
                print(f"    -> Consider: Model compression, quantization, or moving to cloud storage")
            elif file_info['path'].endswith('.json'):
                print(f"    -> Consider: JSON compression or splitting into smaller files")
            elif file_info['path'].endswith('.wav'):
                print(f"    -> Consider: Audio compression or format conversion")
            elif file_info['path'].endswith(('.jpg', '.png')):
                print(f"    -> Consider: Image compression or format optimization")
    
    # Check for virtual environments
    venv_dirs = [d for d in os.listdir(project_dir) if d.startswith('venv_') and os.path.isdir(os.path.join(project_dir, d))]
    if venv_dirs:
        print(f"\nVirtual environments found: {', '.join(venv_dirs)}")
        print("  -> These can be recreated as needed to save space")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

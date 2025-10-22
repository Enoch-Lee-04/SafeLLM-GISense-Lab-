#!/usr/bin/env python3
"""
Repository Cleanup Script for Soundscape-to-Image Project
This script identifies and removes unnecessary/temporary files to organize the repository.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def get_file_size_mb(file_path):
    """Get file size in MB"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except (OSError, PermissionError):
        return 0

def analyze_repository_structure(project_dir):
    """Analyze the repository structure and identify cleanup opportunities"""
    print("Analyzing repository structure...")
    print("-" * 50)
    
    cleanup_candidates = {
        'duplicate_results': [],
        'temp_files': [],
        'cache_files': [],
        'old_scripts': [],
        'unused_data': [],
        'large_json_files': [],
        'backup_files': []
    }
    
    total_files = 0
    total_size = 0
    
    for root, dirs, files in os.walk(project_dir):
        # Skip virtual environments and git directories
        dirs[:] = [d for d in dirs if not d.startswith('venv_') and d != '.git' and d != '__pycache__']
        
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, project_dir)
            file_size_mb = get_file_size_mb(file_path)
            
            total_files += 1
            total_size += file_size_mb
            
            # Identify cleanup candidates
            if 'results' in relative_path.lower() and 'fixed' in relative_path.lower():
                cleanup_candidates['duplicate_results'].append({
                    'path': relative_path,
                    'size_mb': file_size_mb
                })
            
            if file.endswith('.pyc') or file.endswith('.pyo'):
                cleanup_candidates['cache_files'].append({
                    'path': relative_path,
                    'size_mb': file_size_mb
                })
            
            if file.startswith('temp_') or file.startswith('tmp_'):
                cleanup_candidates['temp_files'].append({
                    'path': relative_path,
                    'size_mb': file_size_mb
                })
            
            if file.endswith('.json') and file_size_mb > 5:
                cleanup_candidates['large_json_files'].append({
                    'path': relative_path,
                    'size_mb': file_size_mb
                })
            
            if file.endswith('.bak') or file.endswith('.backup'):
                cleanup_candidates['backup_files'].append({
                    'path': relative_path,
                    'size_mb': file_size_mb
                })
            
            if 'debug_' in file or 'test_' in file:
                cleanup_candidates['old_scripts'].append({
                    'path': relative_path,
                    'size_mb': file_size_mb
                })
    
    return cleanup_candidates, total_files, total_size

def identify_duplicate_directories(project_dir):
    """Identify duplicate or similar directories"""
    print("\nIdentifying duplicate directories...")
    print("-" * 50)
    
    directories = []
    for item in os.listdir(project_dir):
        item_path = os.path.join(project_dir, item)
        if os.path.isdir(item_path) and not item.startswith('venv_') and item != '.git':
            directories.append(item)
    
    # Look for patterns like "results", "results_fixed", "results_final"
    duplicate_patterns = {}
    
    for dir_name in directories:
        base_name = dir_name.replace('_fixed', '').replace('_final', '').replace('_v2', '')
        if base_name not in duplicate_patterns:
            duplicate_patterns[base_name] = []
        duplicate_patterns[base_name].append(dir_name)
    
    # Find directories with multiple versions
    duplicates = {k: v for k, v in duplicate_patterns.items() if len(v) > 1}
    
    return duplicates

def analyze_large_files(project_dir, min_size_mb=10):
    """Analyze large files that might be candidates for cleanup"""
    print(f"\nAnalyzing large files (>={min_size_mb}MB)...")
    print("-" * 50)
    
    large_files = []
    
    for root, dirs, files in os.walk(project_dir):
        # Skip virtual environments
        dirs[:] = [d for d in dirs if not d.startswith('venv_')]
        
        for file in files:
            file_path = os.path.join(root, file)
            file_size_mb = get_file_size_mb(file_path)
            
            if file_size_mb >= min_size_mb:
                relative_path = os.path.relpath(file_path, project_dir)
                large_files.append({
                    'path': relative_path,
                    'size_mb': file_size_mb
                })
    
    return sorted(large_files, key=lambda x: x['size_mb'], reverse=True)

def create_cleanup_plan(cleanup_candidates, duplicate_dirs, large_files, project_dir):
    """Create a comprehensive cleanup plan"""
    print("\n" + "="*60)
    print("REPOSITORY CLEANUP PLAN")
    print("="*60)
    
    total_space_to_free = 0
    
    print("\n1. DUPLICATE RESULTS DIRECTORIES")
    print("-" * 40)
    if duplicate_dirs:
        for base_name, versions in duplicate_dirs.items():
            print(f"\n{base_name}:")
            for version in versions:
                version_path = os.path.join(project_dir, version)
                if os.path.exists(version_path):
                    size_mb = sum(get_file_size_mb(os.path.join(version_path, f)) 
                                for f in os.listdir(version_path) 
                                if os.path.isfile(os.path.join(version_path, f)))
                    print(f"  - {version} ({size_mb:.1f} MB)")
                    if 'fixed' in version or 'final' in version:
                        total_space_to_free += size_mb
                        print(f"    -> KEEP (most recent)")
                    else:
                        print(f"    -> REMOVE (older version)")
    else:
        print("No duplicate directories found.")
    
    print("\n2. TEMPORARY AND CACHE FILES")
    print("-" * 40)
    for category, files in cleanup_candidates.items():
        if files:
            print(f"\n{category.replace('_', ' ').title()}:")
            for file_info in files:
                print(f"  - {file_info['path']} ({file_info['size_mb']:.1f} MB)")
                total_space_to_free += file_info['size_mb']
    
    print("\n3. LARGE FILES ANALYSIS")
    print("-" * 40)
    if large_files:
        print("Files that might need attention:")
        for file_info in large_files[:10]:  # Top 10 largest
            print(f"  - {file_info['path']} ({file_info['size_mb']:.1f} MB)")
    
    print(f"\n4. CLEANUP SUMMARY")
    print("-" * 40)
    print(f"Total space that can be freed: {total_space_to_free:.1f} MB ({total_space_to_free/1024:.1f} GB)")
    
    return total_space_to_free

def execute_cleanup(project_dir, cleanup_candidates, duplicate_dirs):
    """Execute the cleanup operations"""
    print("\n" + "="*60)
    print("EXECUTING CLEANUP")
    print("="*60)
    
    removed_files = []
    total_space_freed = 0
    
    # Remove duplicate result directories
    print("\n1. Removing duplicate result directories...")
    for base_name, versions in duplicate_dirs.items():
        for version in versions:
            if 'fixed' not in version and 'final' not in version:
                version_path = os.path.join(project_dir, version)
                if os.path.exists(version_path):
                    try:
                        size_mb = sum(get_file_size_mb(os.path.join(version_path, f)) 
                                    for f in os.listdir(version_path) 
                                    if os.path.isfile(os.path.join(version_path, f)))
                        shutil.rmtree(version_path)
                        removed_files.append({
                            'path': version,
                            'type': 'directory',
                            'size_mb': size_mb
                        })
                        total_space_freed += size_mb
                        print(f"  Removed: {version} ({size_mb:.1f} MB)")
                    except Exception as e:
                        print(f"  Failed to remove {version}: {e}")
    
    # Remove temporary and cache files
    print("\n2. Removing temporary and cache files...")
    for category, files in cleanup_candidates.items():
        if files:
            print(f"\nRemoving {category.replace('_', ' ')}:")
            for file_info in files:
                file_path = os.path.join(project_dir, file_info['path'])
                try:
                    os.remove(file_path)
                    removed_files.append({
                        'path': file_info['path'],
                        'type': 'file',
                        'size_mb': file_info['size_mb']
                    })
                    total_space_freed += file_info['size_mb']
                    print(f"  Removed: {file_info['path']} ({file_info['size_mb']:.1f} MB)")
                except Exception as e:
                    print(f"  Failed to remove {file_info['path']}: {e}")
    
    # Create cleanup log
    cleanup_log_path = os.path.join(project_dir, "repository_cleanup_log.txt")
    with open(cleanup_log_path, 'w') as f:
        f.write(f"Repository Cleanup Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Files and directories removed:\n\n")
        
        for item in removed_files:
            f.write(f"Type: {item['type']}\n")
            f.write(f"Path: {item['path']}\n")
            f.write(f"Size: {item['size_mb']:.2f} MB\n")
            f.write(f"Removed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 40 + "\n")
    
    print(f"\nCleanup log created: {cleanup_log_path}")
    print(f"Total space freed: {total_space_freed:.1f} MB ({total_space_freed/1024:.1f} GB)")
    
    return removed_files, total_space_freed

def organize_repository(project_dir):
    """Suggest repository organization improvements"""
    print("\n" + "="*60)
    print("REPOSITORY ORGANIZATION SUGGESTIONS")
    print("="*60)
    
    suggestions = []
    
    # Check for scattered result files
    result_files = []
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            if 'result' in file.lower() and file.endswith('.json'):
                result_files.append(os.path.relpath(os.path.join(root, file), project_dir))
    
    if len(result_files) > 5:
        suggestions.append("Consider creating a 'results' directory to organize all result files")
    
    # Check for scattered scripts
    script_files = []
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('setup_') and not file.startswith('cleanup_'):
                script_files.append(os.path.relpath(os.path.join(root, file), project_dir))
    
    if len(script_files) > 10:
        suggestions.append("Consider organizing scripts into subdirectories (training/, evaluation/, utils/)")
    
    # Check for data organization
    data_dirs = [d for d in os.listdir(project_dir) if 'data' in d.lower()]
    if len(data_dirs) > 1:
        suggestions.append("Consider consolidating data directories")
    
    if suggestions:
        print("Organization suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
    else:
        print("Repository is well organized!")
    
    return suggestions

def main():
    """Main cleanup function"""
    print("Repository Cleanup Script for Soundscape-to-Image Project")
    print("=" * 60)
    
    project_dir = Path(__file__).parent.absolute()
    print(f"Project directory: {project_dir}")
    
    # Analyze repository
    cleanup_candidates, total_files, total_size = analyze_repository_structure(project_dir)
    duplicate_dirs = identify_duplicate_directories(project_dir)
    large_files = analyze_large_files(project_dir)
    
    print(f"\nRepository Analysis Complete:")
    print(f"Total files: {total_files}")
    print(f"Total size: {total_size:.1f} MB ({total_size/1024:.1f} GB)")
    
    # Create cleanup plan
    total_space_to_free = create_cleanup_plan(cleanup_candidates, duplicate_dirs, large_files, project_dir)
    
    if total_space_to_free > 0:
        print(f"\nProceeding with cleanup...")
        
        # Execute cleanup
        removed_files, actual_space_freed = execute_cleanup(project_dir, cleanup_candidates, duplicate_dirs)
        
        print(f"\n" + "="*60)
        print("CLEANUP COMPLETE")
        print("="*60)
        print(f"Files/directories removed: {len(removed_files)}")
        print(f"Space freed: {actual_space_freed:.1f} MB ({actual_space_freed/1024:.1f} GB)")
        
        # Organization suggestions
        organize_repository(project_dir)
        
        print(f"\nRepository cleanup successful!")
    else:
        print(f"\nNo cleanup needed - repository is already clean!")

if __name__ == "__main__":
    main()

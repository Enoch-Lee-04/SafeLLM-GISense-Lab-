#!/usr/bin/env python3
"""
Windows Disk Cleanup Script
This script helps free up disk space by cleaning temporary files, caches, and other unnecessary data.
"""

import os
import shutil
import subprocess
import tempfile
import glob
from pathlib import Path
import psutil

def get_disk_usage():
    """Get current disk usage information"""
    disk_usage = psutil.disk_usage('C:')
    total_gb = disk_usage.total / (1024**3)
    free_gb = disk_usage.free / (1024**3)
    used_gb = disk_usage.used / (1024**3)
    percent_free = (free_gb / total_gb) * 100
    
    print(f"Current C: Drive Status:")
    print(f"  Total Space: {total_gb:.2f} GB")
    print(f"  Used Space: {used_gb:.2f} GB")
    print(f"  Free Space: {free_gb:.2f} GB")
    print(f"  Percent Free: {percent_free:.2f}%")
    print()
    
    return free_gb

def clean_temp_files():
    """Clean Windows temporary files"""
    print("Cleaning Windows temporary files...")
    cleaned_size = 0
    
    # Common temp directories
    temp_dirs = [
        os.environ.get('TEMP', ''),
        os.environ.get('TMP', ''),
        os.path.join(os.environ.get('USERPROFILE', ''), 'AppData', 'Local', 'Temp'),
        os.path.join(os.environ.get('WINDIR', ''), 'Temp'),
        os.path.join(os.environ.get('WINDIR', ''), 'Prefetch'),
    ]
    
    for temp_dir in temp_dirs:
        if temp_dir and os.path.exists(temp_dir):
            try:
                print(f"  Cleaning: {temp_dir}")
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_size += file_size
                        except (OSError, PermissionError):
                            pass  # Skip files that can't be deleted
                
                # Remove empty directories
                for root, dirs, files in os.walk(temp_dir, topdown=False):
                    for dir_name in dirs:
                        try:
                            dir_path = os.path.join(root, dir_name)
                            if not os.listdir(dir_path):
                                os.rmdir(dir_path)
                        except (OSError, PermissionError):
                            pass
                            
            except (OSError, PermissionError):
                print(f"  Could not clean: {temp_dir}")
    
    print(f"  Cleaned: {cleaned_size / (1024**2):.2f} MB from temp files")
    return cleaned_size

def clean_browser_caches():
    """Clean browser caches"""
    print("Cleaning browser caches...")
    cleaned_size = 0
    
    user_profile = os.environ.get('USERPROFILE', '')
    if not user_profile:
        return cleaned_size
    
    # Chrome cache
    chrome_cache = os.path.join(user_profile, 'AppData', 'Local', 'Google', 'Chrome', 'User Data', 'Default', 'Cache')
    if os.path.exists(chrome_cache):
        try:
            print(f"  Cleaning Chrome cache: {chrome_cache}")
            shutil.rmtree(chrome_cache)
            cleaned_size += get_folder_size(chrome_cache)
        except (OSError, PermissionError):
            print(f"  Could not clean Chrome cache")
    
    # Edge cache
    edge_cache = os.path.join(user_profile, 'AppData', 'Local', 'Microsoft', 'Edge', 'User Data', 'Default', 'Cache')
    if os.path.exists(edge_cache):
        try:
            print(f"  Cleaning Edge cache: {edge_cache}")
            shutil.rmtree(edge_cache)
            cleaned_size += get_folder_size(edge_cache)
        except (OSError, PermissionError):
            print(f"  Could not clean Edge cache")
    
    # Firefox cache
    firefox_cache = os.path.join(user_profile, 'AppData', 'Local', 'Mozilla', 'Firefox', 'Profiles')
    if os.path.exists(firefox_cache):
        try:
            for profile_dir in os.listdir(firefox_cache):
                cache_dir = os.path.join(firefox_cache, profile_dir, 'cache2')
                if os.path.exists(cache_dir):
                    print(f"  Cleaning Firefox cache: {cache_dir}")
                    shutil.rmtree(cache_dir)
                    cleaned_size += get_folder_size(cache_dir)
        except (OSError, PermissionError):
            print(f"  Could not clean Firefox cache")
    
    print(f"  Cleaned: {cleaned_size / (1024**2):.2f} MB from browser caches")
    return cleaned_size

def clean_windows_update_cache():
    """Clean Windows Update cache"""
    print("Cleaning Windows Update cache...")
    cleaned_size = 0
    
    # Windows Update cache
    update_cache = os.path.join(os.environ.get('WINDIR', ''), 'SoftwareDistribution', 'Download')
    if os.path.exists(update_cache):
        try:
            print(f"  Cleaning Windows Update cache: {update_cache}")
            for root, dirs, files in os.walk(update_cache):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        cleaned_size += file_size
                    except (OSError, PermissionError):
                        pass
        except (OSError, PermissionError):
            print(f"  Could not clean Windows Update cache")
    
    print(f"  Cleaned: {cleaned_size / (1024**2):.2f} MB from Windows Update cache")
    return cleaned_size

def clean_recycle_bin():
    """Empty the Recycle Bin"""
    print("Emptying Recycle Bin...")
    try:
        # Use PowerShell to empty recycle bin
        subprocess.run([
            'powershell', '-Command', 
            'Clear-RecycleBin -Force -ErrorAction SilentlyContinue'
        ], capture_output=True)
        print("  Recycle Bin emptied")
        return True
    except Exception as e:
        print(f"  Could not empty Recycle Bin: {e}")
        return False

def clean_python_cache():
    """Clean Python cache files"""
    print("Cleaning Python cache files...")
    cleaned_size = 0
    
    # Find all __pycache__ directories
    user_profile = os.environ.get('USERPROFILE', '')
    if user_profile:
        for root, dirs, files in os.walk(user_profile):
            if '__pycache__' in dirs:
                pycache_dir = os.path.join(root, '__pycache__')
                try:
                    print(f"  Cleaning: {pycache_dir}")
                    shutil.rmtree(pycache_dir)
                    cleaned_size += get_folder_size(pycache_dir)
                except (OSError, PermissionError):
                    pass
    
    print(f"  Cleaned: {cleaned_size / (1024**2):.2f} MB from Python cache")
    return cleaned_size

def get_folder_size(folder_path):
    """Get the size of a folder"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError):
        pass
    return total_size

def run_disk_cleanup():
    """Run Windows Disk Cleanup utility"""
    print("Running Windows Disk Cleanup...")
    try:
        # Run cleanmgr with /sagerun:1 for automatic cleanup
        subprocess.run(['cleanmgr', '/sagerun:1'], capture_output=True)
        print("  Windows Disk Cleanup completed")
        return True
    except Exception as e:
        print(f"  Could not run Disk Cleanup: {e}")
        return False

def main():
    """Main cleanup function"""
    print("Windows Disk Cleanup Script")
    print("=" * 50)
    
    # Check initial disk space
    initial_free = get_disk_usage()
    
    if initial_free < 1.0:  # Less than 1 GB free
        print("WARNING: Very low disk space detected!")
        print("This script will attempt to free up space safely.")
        print()
    
    total_cleaned = 0
    
    # Run cleanup operations
    print("Starting cleanup operations...")
    print("-" * 30)
    
    # Clean temporary files
    total_cleaned += clean_temp_files()
    
    # Clean browser caches
    total_cleaned += clean_browser_caches()
    
    # Clean Windows Update cache
    total_cleaned += clean_windows_update_cache()
    
    # Clean Python cache
    total_cleaned += clean_python_cache()
    
    # Empty Recycle Bin
    clean_recycle_bin()
    
    # Run Windows Disk Cleanup
    run_disk_cleanup()
    
    # Check final disk space
    print("\n" + "=" * 50)
    print("CLEANUP SUMMARY")
    print("=" * 50)
    
    final_free = get_disk_usage()
    space_freed = final_free - initial_free
    
    print(f"Space freed: {space_freed:.2f} GB")
    print(f"Total cleaned: {total_cleaned / (1024**3):.2f} GB")
    
    if final_free > 5.0:
        print("SUCCESS: Sufficient space available for virtual environments!")
    elif final_free > 1.0:
        print("WARNING: Some space freed, but still low. Consider additional cleanup.")
    else:
        print("CRITICAL: Still very low on space. Manual intervention needed.")
        print("Consider:")
        print("  - Uninstalling unused programs")
        print("  - Moving files to external storage")
        print("  - Using cloud storage for large files")
    
    print("\nCleanup complete!")

if __name__ == "__main__":
    main()

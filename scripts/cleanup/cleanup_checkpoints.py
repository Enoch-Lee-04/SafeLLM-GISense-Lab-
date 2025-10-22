#!/usr/bin/env python3
"""
Checkpoint Cleanup Script for Soundscape-to-Image Project
This script removes old training checkpoints, keeping only the latest 2-3 checkpoints
to free up significant disk space.
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime

def get_checkpoint_info(file_path):
    """Extract epoch number and file info from checkpoint filename"""
    filename = os.path.basename(file_path)
    
    # Extract epoch number from filename patterns like "imagen_1_30_epochs.pt"
    epoch_match = re.search(r'_(\d+)_epochs\.pt$', filename)
    if epoch_match:
        epoch = int(epoch_match.group(1))
        return {
            'epoch': epoch,
            'file_path': file_path,
            'filename': filename,
            'size_mb': os.path.getsize(file_path) / (1024 * 1024)
        }
    
    # Handle other checkpoint patterns
    if 'optimizer.pt' in filename:
        return {
            'epoch': 999999,  # Put optimizer at end
            'file_path': file_path,
            'filename': filename,
            'size_mb': os.path.getsize(file_path) / (1024 * 1024)
        }
    
    return None

def analyze_checkpoints(checkpoint_dir):
    """Analyze checkpoint directory and identify files to keep/remove"""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return [], []
    
    checkpoint_files = []
    
    # Find all checkpoint files
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            file_path = os.path.join(checkpoint_dir, file)
            checkpoint_info = get_checkpoint_info(file_path)
            if checkpoint_info:
                checkpoint_files.append(checkpoint_info)
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: x['epoch'])
    
    # Keep the latest 3 checkpoints (including optimizer if present)
    files_to_keep = checkpoint_files[-3:] if len(checkpoint_files) >= 3 else checkpoint_files
    files_to_remove = checkpoint_files[:-3] if len(checkpoint_files) > 3 else []
    
    return files_to_keep, files_to_remove

def cleanup_qwen_checkpoints(qwen_model_dir):
    """Clean up Qwen model checkpoints"""
    print("Analyzing Qwen model checkpoints...")
    
    checkpoint_dirs = []
    for item in os.listdir(qwen_model_dir):
        item_path = os.path.join(qwen_model_dir, item)
        if os.path.isdir(item_path) and item.startswith('checkpoint-'):
            checkpoint_dirs.append(item_path)
    
    if not checkpoint_dirs:
        print("No Qwen checkpoint directories found.")
        return 0
    
    # Sort checkpoint directories by number
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
    
    # Keep only the latest checkpoint directory
    files_to_remove = []
    total_size_to_free = 0
    
    for checkpoint_dir in checkpoint_dirs[:-1]:  # Remove all but the latest
        print(f"  Marking for removal: {os.path.basename(checkpoint_dir)}")
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                files_to_remove.append({
                    'path': file_path,
                    'size_mb': file_size,
                    'relative_path': os.path.relpath(file_path, qwen_model_dir)
                })
                total_size_to_free += file_size
    
    return files_to_remove, total_size_to_free

def cleanup_imagen_checkpoints(checkpoint_dir):
    """Clean up Imagen training checkpoints"""
    print("Analyzing Imagen checkpoints...")
    
    files_to_keep, files_to_remove = analyze_checkpoints(checkpoint_dir)
    
    print(f"  Found {len(files_to_keep + files_to_remove)} checkpoint files")
    print(f"  Keeping {len(files_to_keep)} latest checkpoints")
    print(f"  Removing {len(files_to_remove)} old checkpoints")
    
    if files_to_keep:
        print("  Files to keep:")
        for file_info in files_to_keep:
            print(f"    - {file_info['filename']} (epoch {file_info['epoch']}, {file_info['size_mb']:.1f} MB)")
    
    if files_to_remove:
        print("  Files to remove:")
        for file_info in files_to_remove:
            print(f"    - {file_info['filename']} (epoch {file_info['epoch']}, {file_info['size_mb']:.1f} MB)")
    
    return files_to_remove

def create_backup_log(removed_files, backup_log_path):
    """Create a log of removed files for potential recovery"""
    with open(backup_log_path, 'w') as f:
        f.write(f"Checkpoint Cleanup Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Files removed during cleanup:\n\n")
        
        for file_info in removed_files:
            f.write(f"File: {file_info['relative_path']}\n")
            f.write(f"Size: {file_info['size_mb']:.2f} MB\n")
            f.write(f"Removed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 40 + "\n")

def main():
    """Main cleanup function"""
    print("Checkpoint Cleanup Script for Soundscape-to-Image Project")
    print("=" * 60)
    
    project_dir = Path(__file__).parent.absolute()
    print(f"Project directory: {project_dir}")
    print()
    
    all_files_to_remove = []
    total_space_to_free = 0
    
    # Clean up Imagen checkpoints
    imagen_checkpoint_dir = os.path.join(project_dir, "checkpoints")
    if os.path.exists(imagen_checkpoint_dir):
        print("1. CLEANING UP IMAGEN CHECKPOINTS")
        print("-" * 40)
        
        imagen_files_to_remove = cleanup_imagen_checkpoints(imagen_checkpoint_dir)
        
        for file_info in imagen_files_to_remove:
            all_files_to_remove.append({
                'path': file_info['file_path'],
                'size_mb': file_info['size_mb'],
                'relative_path': os.path.relpath(file_info['file_path'], project_dir)
            })
            total_space_to_free += file_info['size_mb']
        
        print(f"  Total space to free from Imagen checkpoints: {total_space_to_free:.1f} MB")
        print()
    
    # Clean up Qwen checkpoints
    qwen_model_dir = os.path.join(project_dir, "qwen_safety_model")
    if os.path.exists(qwen_model_dir):
        print("2. CLEANING UP QWEN CHECKPOINTS")
        print("-" * 40)
        
        qwen_files_to_remove, qwen_space_to_free = cleanup_qwen_checkpoints(qwen_model_dir)
        
        for file_info in qwen_files_to_remove:
            all_files_to_remove.append(file_info)
        
        total_space_to_free += qwen_space_to_free
        print(f"  Total space to free from Qwen checkpoints: {qwen_space_to_free:.1f} MB")
        print()
    
    # Summary
    print("3. CLEANUP SUMMARY")
    print("-" * 40)
    print(f"Total files to remove: {len(all_files_to_remove)}")
    print(f"Total space to free: {total_space_to_free:.1f} MB ({total_space_to_free/1024:.1f} GB)")
    
    if not all_files_to_remove:
        print("No files need to be removed.")
        return
    
    # Auto-proceed with cleanup (removing user prompt)
    print()
    print("Proceeding with automatic cleanup...")
    
    # Create backup log
    backup_log_path = os.path.join(project_dir, "checkpoint_cleanup_log.txt")
    create_backup_log(all_files_to_remove, backup_log_path)
    print(f"Created backup log: {backup_log_path}")
    
    # Perform cleanup
    print("\n4. PERFORMING CLEANUP")
    print("-" * 40)
    
    removed_count = 0
    actual_space_freed = 0
    
    for file_info in all_files_to_remove:
        try:
            file_size = file_info['size_mb']
            os.remove(file_info['path'])
            removed_count += 1
            actual_space_freed += file_size
            print(f"  Removed: {file_info['relative_path']} ({file_size:.1f} MB)")
        except (OSError, PermissionError) as e:
            print(f"  Failed to remove {file_info['relative_path']}: {e}")
    
    print()
    print("5. CLEANUP COMPLETE")
    print("-" * 40)
    print(f"Files removed: {removed_count}/{len(all_files_to_remove)}")
    print(f"Space freed: {actual_space_freed:.1f} MB ({actual_space_freed/1024:.1f} GB)")
    print(f"Backup log created: {backup_log_path}")
    
    if actual_space_freed > 0:
        print("\nSUCCESS: Significant space has been freed up!")
        print("You can now proceed with virtual environment setup.")
    else:
        print("\nNo space was freed. Check file permissions or try running as administrator.")

if __name__ == "__main__":
    main()

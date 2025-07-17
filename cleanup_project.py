#!/usr/bin/env python3
"""
StockSage.AI Project Cleanup Script

This script helps clean up unnecessary files from your project directory
and removes them from Git tracking before pushing to GitHub.

Usage:
    python cleanup_project.py --dry-run    # Preview what will be deleted
    python cleanup_project.py --execute    # Actually perform the cleanup
"""

import os
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import List, Set

def get_git_tracked_files() -> Set[str]:
    """Get list of files currently tracked by Git"""
    try:
        result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True, check=True)
        return set(result.stdout.strip().split('\n'))
    except subprocess.CalledProcessError:
        print("❌ Error: Not in a Git repository or Git not available")
        return set()

def get_files_to_remove() -> List[Path]:
    """Define patterns and directories of files to remove"""
    
    files_to_remove = []
    
    # Root directory .md files (documentation clutter)
    root_md_patterns = [
        "FIXES_IMPLEMENTED.md",
        "IMPLEMENTATION_COMPLETION_SUMMARY.md", 
        "MODEL_PERFORMANCE_IMPROVEMENTS.md",
        "ERROR_FIXES_SUMMARY.md",
        "MODEL_IMPROVEMENTS_SUMMARY.md",
        "EVALUATE_MODELS_FIX_ANALYSIS.md",
        "DATA_VALIDATION_README.md",
        "PHASE_4_COMPLETION.md",
        "STEP_8_COMPLETION_SUMMARY.md",
        "TRAINING_GUIDE.md",
        "future_enhancements.md",
        "project_structure.md", 
        "stock_genai_blueprint.md",
        "stock_sage_rules.mdc"
    ]
    
    for pattern in root_md_patterns:
        file_path = Path(pattern)
        if file_path.exists():
            files_to_remove.append(file_path)
    
    # CSV files in root
    csv_files = [
        "unified_dataset_sample.csv"
    ]
    
    for csv_file in csv_files:
        file_path = Path(csv_file)
        if file_path.exists():
            files_to_remove.append(file_path)
    
    # Entire directories to remove
    directories_to_remove = [
        "reports/",
        "results/",
        "logs/",
        ".pytest_cache/"
    ]
    
    for dir_path in directories_to_remove:
        dir_path_obj = Path(dir_path)
        if dir_path_obj.exists():
            files_to_remove.append(dir_path_obj)
    
    return files_to_remove

def get_important_files_to_keep() -> List[str]:
    """Define important files that should NEVER be deleted"""
    return [
        # Core Python files
        "requirements.txt",
        "requirements-core.txt",
        "src/",
        "scripts/",
        "tests/",
        "notebooks/",
        "config/",
        
        # Git and GitHub
        ".git/",
        ".github/",
        ".gitignore",
        
        # Important documentation
        "README.md",
        "README_QUICKSTART.md",
        
        # Virtual environment
        ".venv/",
        
        # Data directories (structure, not content)
        "data/",
    ]

def remove_from_git(file_path: Path) -> bool:
    """Remove file from Git tracking"""
    try:
        if file_path.is_dir():
            result = subprocess.run(['git', 'rm', '-r', '--cached', str(file_path)], 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(['git', 'rm', '--cached', str(file_path)], 
                                  capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False

def remove_file_or_dir(file_path: Path) -> bool:
    """Remove file or directory from filesystem"""
    try:
        if file_path.is_dir():
            shutil.rmtree(file_path)
            print(f"  🗂️  Removed directory: {file_path}")
        else:
            file_path.unlink()
            print(f"  📄 Removed file: {file_path}")
        return True
    except (OSError, PermissionError) as e:
        print(f"  ❌ Failed to remove {file_path}: {e}")
        return False

def calculate_size(file_path: Path) -> int:
    """Calculate total size of file or directory in bytes"""
    if file_path.is_file():
        return file_path.stat().st_size
    elif file_path.is_dir():
        total = 0
        try:
            for item in file_path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        except (OSError, PermissionError):
            pass
        return total
    return 0

def format_size(size_bytes: float) -> str:
    """Format bytes into human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def main():
    parser = argparse.ArgumentParser(description='Clean up StockSage.AI project files')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--execute', action='store_true',
                       help='Actually perform the cleanup')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.execute:
        print("❓ Please specify either --dry-run or --execute")
        parser.print_help()
        return
    
    print("🧹 StockSage.AI Project Cleanup")
    print("=" * 50)
    
    # Get files to remove
    files_to_remove = get_files_to_remove()
    git_tracked = get_git_tracked_files()
    
    if not files_to_remove:
        print("✅ No unnecessary files found!")
        return
    
    # Calculate total size
    total_size = sum(calculate_size(f) for f in files_to_remove if f.exists())
    
    print(f"\n📊 Found {len(files_to_remove)} items to clean up")
    print(f"💾 Total size to be freed: {format_size(total_size)}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Files that would be removed:")
        for file_path in files_to_remove:
            if file_path.exists():
                size = calculate_size(file_path)
                file_type = "📁 DIR " if file_path.is_dir() else "📄 FILE"
                tracked = "🔗 (Git tracked)" if str(file_path) in git_tracked else "🔄 (Local only)"
                print(f"  {file_type} {file_path} - {format_size(size)} {tracked}")
        
        print(f"\n⚠️  This is a DRY RUN. Run with --execute to actually remove files.")
        return
    
    if args.execute:
        print("\n🚀 EXECUTING CLEANUP...")
        
        removed_count = 0
        freed_size = 0
        
        for file_path in files_to_remove:
            if not file_path.exists():
                continue
                
            size = calculate_size(file_path)
            
            # Remove from Git if tracked
            if str(file_path) in git_tracked:
                if remove_from_git(file_path):
                    print(f"  🔗 Removed from Git: {file_path}")
                else:
                    print(f"  ⚠️  Could not remove from Git: {file_path}")
            
            # Remove from filesystem
            if remove_file_or_dir(file_path):
                removed_count += 1
                freed_size += size
        
        print(f"\n✅ Cleanup completed!")
        print(f"📊 Removed {removed_count} items")
        print(f"💾 Freed up {format_size(freed_size)} of space")
        
        # Check if there are changes to commit
        try:
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, check=True)
            if result.stdout.strip():
                print("\n🔄 Git changes detected. You can now commit and push:")
                print("   git add .")
                print("   git commit -m 'Clean up unnecessary files and improve .gitignore'")
                print("   git push origin dev")  # Using dev branch per user's memory
            else:
                print("\n✅ No Git changes to commit.")
        except subprocess.CalledProcessError:
            print("\n⚠️  Could not check Git status")

if __name__ == "__main__":
    main() 
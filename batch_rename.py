#!/usr/bin/env python3
"""
Batch rename files and folders in inputs_outputs directory.
Renames all occurrences of --input pattern to --output pattern.
"""

import argparse
import os
import sys
from pathlib import Path


def rename_files_and_folders(directory, input_pattern, output_pattern, dry_run=False):
    """
    Recursively rename files and folders that contain input_pattern.
    
    Args:
        directory: Directory to search in
        input_pattern: Pattern to find and replace
        output_pattern: Pattern to replace with
        dry_run: If True, only show what would be renamed without actually renaming
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist.", file=sys.stderr)
        return False
    
    renamed_count = 0
    errors = []
    
    # Collect all paths first (files and directories), sorted by depth (deepest first)
    # This ensures we rename children before parents
    all_paths = []
    for root, dirs, files in os.walk(directory, topdown=False):
        root_path = Path(root)
        
        # Add files first
        for file in files:
            all_paths.append(root_path / file)
        
        # Add directories
        for dir_name in dirs:
            all_paths.append(root_path / dir_name)
    
    # Process paths (deepest first, so we rename children before parents)
    for path in all_paths:
        if input_pattern in path.name:
            new_name = path.name.replace(input_pattern, output_pattern)
            new_path = path.parent / new_name
            
            # Skip if the new name is the same as the old name
            if new_path == path:
                continue
            
            # Check if target already exists
            if new_path.exists():
                print(f"Warning: '{new_path}' already exists. Skipping '{path}'")
                continue
            
            try:
                if dry_run:
                    print(f"Would rename: '{path}' -> '{new_path.name}'")
                else:
                    path.rename(new_path)
                    print(f"Renamed: '{path}' -> '{new_path.name}'")
                renamed_count += 1
            except Exception as e:
                error_msg = f"Error renaming '{path}': {e}"
                errors.append(error_msg)
                print(error_msg, file=sys.stderr)
    
    # Summary
    if dry_run:
        print(f"\nDry run complete. Would rename {renamed_count} item(s).")
    else:
        print(f"\nRenamed {renamed_count} item(s).")
    
    if errors:
        print(f"\n{len(errors)} error(s) occurred.", file=sys.stderr)
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch rename files and folders in inputs_outputs directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rename all occurrences of "z3mm_objects" to "z5mm_objects"
  python batch_rename.py --input z3mm_objects --output z5mm_objects
  
  # Dry run to preview changes
  python batch_rename.py --input z3mm_objects --output z5mm_objects --dry-run
  
  # Rename files in a different directory
  python batch_rename.py --input old_name --output new_name --dir custom/path
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input pattern to find and replace in filenames'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output pattern to replace with'
    )
    
    parser.add_argument(
        '--dir',
        default='inputs_outputs',
        help='Directory to search in (default: inputs_outputs)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without actually renaming'
    )
    
    args = parser.parse_args()
    
    if args.input == args.output:
        print("Error: --input and --output cannot be the same.", file=sys.stderr)
        sys.exit(1)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be renamed\n")
    
    success = rename_files_and_folders(
        args.dir,
        args.input,
        args.output,
        dry_run=args.dry_run
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Import Migration Script - xyzcalc to calibration.utils
------------------------------------------------------
Migrates all imports from lib.xyzcalc to lib.calibration.utils.
This script updates all Python files that import from the duplicate xyzcalc module.

Run this before removing lib/xyzcalc/ directory.
"""

import re
import os
from pathlib import Path
from typing import List, Tuple

def find_files_to_migrate(root_dir: Path) -> List[Path]:
    """Find all Python files that import from lib.calibration.utils."""
    python_files = []
    
    # Skip venv and __pycache__
    skip_dirs = {'venv', '__pycache__', '.git', 'old'}
    
    for py_file in root_dir.rglob("*.py"):
        # Skip if in excluded directories
        if any(skip_dir in py_file.parts for skip_dir in skip_dirs):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'lib.xyzcalc' in content or 'from lib.xyzcalc' in content or 'import lib.calibration.utils' in content:
                    python_files.append(py_file)
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")
    
    return python_files

def migrate_imports(file_path: Path) -> Tuple[bool, int]:
    """
    Migrate imports in a single file.
    Returns: (success, changes_count)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes = 0
        
        # Pattern 1: from lib.calibration.utils import ... or from lib.calibration.utils.calibration_utils import ...
        pattern1 = r'from lib\.xyzcalc(?:\.calibration_utils)? import (.*?)(?:\n|$)'
        def replacer1(match):
            nonlocal changes
            imports = match.group(1)
            changes += 1
            return f'from lib.calibration.utils import {imports}\n'
        
        content = re.sub(pattern1, replacer1, content, flags=re.MULTILINE)
        
        # Pattern 2: import lib.calibration.utils or import lib.calibration.utils
        pattern2 = r'import lib\.xyzcalc(?:\.calibration_utils)?'
        def replacer2(match):
            nonlocal changes
            changes += 1
            return 'import lib.calibration.utils'
        
        content = re.sub(pattern2, replacer2, content, flags=re.MULTILINE)
        
        # Pattern 3: lib.calibration.utils. in code (if any)
        pattern3 = r'lib\.xyzcalc\.'
        def replacer3(match):
            nonlocal changes
            changes += 1
            return 'lib.calibration.utils.'
        
        content = re.sub(pattern3, replacer3, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes
        
        return False, 0
        
    except Exception as e:
        print(f"Error migrating {file_path}: {e}")
        return False, 0

def main():
    """Main migration function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print("Finding files with lib.xyzcalc imports...")
    files_to_migrate = find_files_to_migrate(project_root)
    
    if not files_to_migrate:
        print("No files found with lib.xyzcalc imports.")
        return
    
    print(f"\nFound {len(files_to_migrate)} file(s) to migrate:\n")
    for f in files_to_migrate:
        print(f"  - {f.relative_to(project_root)}")
    
    print("\nMigrating imports...")
    total_changes = 0
    success_count = 0
    
    for file_path in files_to_migrate:
        success, changes = migrate_imports(file_path)
        if success:
            success_count += 1
            total_changes += changes
            rel_path = file_path.relative_to(project_root)
            print(f"[OK] {rel_path}: {changes} import(s) updated")
    
    print(f"\nMigration complete: {success_count}/{len(files_to_migrate)} files migrated, {total_changes} total changes.")
    print("\nNext steps:")
    print("1. Verify the imports work correctly")
    print("2. Ensure lib/calibration/utils.py contains all functions from lib/xyzcalc/calibration_utils.py")
    print("3. Update lib/calibration/__init__.py to export the functions")
    print("4. Delete lib/xyzcalc/ directory")

if __name__ == "__main__":
    main()

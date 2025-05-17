import os
from pathlib import Path

def print_project_structure(startpath='.', max_level=2):
    """Print project structure limited to specified depth"""
    startpath = Path(startpath)
    print(f"Project Structure (max depth: {max_level})")
    print(f"{startpath.name}/")
    
    for root, dirs, files in os.walk(startpath):
        # Skip virtual environment directories
        dirs[:] = [d for d in dirs if not d.startswith(('venv', '.', '__'))]
        
        level = root.replace(str(startpath), '').count(os.sep)
        if level >= max_level:
            continue
            
        indent = '│   ' * level
        print(f"{indent}├── {os.path.basename(root)}/")
        
        subindent = '│   ' * (level + 1)
        for f in files[:10]:  # Limit to 10 files per directory
            if level < max_level - 1 or root == str(startpath):
                print(f"{subindent}├── {f}")
        if len(files) > 10:
            print(f"{subindent}└── ...{len(files)-10} more files")

# Usage - will show 2 levels deep from current directory
print_project_structure(max_level=2)
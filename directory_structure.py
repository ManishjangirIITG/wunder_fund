import os
from pathlib import Path

# Directories and files to exclude from the structure
IGNORE_DIRS = {
    '.git', '.idea', '.vscode', '__pycache__', 'venv', '.venv', 
    'env', 'node_modules', 'build', 'dist', '.pytest_cache'
}
IGNORE_FILES = {'.DS_Store', 'directory_structure.txt'}

def generate_tree(dir_path: Path, prefix: str = "") -> list[str]:
    """Recursively generates tree lines for directory structure."""
    lines = []
    
    # Get filtered list of contents
    try:
        entries = sorted(
            [e for e in dir_path.iterdir() if e.name not in IGNORE_DIRS and e.name not in IGNORE_FILES],
            key=lambda e: (not e.is_dir(), e.name.lower()) # Directories first
        )
    except PermissionError:
        return [prefix + "└── [Permission Denied]"]

    count = len(entries)
    for i, entry in enumerate(entries):
        is_last = (i == count - 1)
        connector = "└── " if is_last else "├── "
        
        lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
        
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            lines.extend(generate_tree(entry, prefix + extension))
            
    return lines

def main():
    # Target parent directory of the current working directory
    target_dir = Path.cwd()
    output_file = Path.cwd() / "directory_structure.txt"
    
    print(f"Scanning directory: {target_dir}")
    
    tree_lines = [f"{target_dir.name}/"] + generate_tree(target_dir)
    tree_content = "\n".join(tree_lines)
    
    # Save to text file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(tree_content)
        
    print(f"Directory structure successfully saved to: {output_file.name}")

if __name__ == "__main__":
    main()
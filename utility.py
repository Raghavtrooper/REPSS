import sys
import os


""""
Adds the directory where this script resides (assumed project root) to sys.path.
Ensures internal module imports (e.g., shared/, api/, etc.) work correctly.
"""
def root_to_sys(caller: str):

    current_script_dir = os.path.dirname(os.path.abspath(caller))
    project_root = current_script_dir

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    return project_root
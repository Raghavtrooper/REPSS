import sys
import os

# --- Add project root to sys.path ---
# Get the absolute path of the directory where this script (run_etl.py) resides
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# The current_script_dir is already the project root
project_root = current_script_dir 

# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End sys.path modification ---

# Now, import the main function from your ETL module
# This import will now work because the project root is in sys.path
from etl.main_etl import main

if __name__ == "__main__":
    print(f"Starting ETL process from: {project_root}")
    main()

import sys
from utility import root_to_sys

# --- Add project root to sys.path ---
project_root = root_to_sys(__file__)

# Now, import the main function from your ETL module
# This import will now work because the project root is in sys.path
from etl.main_etl import main

if __name__ == "__main__":
    print(f"Starting ETL process from: {project_root}")
    main()

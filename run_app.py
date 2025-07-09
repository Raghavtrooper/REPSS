import sys
import subprocess
from utility import root_to_sys

# --- Add project root to sys.path ---
project_root = root_to_sys(__file__)

if __name__ == "__main__":
    print(f"Launching terminal app from: {project_root}")
    
    # Define the command to run the terminal-based main_app.py
    # We use 'sys.executable' to ensure the correct Python interpreter is used
    command = [sys.executable, "app/main_app.py"]
    
    try:
        # Use subprocess.run to execute the command
        # check=True will raise an exception if the command returns a non-zero exit code
        # shell=False is generally safer and recommended for commands as a list
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("\nError: Python interpreter or app/main_app.py not found.")
        print("Please ensure Python is installed and the path to app/main_app.py is correct.")
    except subprocess.CalledProcessError as e:
        print(f"\nError launching terminal app. Command failed with exit code {e.returncode}")
        print(f"Error output:\n{e.stderr.decode()}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


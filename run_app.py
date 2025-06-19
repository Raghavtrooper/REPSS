import sys
import os
import subprocess

# --- Add project root to sys.path ---
# Get the absolute path of the directory where this script (run_app.py) resides
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# The current_script_dir is already the project root
project_root = current_script_dir 

# Add the project root to sys.path if it's not already there
# This is crucial for Streamlit to find 'shared' and other internal modules
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End sys.path modification ---

if __name__ == "__main__":
    print(f"Launching Streamlit app from: {project_root}")
    
    # Define the command to run Streamlit
    # We use 'sys.executable' to ensure the correct Python interpreter (e.g., from your Anaconda env) is used
    # And 'streamlit' is usually a script/module within that interpreter's environment
    command = [sys.executable, "-m", "streamlit", "run", "app/main_app.py"]
    
    try:
        # Use subprocess.run to execute the command
        # check=True will raise an exception if the command returns a non-zero exit code
        # shell=False is generally safer and recommended for commands as a list
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("\nError: Streamlit command not found.")
        print("Please ensure Streamlit is installed in your active Python environment.")
        print("You can install it using: pip install streamlit")
    except subprocess.CalledProcessError as e:
        print(f"\nError launching Streamlit app. Command failed with exit code {e.returncode}")
        print(f"Error output:\n{e.stderr.decode()}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


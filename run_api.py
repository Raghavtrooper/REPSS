import sys
import subprocess
from utility import root_to_sys

# --- Add project root to sys.path ---
project_root = root_to_sys(__file__)


if __name__ == "__main__":
    print(f"Launching Fast-API server from: {project_root}")
    
    command = [sys.executable, "-m", "uvicorn", "api.rest_controller:app", "--reload"]
    
    try:
        # executing the command
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("Please ensure Fast-API and uvicorn is installed in your active Python environment.")
        print("You can install it using: pip install fastapi uvicorn")
    except subprocess.CalledProcessError as e:
        print(f"\nError launching Fast-API app. Command failed with exit code {e.returncode}")
        print(f"Error output:\n{e.stderr.decode()}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
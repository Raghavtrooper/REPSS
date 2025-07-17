import os
import shutil
import re
from pathlib import Path

# Input/output directories
INPUT_DIR = r"E:\projects\nasns\input"
OUTPUT_DIR = r"E:\projects\nasns\output"

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.doc'}

def is_allowed_file(file_path):
    return (
        file_path.suffix.lower() in ALLOWED_EXTENSIONS and
        not file_path.name.startswith("~$")
    )

def normalize_filename(filename):
    name, ext = os.path.splitext(filename)
    name = re.sub(r'[\s_\-]*\(?copy\)?[\s_\-]*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[\s_\-]*\(\d+\)$', '', name)
    name = re.sub(r'[\s_\-]*\d+$', '', name)
    return name.lower() + ext.lower()

def copy_files(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seen_normalized_names = set()
    existing_output_normalized_names = {
        normalize_filename(f.name)
        for f in output_path.iterdir()
        if f.is_file() and is_allowed_file(f)
    }

    total_files = 0
    copied_files = 0
    skipped_duplicates = 0
    skipped_locked = 0

    for root, dirs, files in os.walk(input_path):
        for file in files:
            file_path = Path(root) / file

            if not is_allowed_file(file_path):
                if file.startswith("~$"):
                    skipped_locked += 1
                continue

            total_files += 1
            normalized_name = normalize_filename(file)

            if normalized_name in seen_normalized_names or normalized_name in existing_output_normalized_names:
                skipped_duplicates += 1
                print(f"Skipped duplicate: {file_path}")
                continue

            seen_normalized_names.add(normalized_name)

            dest_file_path = output_path / file
            if dest_file_path.exists():
                dest_file_path = output_path / get_unique_filename(output_path, file)

            shutil.copy2(file_path, dest_file_path)
            copied_files += 1
            print(f"Copied: {file_path} -> {dest_file_path}")

    # Final summary
    print("\n===== Summary =====")
    print(f"Total allowed files scanned: {total_files}")
    print(f"Files copied:                {copied_files}")
    print(f"Duplicates skipped:          {skipped_duplicates}")
    print(f"Lock files skipped:          {skipped_locked}")

def get_unique_filename(dest_dir, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(dest_dir, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename

if __name__ == "__main__":
    copy_files(INPUT_DIR, OUTPUT_DIR)

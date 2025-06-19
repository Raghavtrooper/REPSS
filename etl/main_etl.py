import os
import json
import shutil 
import pandas as pd # Import pandas for gold layer transformation

# Import modules from the current project structure
from shared.config import (
    RAW_RESUMES_DIR, OUTPUT_JSON_FILE, GOLD_LAYER_PARQUET_FILE, CHROMA_DB_DIR, OLLAMA_MODEL,
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE,
    MINIO_BUCKET_NAME, MINIO_OBJECT_NAME # MINIO_OBJECT_NAME will now be the parquet file
)
from etl.data_loader import load_document_content
from etl.llm_extractor import initialize_ollama_llm, get_extraction_prompt, process_resume_file
from etl.gold_layer_transformer import transform_to_gold_layer # Import the new transformer
from etl.chroma_builder import generate_chroma_db
from etl.minio_utils import get_minio_client, upload_to_minio

def main():
    print("Starting structured data extraction and embedding generation (supports PDF, DOCX, TXT)...")

    # Ensure raw_resumes directory exists and possibly add a placeholder if empty
    if not os.path.exists(RAW_RESUMES_DIR):
        os.makedirs(RAW_RESUMES_DIR)
        print(f"Created directory: {RAW_RESUMES_DIR}")
        with open(os.path.join(RAW_RESUMES_DIR, "example_resume.txt"), "w", encoding="utf-8") as f:
            f.write("John Doe. Software Engineer with 5 years experience in Python and AWS. Graduated from ABC University. Contact: john.doe@example.com.")
        print("Created a placeholder 'example_resume.txt'. Please add your actual raw resume files (.pdf, .docx, .txt) to this directory.")
        print("Exiting. Run again after adding your resume files.")
        exit()

    # Step 1: Initialize LLM and Prompt
    try:
        llm_for_extraction = initialize_ollama_llm(OLLAMA_MODEL)
        extraction_prompt = get_extraction_prompt()
    except Exception as e:
        print(f"Initialization failed: {e}")
        exit()

    # Get a set of current resume filenames in the raw_resumes directory
    current_raw_resume_filenames = set()
    for f in os.listdir(RAW_RESUMES_DIR):
        if f.lower().endswith(('.pdf', '.docx', '.txt')):
            current_raw_resume_filenames.add(f)
    
    # Handle case where raw_resumes directory becomes empty
    if not current_raw_resume_filenames:
        print(f"\nNo .pdf, .docx, or .txt resume files found in '{RAW_RESUMES_DIR}'.")
        print("Existing profiles and ChromaDB will be cleared as no source files are present.")
        
        # Clear existing JSON
        if os.path.exists(OUTPUT_JSON_FILE):
            os.remove(OUTPUT_JSON_FILE)
            print(f"Removed '{OUTPUT_JSON_FILE}'.")
        # Clear existing Parquet
        if os.path.exists(GOLD_LAYER_PARQUET_FILE):
            os.remove(GOLD_LAYER_PARQUET_FILE)
            print(f"Removed '{GOLD_LAYER_PARQUET_FILE}'.")
        
        # Ensure ChromaDB is cleared if no resumes are present
        if os.path.exists(CHROMA_DB_DIR):
            print(f"Clearing existing ChromaDB directory: {CHROMA_DB_DIR}")
            shutil.rmtree(CHROMA_DB_DIR)
        
        print("Offline processing finished. No employee profiles available.")
        exit()


    # Step 2: Load existing structured profiles (Silver Layer) and identify those to keep
    existing_profiles_from_json = []
    if os.path.exists(OUTPUT_JSON_FILE):
        print(f"\nLoading existing structured profiles from '{OUTPUT_JSON_FILE}' (Silver Layer)...")
        with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            try:
                existing_profiles_from_json = json.load(f)
                print(f"Loaded {len(existing_profiles_from_json)} profiles from JSON.")
            except json.JSONDecodeError as e:
                print(f"WARNING: Could not decode existing JSON file '{OUTPUT_JSON_FILE}'. It might be corrupted. Starting fresh. Error: {e}")
            except Exception as e:
                print(f"WARNING: An unexpected error occurred while loading existing JSON. Starting fresh. Error: {e}")
    else:
        print(f"\n'{OUTPUT_JSON_FILE}' not found. Will create a new one.")

    # Filter existing profiles: keep only those whose original file still exists
    profiles_to_keep = []
    processed_filenames_in_json = set() # To quickly check which filenames are in the loaded JSON
    for profile in existing_profiles_from_json:
        original_filename = profile.get('_original_filename')
        if original_filename and original_filename in current_raw_resume_filenames:
            profiles_to_keep.append(profile)
            processed_filenames_in_json.add(original_filename)
        else:
            if original_filename:
                print(f"  Removed profile for deleted resume: {original_filename}")
            else:
                print(f"  Removed profile with no original filename tracking or source missing: {profile.get('name', 'Unknown')}")
    
    print(f"Keeping {len(profiles_to_keep)} profiles whose source files still exist.")

    # Step 3: Identify and process new resumes
    new_resume_files_to_process = []
    for filename in current_raw_resume_filenames:
        if filename not in processed_filenames_in_json:
            new_resume_files_to_process.append(os.path.join(RAW_RESUMES_DIR, filename))

    newly_extracted_profiles = []
    if new_resume_files_to_process:
        print(f"\nProcessing {len(new_resume_files_to_process)} new or previously unprocessed resumes...")
        for file_path in new_resume_files_to_process:
            structured_data = process_resume_file(file_path, llm_for_extraction, extraction_prompt, load_document_content)
            if structured_data:
                structured_data['_original_filename'] = os.path.basename(file_path) 
                newly_extracted_profiles.append(structured_data)
        print(f"Successfully extracted {len(newly_extracted_profiles)} new profiles.")
    else:
        print("\nNo new resumes detected to process.")

    # Step 4: Consolidate all silver-layer profiles
    all_structured_profiles_final_silver = profiles_to_keep + newly_extracted_profiles

    if not all_structured_profiles_final_silver:
        print("No structured profiles (kept or newly extracted) available to proceed. Exiting.")
        exit()

    # Step 5: Save ALL (kept + new) Structured Data to JSON (Silver Layer)
    print(f"\nSaving {len(all_structured_profiles_final_silver)} combined structured profiles to '{OUTPUT_JSON_FILE}' (Silver Layer)...")
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_structured_profiles_final_silver, f, indent=2, ensure_ascii=False)
    print("Silver Layer data saved.")

    # --- NEW STEP: Gold Layer Transformation ---
    print("\nStarting Gold Layer Transformation...")
    gold_df = transform_to_gold_layer(all_structured_profiles_final_silver)
    
    if gold_df.empty:
        print("Gold layer transformation resulted in an empty DataFrame. Exiting.")
        exit()

    print(f"Saving {len(gold_df)} gold layer profiles to '{GOLD_LAYER_PARQUET_FILE}'...")
    gold_df.to_parquet(GOLD_LAYER_PARQUET_FILE, index=False)
    print("Gold Layer data saved as Parquet.")

    # --- End Gold Layer Transformation ---

    # Step 6: Upload Gold Layer Parquet to MinIO
    print(f"\nUploading '{GOLD_LAYER_PARQUET_FILE}' to MinIO bucket '{MINIO_BUCKET_NAME}' as '{MINIO_OBJECT_NAME}'...")
    try:
        minio_client = get_minio_client(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE)
        # Ensure we are uploading the parquet file
        upload_to_minio(minio_client, MINIO_BUCKET_NAME, MINIO_OBJECT_NAME, GOLD_LAYER_PARQUET_FILE)
        print("Upload complete.")
    except Exception as e:
        print(f"MinIO operation failed: {e}")
        print("Please ensure your MinIO server is running and accessible with the correct credentials in .env.")
        exit()

    # Step 7: Generate and Persist ChromaDB from Gold Layer Profiles
    # ChromaDB expects a list of dictionaries, so convert DataFrame records back to list of dicts
    print("\nGenerating and persisting ChromaDB from Gold Layer profiles...")
    all_structured_profiles_final_gold = gold_df.to_dict(orient='records')
    generate_chroma_db(all_structured_profiles_final_gold, CHROMA_DB_DIR)
    print("ChromaDB generation complete.")

    print("\nOffline processing finished. You can now run your Streamlit app.")
    print(f"Remember to use `python run_app.py` from the project root.")

if __name__ == "__main__":
    main()

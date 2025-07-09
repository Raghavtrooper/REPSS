import os
import json
import shutil
import pandas as pd # Import pandas for gold layer transformation
from minio.error import S3Error # Import S3Error for specific error handling

# Import modules from the current project structure
from shared.config import (
    RAW_RESUMES_DIR, OUTPUT_JSON_FILE, GOLD_LAYER_PARQUET_FILE, OLLAMA_MODEL,
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE,
    MINIO_BUCKET_NAME, MINIO_OBJECT_NAME, # MINIO_OBJECT_NAME will now be the parquet file
    QDRANT_COLLECTION_NAME, # Import Qdrant collection name
    RAW_RESUMES_BUCKET_NAME, # New: MinIO bucket for raw resumes
    SILVER_LAYER_BUCKET_NAME, # New: MinIO bucket for silver layer JSON
    REJECTED_PROFILES_BUCKET_NAME # New: MinIO bucket for rejected profiles
)
from etl.data_loader import load_document_content
from etl.llm_extractor import initialize_ollama_llm, get_extraction_prompt, process_resume_file
from etl.gold_layer_transformer import transform_to_gold_layer # Import the new transformer
from etl.qdrant_builder import generate_qdrant_db # ADD THIS IMPORT
from etl.minio_utils import get_minio_client, upload_to_minio, download_from_minio # Import new download function

def main():
    print("Starting structured data extraction and embedding generation (supports PDF, DOCX, TXT)....\n")

    # Initialize MinIO client
    try:
        minio_client = get_minio_client(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE)
    except Exception as e:
        print(f"MinIO client initialization failed: {e}")
        print("Please ensure your MinIO server is running and accessible with the correct credentials in .env.")
        exit()

    # Ensure raw_resumes directory exists locally for temporary downloads
    os.makedirs(RAW_RESUMES_DIR, exist_ok=True)

    # Step 1: Download raw resumes from MinIO bucket
    print(f"Downloading raw resumes from MinIO bucket '{RAW_RESUMES_BUCKET_NAME}' to '{RAW_RESUMES_DIR}'...")
    raw_resume_objects = []
    try:
        # List objects in the raw resumes bucket
        objects = minio_client.list_objects(RAW_RESUMES_BUCKET_NAME, recursive=True)
        for obj in objects:
            if obj.is_dir: # Skip directories
                continue
            file_extension = os.path.splitext(obj.object_name)[1].lower()
            if file_extension in ('.pdf', '.docx', '.txt'):
                local_file_path = os.path.join(RAW_RESUMES_DIR, os.path.basename(obj.object_name))
                download_from_minio(minio_client, RAW_RESUMES_BUCKET_NAME, obj.object_name, local_file_path)
                raw_resume_objects.append(obj.object_name)
        
        if not raw_resume_objects:
            print(f"Error: MinIO bucket '{RAW_RESUMES_BUCKET_NAME}' is empty or contains no supported files.")
            print("Please upload your raw resume files (.pdf, .docx, .txt) to this bucket.")
            print("Exiting ETL process.")
            exit()
        print(f"Downloaded {len(raw_resume_objects)} raw resumes.")

    except S3Error as e: # Catch S3Error specifically
        print(f"MinIO S3 Error during raw resume download: {e}")
        print("Please ensure the rawresumes bucket exists and is accessible.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during raw resume download: {e}")
        exit()

    # Step 2: Initialize LLM and Prompt
    try:
        llm_for_extraction = initialize_ollama_llm(OLLAMA_MODEL)
        extraction_prompt = get_extraction_prompt()
    except Exception as e:
        print(f"Initialization failed: {e}")
        exit()

    # Get a set of current resume filenames (downloaded from MinIO)
    current_raw_resume_filenames = set(os.listdir(RAW_RESUMES_DIR))

    # Step 3: Load existing structured profiles (Silver Layer) from MinIO and identify those to keep
    existing_profiles_from_json = []
    silver_layer_object_name = os.path.basename(OUTPUT_JSON_FILE) # e.g., "structured_profiles.json"
    temp_silver_layer_local_path = OUTPUT_JSON_FILE # Use the same path for local temp file

    print(f"Attempting to download existing silver layer JSON from '{SILVER_LAYER_BUCKET_NAME}/{silver_layer_object_name}'...")
    try:
        download_from_minio(minio_client, SILVER_LAYER_BUCKET_NAME, silver_layer_object_name, temp_silver_layer_local_path)
        with open(temp_silver_layer_local_path, 'r', encoding='utf-8') as f:
            try:
                existing_profiles_from_json = json.load(f)
                print(f"Loaded {len(existing_profiles_from_json)} profiles from existing Silver Layer JSON in MinIO.")
            except json.JSONDecodeError as e:
                print(f"WARNING: Could not decode existing JSON file from MinIO. It might be corrupted. Starting fresh. Error: {e}")
            except Exception as e:
                print(f"WARNING: An unexpected error occurred while loading existing JSON from MinIO. Starting fresh. Error: {e}")
    except S3Error as e: # Catch S3Error specifically
        if "NoSuchKey" in str(e): # Specific error for object not found
            print(f"'{silver_layer_object_name}' not found in '{SILVER_LAYER_BUCKET_NAME}'. Will create a new one.")
        else:
            print(f"MinIO S3 Error during silver layer JSON download: {e}")
            print("Proceeding without existing silver layer data.")
    except Exception as e:
        print(f"An unexpected error occurred while trying to download silver layer JSON: {e}")
        print("Proceeding without existing silver layer data.")


    # Filter existing profiles: keep only those whose original file still exists in the downloaded raw resumes
    profiles_to_keep = []
    processed_filenames_in_json = set() # To quickly check which filenames are in the loaded JSON
    for profile in existing_profiles_from_json:
        original_filename = profile.get('_original_filename')
        if original_filename and original_filename in current_raw_resume_filenames:
            profiles_to_keep.append(profile)
            processed_filenames_in_json.add(original_filename)
        else:
            if original_filename:
                print(f"  Removed profile for deleted or non-downloaded resume: {original_filename}")
            else:
                print(f"  Removed profile with no original filename tracking or source missing: {profile.get('name', 'Unknown')}")
    
    print(f"Keeping {len(profiles_to_keep)} profiles whose source files still exist locally.")

    # Step 4: Identify and process new resumes
    new_resume_files_to_process = []
    for filename in current_raw_resume_filenames:
        if filename not in processed_filenames_in_json:
            new_resume_files_to_process.append(os.path.join(RAW_RESUMES_DIR, filename))

    newly_extracted_profiles = []
    if new_resume_files_to_process:
        print(f"\nProcessing {len(new_resume_files_to_process)} new or previously unprocessed resumes...")
        for file_path in new_resume_files_to_process:
            # load_document_content now returns (content_as_string, metadata_dict)
            content, doc_metadata = load_document_content(file_path)
            if content:
                # process_resume_file now handles the tuple from load_document_content and merges metadata
                structured_data = process_resume_file(llm_for_extraction, extraction_prompt, content, file_path, doc_metadata)
                if structured_data:
                    # Add original filename for tracking
                    structured_data['_original_filename'] = os.path.basename(file_path)
                    newly_extracted_profiles.append(structured_data)
        print(f"Successfully extracted {len(newly_extracted_profiles)} new profiles.")
    else:
        print("\nNo new resumes detected to process.")

    # Step 5: Consolidate all silver-layer profiles
    all_structured_profiles_final_silver = profiles_to_keep + newly_extracted_profiles

    if not all_structured_profiles_final_silver:
        print("No structured profiles (kept or newly extracted) available to proceed. Exiting.")
        exit()

    # --- NEW STEP: Filter and Reject Profiles ---
    processed_profiles = []
    rejected_profiles_count = 0
    print("\nFiltering profiles based on contact information (email and phone number)...")
    for profile in all_structured_profiles_final_silver:
        email = profile.get('email_id')
        phone = profile.get('phone_number')

        # Check if both email and phone are None or empty strings (after cleaning by llm_extractor)
        if (email is None or str(email).strip() == '' or str(email).lower() == 'not found') and \
           (phone is None or str(phone).strip() == '' or str(phone).lower() == 'not found'):
            
            original_filename = profile.get('_original_filename', 'unknown_file')
            print(f"  Rejecting profile for '{original_filename}': Missing both email and phone number.")
            rejected_profiles_count += 1

            # Upload the original raw resume to the rejected bucket
            local_raw_resume_path = os.path.join(RAW_RESUMES_DIR, original_filename)
            if os.path.exists(local_raw_resume_path):
                try:
                    upload_to_minio(minio_client, REJECTED_PROFILES_BUCKET_NAME, original_filename, local_raw_resume_path)
                    print(f"  Uploaded rejected raw resume '{original_filename}' to '{REJECTED_PROFILES_BUCKET_NAME}'.")
                except Exception as e:
                    print(f"  Error uploading rejected resume '{original_filename}' to MinIO: {e}")
            else:
                print(f"  Warning: Original raw resume file '{original_filename}' not found locally for rejection upload.")
        else:
            processed_profiles.append(profile)
    
    all_structured_profiles_final_silver = processed_profiles
    print(f"Finished filtering. {rejected_profiles_count} profiles rejected. {len(all_structured_profiles_final_silver)} profiles remaining for further processing.")

    if not all_structured_profiles_final_silver:
        print("No profiles remaining after filtering. Exiting ETL process.")
        exit()
    # --- END NEW STEP: Filter and Reject Profiles ---

    # Step 6: Save ALL (kept + new) Structured Data to JSON (Silver Layer) locally
    print(f"\nSaving {len(all_structured_profiles_final_silver)} combined structured profiles to '{OUTPUT_JSON_FILE}' (Silver Layer) locally...")
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_structured_profiles_final_silver, f, indent=2, ensure_ascii=False)
    print("Silver Layer data saved locally.")

    # Step 7: Upload Silver Layer JSON to MinIO
    print(f"\nUploading '{OUTPUT_JSON_FILE}' to MinIO bucket '{SILVER_LAYER_BUCKET_NAME}' as '{silver_layer_object_name}'...")
    try:
        upload_to_minio(minio_client, SILVER_LAYER_BUCKET_NAME, silver_layer_object_name, OUTPUT_JSON_FILE)
        print("Silver Layer JSON upload complete.")
    except Exception as e:
        print(f"MinIO upload of Silver Layer JSON failed: {e}")
        print("Please ensure your MinIO server is running and accessible with the correct credentials in .env.")
        exit()

    # --- Gold Layer Transformation ---
    # The gold layer transformation is kept here for potential future use (e.g., if you want
    # to store a deduplicated and enriched dataset separately), and its output will now
    # also be passed to Qdrant for payload enrichment.
    print("\nStarting Gold Layer Transformation...")
    gold_df = transform_to_gold_layer(all_structured_profiles_final_silver)
    
    if gold_df.empty:
        print("Gold layer transformation resulted in an empty DataFrame. No Gold Layer Parquet will be generated or uploaded.")
        # If gold_df is empty, we must ensure all_structured_profiles_final_silver is used for payload as well
        # to avoid errors in Qdrant builder.
        gold_layer_for_qdrant = all_structured_profiles_final_silver
    else:
        print(f"Saving {len(gold_df)} gold layer profiles to '{GOLD_LAYER_PARQUET_FILE}'...")
        gold_df.to_parquet(GOLD_LAYER_PARQUET_FILE, index=False)
        print("Gold Layer data saved as Parquet.")

        # Upload Gold Layer Parquet to MinIO
        print(f"\nUploading '{GOLD_LAYER_PARQUET_FILE}' to MinIO bucket '{MINIO_BUCKET_NAME}' as '{MINIO_OBJECT_NAME}'...")
        try:
            upload_to_minio(minio_client, MINIO_BUCKET_NAME, MINIO_OBJECT_NAME, GOLD_LAYER_PARQUET_FILE)
            print("Gold Layer Parquet upload complete.")
        except Exception as e:
            print(f"MinIO operation failed for Gold Layer Parquet: {e}")
            print("Please ensure your MinIO server is running and accessible with the correct credentials in .env.")
        
        gold_layer_for_qdrant = gold_df.to_dict(orient='records')
    # --- End Gold Layer Transformation ---

    # Step 8: Generate and Persist Qdrant DB
    # This step now uses the 'all_structured_profiles_final_silver' data for generating embeddings,
    # and the 'gold_layer_for_qdrant' data for the payload of each Qdrant point.
    print(f"\nGenerating and persisting Qdrant DB into collection '{QDRANT_COLLECTION_NAME}'...")
    generate_qdrant_db(all_structured_profiles_final_silver, gold_layer_for_qdrant)
    print("Qdrant DB generation complete.")

    # Clean up downloaded raw resumes and temporary silver layer JSON
    print(f"\nCleaning up local raw resumes directory: {RAW_RESUMES_DIR}")
    shutil.rmtree(RAW_RESUMES_DIR, ignore_errors=True)
    if os.path.exists(OUTPUT_JSON_FILE):
        os.remove(OUTPUT_JSON_FILE)
    print("Local temporary files cleaned up.")

    print("\nOffline processing finished. You can now run your Streamlit app.")
    print(f"Remember to use `python run_app.py` from the project root.")

if __name__ == "__main__":
    main()

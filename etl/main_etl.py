import os
import sys
import json
import shutil
import signal
import atexit
import pandas as pd
from datetime import datetime
from minio.error import S3Error

from shared.config import (
    RAW_RESUMES_DIR, OUTPUT_JSON_FILE, GOLD_LAYER_PARQUET_FILE, OLLAMA_MODEL,
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE,
    MINIO_BUCKET_NAME, MINIO_OBJECT_NAME,
    QDRANT_COLLECTION_NAME,
    RAW_RESUMES_BUCKET_NAME,
    SILVER_LAYER_BUCKET_NAME,
    REJECTED_PROFILES_BUCKET_NAME,
    STALE_PROFILES_BUCKET_NAME
)
from etl.data_loader import load_document_content
from etl.llm_extractor import initialize_ollama_llm, get_extraction_prompt, process_resume_file
from etl.gold_layer_transformer import transform_to_gold_layer
from etl.qdrant_builder import generate_qdrant_db
from etl.minio_utils import get_minio_client, upload_to_minio, download_from_minio

# -------- Cleanup Hooks --------
TEMP_PATHS = []

def cleanup():
    print("\nCleaning up temporary files...")
    for path in TEMP_PATHS:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                print(f"  Removed directory: {path}")
            elif os.path.isfile(path):
                os.remove(path)
                print(f"  Removed file: {path}")
        except Exception as e:
            print(f"  Failed to remove {path}: {e}")

def handle_exit(signum=None, frame=None):
    print(f"\nReceived signal {signum}, exiting gracefully...")
    cleanup()
    sys.exit(1)

atexit.register(cleanup)
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# -------- Main ETL Logic --------
def main():
    print("Starting structured data extraction and embedding generation (supports PDF, DOCX, TXT)....\n")

    try:
        minio_client = get_minio_client(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE)
    except Exception as e:
        print(f"MinIO client initialization failed: {e}")
        exit()

    os.makedirs(RAW_RESUMES_DIR, exist_ok=True)
    TEMP_PATHS.append(RAW_RESUMES_DIR)

    processed_filenames_in_json = set()
    silver_layer_object_name = os.path.basename(OUTPUT_JSON_FILE)
    temp_silver_layer_local_path = OUTPUT_JSON_FILE
    existing_profiles_from_json = []

    os.makedirs(os.path.dirname(temp_silver_layer_local_path) or ".", exist_ok=True)
    TEMP_PATHS.append(OUTPUT_JSON_FILE)

    try:
        download_from_minio(minio_client, SILVER_LAYER_BUCKET_NAME, silver_layer_object_name, temp_silver_layer_local_path)
        with open(temp_silver_layer_local_path, 'r', encoding='utf-8') as f:
            existing_profiles_from_json = json.load(f)
            processed_filenames_in_json = {
                profile.get('_original_filename') for profile in existing_profiles_from_json
                if profile.get('_original_filename')
            }
            print(f"Loaded {len(processed_filenames_in_json)} previously processed resumes from Silver Layer JSON.")

        actual_filenames_in_bucket = {
            os.path.basename(obj.object_name)
            for obj in minio_client.list_objects(RAW_RESUMES_BUCKET_NAME, recursive=True)
            if not obj.is_dir
        }
        valid_profiles = []
        stale_profiles = []
        for profile in existing_profiles_from_json:
            original_filename = profile.get('_original_filename')
            if original_filename in actual_filenames_in_bucket:
                valid_profiles.append(profile)
            else:
                print(f"  Stale profile found: '{original_filename}' (file missing in raw-resumes bucket)")
                stale_profiles.append(profile)

        if stale_profiles:
            stale_metadata_filename = f"stale_profiles_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            stale_metadata_local_path = os.path.join("/tmp", stale_metadata_filename)
            TEMP_PATHS.append(stale_metadata_local_path)
            with open(stale_metadata_local_path, 'w', encoding='utf-8') as f:
                json.dump(stale_profiles, f, indent=2, ensure_ascii=False)
            try:
                upload_to_minio(minio_client, STALE_PROFILES_BUCKET_NAME, stale_metadata_filename, stale_metadata_local_path)
                print(f"  Uploaded stale profiles metadata to '{STALE_PROFILES_BUCKET_NAME}/{stale_metadata_filename}'.")
            except Exception as e:
                print(f"  Failed to upload stale profiles metadata: {e}")

        existing_profiles_from_json = valid_profiles
        processed_filenames_in_json = {
            profile.get('_original_filename') for profile in existing_profiles_from_json
        }
        print(f"Validation complete. Removed {len(stale_profiles)} stale profile(s) from JSON.")

    except S3Error as e:
        if "NoSuchKey" in str(e):
            print("No existing Silver Layer JSON found in MinIO. All resumes will be considered new.")
        else:
            print(f"MinIO error reading Silver Layer JSON: {e}")
    except Exception as e:
        print(f"Failed to load existing JSON: {e}")

    raw_resume_objects = []
    new_downloads = 0
    try:
        objects = minio_client.list_objects(RAW_RESUMES_BUCKET_NAME, recursive=True)
        for obj in objects:
            if obj.is_dir:
                continue
            file_extension = os.path.splitext(obj.object_name)[1].lower()
            if file_extension not in ('.pdf', '.docx', '.txt'):
                continue

            filename = os.path.basename(obj.object_name)
            if filename in processed_filenames_in_json:
                print(f"  Skipping already processed resume: {filename}")
                continue

            local_path = os.path.join(RAW_RESUMES_DIR, filename)
            download_from_minio(minio_client, RAW_RESUMES_BUCKET_NAME, obj.object_name, local_path)
            raw_resume_objects.append(filename)
            new_downloads += 1

        if new_downloads == 0:
            print("No new resumes to process. All files in MinIO have already been processed.")
            exit()

        print(f"Downloaded {new_downloads} new resume(s) from MinIO.")

    except S3Error as e:
        print(f"MinIO S3 error while listing/downloading raw resumes: {e}")
        exit()
    except Exception as e:
        print(f"Unexpected error while downloading raw resumes: {e}")
        exit()

    try:
        llm_for_extraction = initialize_ollama_llm(OLLAMA_MODEL)
        extraction_prompt = get_extraction_prompt()
    except Exception as e:
        print(f"Initialization failed: {e}")
        exit()

    new_resume_files_to_process = [
        os.path.join(RAW_RESUMES_DIR, filename)
        for filename in os.listdir(RAW_RESUMES_DIR)
    ]

    newly_extracted_profiles = []
    if new_resume_files_to_process:
        print(f"\nProcessing {len(new_resume_files_to_process)} new resume(s)...")
        for file_path in new_resume_files_to_process:
            content, doc_metadata = load_document_content(file_path)
            if content:
                structured_data = process_resume_file(llm_for_extraction, extraction_prompt, content, file_path, doc_metadata)
                if structured_data:
                    structured_data['_original_filename'] = os.path.basename(file_path)
                    newly_extracted_profiles.append(structured_data)
        print(f"Successfully extracted {len(newly_extracted_profiles)} new profiles.")
    else:
        print("\nNo new resumes found for processing.")

    all_structured_profiles_final_silver = existing_profiles_from_json + newly_extracted_profiles

    if not all_structured_profiles_final_silver:
        print("No structured profiles available to proceed. Exiting.")
        exit()

    processed_profiles = []
    rejected_profiles_count = 0
    print("\nFiltering profiles based on contact information (email and phone number)...")
    for profile in all_structured_profiles_final_silver:
        email = profile.get('email_id')
        phone = profile.get('phone_number')

        if (email is None or str(email).strip() == '' or str(email).lower() == 'not found') and \
           (phone is None or str(phone).strip() == '' or str(phone).lower() == 'not found'):

            original_filename = profile.get('_original_filename', 'unknown_file')
            print(f"  Rejecting profile for '{original_filename}': Missing both email and phone number.")
            rejected_profiles_count += 1

            local_raw_resume_path = os.path.join(RAW_RESUMES_DIR, original_filename)
            if os.path.exists(local_raw_resume_path):
                try:
                    upload_to_minio(minio_client, REJECTED_PROFILES_BUCKET_NAME, original_filename, local_raw_resume_path)
                    print(f"  Uploaded rejected raw resume '{original_filename}' to '{REJECTED_PROFILES_BUCKET_NAME}'.")
                    try:
                        minio_client.remove_object(RAW_RESUMES_BUCKET_NAME, original_filename)
                        print(f"  Removed original resume '{original_filename}' from '{RAW_RESUMES_BUCKET_NAME}'.")
                    except Exception as e:
                        print(f"  Warning: Failed to remove '{original_filename}' from '{RAW_RESUMES_BUCKET_NAME}': {e}")
                except Exception as e:
                    print(f"  Error uploading rejected resume '{original_filename}' to MinIO: {e}")
            else:
                print(f"  Warning: Original raw resume file '{original_filename}' not found locally for rejection upload.")
        else:
            processed_profiles.append(profile)

    all_structured_profiles_final_silver = processed_profiles
    print(f"Finished filtering. {rejected_profiles_count} profiles rejected. {len(all_structured_profiles_final_silver)} profiles remaining.")

    if not all_structured_profiles_final_silver:
        print("No profiles remaining after filtering. Exiting ETL process.")
        exit()

    print(f"\nSaving {len(all_structured_profiles_final_silver)} profiles to '{OUTPUT_JSON_FILE}' (Silver Layer) locally...")
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_structured_profiles_final_silver, f, indent=2, ensure_ascii=False)
    print("Silver Layer data saved locally.")

    print(f"\nUploading '{OUTPUT_JSON_FILE}' to MinIO bucket '{SILVER_LAYER_BUCKET_NAME}' as '{silver_layer_object_name}'...")
    try:
        upload_to_minio(minio_client, SILVER_LAYER_BUCKET_NAME, silver_layer_object_name, OUTPUT_JSON_FILE)
        print("Silver Layer JSON upload complete.")
    except Exception as e:
        print(f"MinIO upload of Silver Layer JSON failed: {e}")
        exit()

    print("\nStarting Gold Layer Transformation...")
    gold_df = transform_to_gold_layer(all_structured_profiles_final_silver)

    if gold_df.empty:
        print("Gold layer transformation resulted in an empty DataFrame. Skipping Gold Layer upload.")
        gold_layer_for_qdrant = all_structured_profiles_final_silver
    else:
        print(f"Saving {len(gold_df)} gold layer profiles to '{GOLD_LAYER_PARQUET_FILE}'...")
        gold_df.to_parquet(GOLD_LAYER_PARQUET_FILE, index=False)
        TEMP_PATHS.append(GOLD_LAYER_PARQUET_FILE)
        print("Gold Layer data saved.")

        print(f"\nUploading '{GOLD_LAYER_PARQUET_FILE}' to MinIO bucket '{MINIO_BUCKET_NAME}' as '{MINIO_OBJECT_NAME}'...")
        try:
            upload_to_minio(minio_client, MINIO_BUCKET_NAME, MINIO_OBJECT_NAME, GOLD_LAYER_PARQUET_FILE)
            print("Gold Layer upload complete.")
        except Exception as e:
            print(f"Gold Layer upload failed: {e}")
        gold_layer_for_qdrant = gold_df.to_dict(orient='records')

    print(f"\nGenerating Qdrant DB collection '{QDRANT_COLLECTION_NAME}'...")
    generate_qdrant_db(all_structured_profiles_final_silver, gold_layer_for_qdrant)
    print("Qdrant DB generation complete.")

    print("\nOffline processing complete. Run your Streamlit app with `python run_app.py`.")

if __name__ == "__main__":
    main()

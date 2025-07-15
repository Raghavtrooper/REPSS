import os
from minio import Minio
from minio.error import S3Error
from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    RAW_RESUMES_BUCKET_NAME,
    REJECTED_PROFILES_BUCKET_NAME
)

def get_minio_client():
    """
    Initializes and returns a MinIO client.
    """
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        return client
    except Exception as e:
        print(f"Error initializing MinIO client: {e}")
        return None

def get_pdf_objects_in_bucket(client, bucket_name):
    """
    Lists all PDF objects in a specified MinIO bucket.
    Args:
        client (Minio): The MinIO client instance.
        bucket_name (str): The name of the bucket to list objects from.
    Returns:
        set: A set of object names (PDFs) found in the bucket.
    """
    pdf_objects = set()
    try:
        # List objects with a prefix (optional, but good for filtering)
        # and include_user_metadata to get more details if needed.
        # For PDFs, we'll just check the object name for '.pdf' extension.
        objects = client.list_objects(bucket_name, recursive=True)
        for obj in objects:
            if obj.object_name.lower().endswith('.pdf'):
                pdf_objects.add(obj.object_name)
        print(f"Found {len(pdf_objects)} PDF objects in '{bucket_name}' bucket.")
    except S3Error as e:
        print(f"S3 Error listing objects in '{bucket_name}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while listing objects in '{bucket_name}': {e}")
    return pdf_objects

def delete_objects_from_bucket(client, bucket_name, objects_to_delete):
    """
    Deletes specified objects from a MinIO bucket.
    Args:
        client (Minio): The MinIO client instance.
        bucket_name (str): The name of the bucket to delete objects from.
        objects_to_delete (set): A set of object names to delete.
    """
    if not objects_to_delete:
        print(f"No objects to delete from '{bucket_name}'.")
        return

    print(f"Attempting to delete {len(objects_to_delete)} objects from '{bucket_name}'...")
    for obj_name in objects_to_delete:
        try:
            client.remove_object(bucket_name, obj_name)
            print(f"Successfully deleted '{obj_name}' from '{bucket_name}'.")
        except S3Error as e:
            print(f"S3 Error deleting '{obj_name}' from '{bucket_name}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred while deleting '{obj_name}' from '{bucket_name}': {e}")

def main():
    """
    Main function to identify rejected PDFs and delete them from raw resumes.
    """
    minio_client = get_minio_client()
    if not minio_client:
        return

    print(f"Checking rejected profiles bucket: '{REJECTED_PROFILES_BUCKET_NAME}'")
    rejected_pdfs = get_pdf_objects_in_bucket(minio_client, REJECTED_PROFILES_BUCKET_NAME)

    if rejected_pdfs:
        print(f"\nPDFs found in '{REJECTED_PROFILES_BUCKET_NAME}':")
        for pdf in rejected_pdfs:
            print(f"- {pdf}")

        print(f"\nNow checking and deleting these PDFs from raw resumes bucket: '{RAW_RESUMES_BUCKET_NAME}'")
        # We need to ensure that the files actually exist in the raw resumes bucket before attempting to delete
        # This step is optional but makes the deletion safer.
        raw_resumes_pdfs = get_pdf_objects_in_bucket(minio_client, RAW_RESUMES_BUCKET_NAME)

        # Find the intersection: PDFs that are both rejected and in raw resumes
        pdfs_to_delete_from_raw = rejected_pdfs.intersection(raw_resumes_pdfs)

        if pdfs_to_delete_from_raw:
            print(f"\nPDFs to be deleted from '{RAW_RESUMES_BUCKET_NAME}':")
            for pdf in pdfs_to_delete_from_raw:
                print(f"- {pdf}")
            delete_objects_from_bucket(minio_client, RAW_RESUMES_BUCKET_NAME, pdfs_to_delete_from_raw)
        else:
            print(f"No common PDF files found between '{REJECTED_PROFILES_BUCKET_NAME}' and '{RAW_RESUMES_BUCKET_NAME}'.")
    else:
        print(f"No PDF files found in the '{REJECTED_PROFILES_BUCKET_NAME}' bucket.")

if __name__ == "__main__":
    main()

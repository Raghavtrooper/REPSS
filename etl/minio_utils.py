from minio import Minio
from minio.error import S3Error
import io # Import io for handling file-like objects

def get_minio_client(endpoint, access_key, secret_key, secure):
    """
    Initializes and returns the MinIO client.
    """
    try:
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        print(f"Connected to MinIO at {endpoint}")
        return client
    except Exception as e:
        print(f"Error connecting to MinIO: {e}")
        raise # Re-raise to stop script execution if MinIO fails

def upload_to_minio(minio_client, bucket_name, object_name, file_path):
    """
    Uploads a file to a specified MinIO bucket.
    Creates the bucket if it doesn't exist.
    """
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created successfully.")
        
        minio_client.fput_object(bucket_name, object_name, file_path)
        print(f"Successfully uploaded '{file_path}' to '{bucket_name}/{object_name}'.")
    except S3Error as e:
        print(f"MinIO S3 Error during upload: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during MinIO upload: {e}")
        raise

def download_from_minio(minio_client, bucket_name, object_name, file_path):
    """
    Downloads a file from a specified MinIO bucket to a local path.
    """
    try:
        minio_client.fget_object(bucket_name, object_name, file_path)
        print(f"Successfully downloaded '{object_name}' from '{bucket_name}' to '{file_path}'.")
    except S3Error as e:
        print(f"MinIO S3 Error during download: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during MinIO download: {e}")
        raise

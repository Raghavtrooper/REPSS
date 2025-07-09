import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# Load environment variables from .env file
load_dotenv()

# --- Qdrant Configuration (from your config.py) ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "157.180.44.51")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # Will be None if not set
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "employee_profiles")

def check_qdrant_data():
    """
    Connects to Qdrant, checks if the collection exists,
    and counts the total number of points (profiles).
    """
    print(f"Attempting to connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)

        # 1. Check if the collection exists
        try:
            collection_info = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            print(f"\nCollection '{QDRANT_COLLECTION_NAME}' exists.")
            print(f"Collection status: {collection_info.status.value}")
            # print(f"Vector count: {collection_info.vectors_count}") # This is redundant as count will give us total
        except UnexpectedResponse as e:
            if e.status_code == 404:
                print(f"\nError: Collection '{QDRANT_COLLECTION_NAME}' does not exist.")
                print("Please ensure your ETL pipeline has run successfully to create the collection.")
                return
            else:
                print(f"\nError accessing collection '{QDRANT_COLLECTION_NAME}': {e}")
                return
        except Exception as e:
            print(f"\nAn unexpected error occurred while checking collection: {e}")
            return

        # 2. Count the total number of points (profiles)
        try:
            count_result = client.count(
                collection_name=QDRANT_COLLECTION_NAME,
                exact=True # Get exact count of all points
            )
            total_profiles = count_result.count
            print(f"\nTotal number of profiles in collection '{QDRANT_COLLECTION_NAME}': {total_profiles}")

            if total_profiles == 0:
                print("The collection is empty. No data has been indexed yet.")
            else:
                # Optionally, still retrieve a small sample to show some data without affecting the total count
                print(f"\nRetrieving a sample of up to 5 points from '{QDRANT_COLLECTION_NAME}':")
                try:
                    points_iterator, _ = client.scroll(
                        collection_name=QDRANT_COLLECTION_NAME,
                        limit=5,
                        with_payload=True,
                        with_vectors=False
                    )

                    if not points_iterator:
                        print("No points found in the sample (this might happen if total_profiles is small but not zero).")
                    else:
                        for i, point in enumerate(points_iterator):
                            print(f"\n--- Sample Point {i+1} (ID: {point.id}) ---")
                            print(f"Payload (Metadata):")
                            for key, value in point.payload.items():
                                if isinstance(value, str) and len(value) > 150:
                                    print(f"  {key}: {value[:150]}...")
                                else:
                                    print(f"  {key}: {value}")
                except Exception as e:
                    print(f"\nError retrieving sample points: {e}")


        except Exception as e:
            print(f"\nError counting points in collection: {e}")
            return

    except Exception as e:
        print(f"Could not connect to Qdrant: {e}")
        print("Please ensure the Qdrant Docker container is running and accessible.")

if __name__ == "__main__":
    check_qdrant_data()
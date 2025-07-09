import os
import shutil
import uuid # Import the uuid module
from qdrant_client import QdrantClient, models
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

# Import Qdrant configuration
from shared.config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, QDRANT_COLLECTION_NAME

def generate_qdrant_db(silver_profiles, gold_profiles):
    """
    Generates and persists the Qdrant DB.
    Connects to a running Qdrant service (e.g., in Docker).
    Embeddings are generated from 'silver_profiles' (raw extracted text),
    while the payload for each Qdrant point comes from 'gold_profiles'
    (deduplicated and enriched data).
    """
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    # Initialize Qdrant Client for connection to the Dockerized service
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)

    # Define the embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a map for quick lookup of gold profiles by their 'id'
    # This assumes 'id' is a consistent unique identifier across silver and gold layers.
    gold_profile_map = {profile['id']: profile for profile in gold_profiles}

    points = []
    # Iterate through silver profiles to generate embeddings
    for i, silver_profile in enumerate(silver_profiles):
        # Retrieve the corresponding gold profile using the 'id'
        profile_id = silver_profile.get('id')
        gold_profile = gold_profile_map.get(profile_id)

        if not gold_profile:
            print(f"  Warning: No matching gold profile found for silver profile ID: {profile_id}. Skipping this point.")
            continue

        # Combine relevant fields from the silver profile into a single text string for the embedding model to process
        # This content string is what the embedding will be based on.
        skills_str = ", ".join(silver_profile.get('skills', [])) if silver_profile.get('skills') else "Not Found"
        companies_str = ", ".join(silver_profile.get('companies_worked_with_duration', [])) if silver_profile.get('companies_worked_with_duration') else "Not Found"


        content = (
            f"Name: {silver_profile.get('name', 'Not Found')}\n"
            f"Email: {silver_profile.get('email_id', 'Not Found')}\n"
            f"Phone: {silver_profile.get('phone_number', 'Not Found')}\n"
            f"Location: {silver_profile.get('location', 'Not Found')}\n"
            f"Objective: {silver_profile.get('objective', 'Not Found')}\n"
            f"Skills: {skills_str}\n"
            f"Qualifications Summary: {silver_profile.get('qualifications_summary', 'Not Found')}\n"
            f"Experience Summary: {silver_profile.get('experience_summary', 'Not Found')}\n"
            f"Companies Worked With Duration: {companies_str}\n" # Added new field
            f"Has Photo: {silver_profile.get('has_photo', False)}"
        )

        # Generate embedding for the content derived from the silver profile
        vector = embedding_function.embed_query(content)

        # Prepare payload (metadata) for Qdrant using the gold profile.
        # Ensure all values are JSON-serializable.
        # The payload will contain the deduplicated and enriched data from the gold layer.
        payload = {k: v for k, v in gold_profile.items()}
        
        # Add the 'content' string (from silver layer) to the payload under 'page_content' key
        # This allows the original text used for embedding to be retrieved with the point.
        payload['page_content'] = content 

        # Generate a UUID for each point ID
        # This ensures the ID is in a format Qdrant expects (UUID or unsigned integer)
        point_id = str(uuid.uuid4()) 

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        )

    if not points:
        print("No documents to add to Qdrant. Aborting Qdrant DB creation.")
        return

    # Check if collection exists and create if not, or recreate if needed
    try:
        client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists. Deleting and recreating.")
        client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
    except Exception as e:
        print(f"Collection '{QDRANT_COLLECTION_NAME}' does not exist or could not be accessed. Creating new collection.")

    print(f"Creating Qdrant collection '{QDRANT_COLLECTION_NAME}' with {len(points)} documents...")
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=len(vector), distance=models.Distance.COSINE),
    )

    # Upsert points (add documents) to the collection
    client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=points,
        wait=True
    )
    print(f"Qdrant DB created successfully with {len(points)} documents in collection '{QDRANT_COLLECTION_NAME}'.")

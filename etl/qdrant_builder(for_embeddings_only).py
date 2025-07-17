import os
import shutil
import uuid # Import the uuid module
from qdrant_client import QdrantClient, models
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document # Keep this import, though we're not using Document objects directly for embedding text

# Import Qdrant configuration
from shared.config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, QDRANT_COLLECTION_NAME

def generate_qdrant_db(silver_profiles, gold_profiles):
    """
    Generates and persists the Qdrant DB.
    Connects to a running Qdrant service (e.g., in Docker).
    Embeddings are generated from a concatenated text representation of 'silver_profiles' (raw extracted structured data),
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
        # Construct a comprehensive text string for embedding from the structured silver_profile
        # Prioritize fields that contain rich textual information for better search relevance
        embedding_text_parts = []
        
        # Add core textual fields if they exist and are not None
        if silver_profile.get('name'):
            embedding_text_parts.append(silver_profile['name'])
        if silver_profile.get('current_job_title'):
            embedding_text_parts.append(silver_profile['current_job_title'])
        if silver_profile.get('objective'):
            embedding_text_parts.append(silver_profile['objective'])
        if silver_profile.get('qualifications_summary'):
            embedding_text_parts.append(silver_profile['qualifications_summary'])
        if silver_profile.get('experience_summary'):
            embedding_text_parts.append(silver_profile['experience_summary'])
        
        # Add list-based fields, joining them into strings
        if silver_profile.get('skills'):
            embedding_text_parts.extend(silver_profile['skills']) # skills is already a list of strings
        if silver_profile.get('certifications'):
            embedding_text_parts.extend(silver_profile['certifications'])
        if silver_profile.get('awards_achievements'):
            embedding_text_parts.extend(silver_profile['awards_achievements'])
        if silver_profile.get('projects'):
            embedding_text_parts.extend(silver_profile['projects'])
        if silver_profile.get('languages'):
            embedding_text_parts.extend(silver_profile['languages'])
        if silver_profile.get('location'):
            embedding_text_parts.append(silver_profile['location'])
        if silver_profile.get('companies_worked_with_duration'):
            embedding_text_parts.extend(silver_profile['companies_worked_with_duration'])
        
        # Join all parts into a single string for embedding.
        # Ensure each part is converted to string and filter out empty parts.
        embedding_content = " ".join([str(part) for part in embedding_text_parts if part])
        
        if not embedding_content.strip():
            print(f"  Skipping embedding for profile ID {silver_profile.get('id', 'N/A')} due to empty content for embedding.")
            continue # Skip if there's no meaningful content to embed

        # Generate embedding vector from the constructed content
        vector = embedding_function.embed_query(embedding_content) 

        # Get the corresponding gold profile as payload for the Qdrant point
        payload = gold_profile_map.get(silver_profile['id'], {})
        if not payload:
            print(f"  WARNING: Gold profile not found for silver profile ID: {silver_profile.get('id', 'N/A')}. Skipping Qdrant point creation.")
            continue # Skip if no corresponding gold profile is found

        # Generate a unique UUID for each Qdrant point ID
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
        print(f"Collection '{QDRANT_COLLECTION_NAME}' does not exist or could not be accessed. Creating new collection. Error: {e}")

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
    print(f"Qdrant DB created and populated with {len(points)} points.")


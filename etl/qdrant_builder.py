import io
import uuid
import math
import requests
import re
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
from langchain_community.embeddings import SentenceTransformerEmbeddings
from rank_bm25 import BM25Okapi
import pickle
import json
import os
import asyncio # Added for async operations with MinIO
from minio import Minio # Added for direct MinIO interaction
from minio.error import S3Error # Added for direct MinIO error handling

QDRANT_COLLECTION_NAME = "employee_profiles1"
QDRANT_HOST = "157.180.44.51"
QDRANT_PORT = 6333

# --- MinIO Configuration for qdrant_builder (Directly used here) ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "157.180.44.51:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "vector-service-models") # Consistent bucket name
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true" # Convert string to boolean

# --- Utility to clean vectors ---
def clean_vector(vec):
    return [
        0.0 if (
            v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
        ) else float(v) for v in vec
    ]

# --- Safe appender for profile fields ---
def safe_append(parts, val):
    if isinstance(val, str) and val.strip().lower() != "not found":
        parts.append(val.strip())
    elif isinstance(val, list):
        parts.extend([
            str(v).strip() for v in val if isinstance(v, str) and v.strip().lower() != "not found"
        ])

async def generate_qdrant_db_async(silver_profiles, gold_profiles):
    print(f"Generating and persisting Qdrant DB into collection '{QDRANT_COLLECTION_NAME}'...")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    dense_embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize MinIO client directly in qdrant_builder
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )

    # --- Step 1: Build Corpus and Map ---
    corpus = []
    profile_id_map = []
    structured_profiles_map = {}
    tokenized_corpus = []

    for profile in silver_profiles:
        parts = []

        for key in [
            "name", "current_job_title", "objective",
            "qualifications_summary", "experience_summary", "location"
        ]:
            safe_append(parts, profile.get(key))

        for key in [
            "skills", "certifications", "awards_achievements",
            "projects", "languages", "companies_worked_with_duration"
        ]:
            safe_append(parts, profile.get(key))

        content = " ".join(parts).strip()
        if not content:
            print(f"  ⚠️ Profile {profile.get('id', 'N/A')} has no valid embedding text. Using fallback placeholder.")
            content = "[EMPTY PROFILE CONTENT]"

        corpus.append(content)
        tokens = re.findall(r'\b\w+\b', content.lower())  # regex-based tokenization
        tokenized_corpus.append(tokens)
        profile_id_map.append(profile["id"])
        structured_profiles_map[profile["id"]] = profile

    print("Encoding dense and sparse vectors in batch...")
    dense_results = dense_embedder.embed_documents(corpus)

    # --- Use BM25 to get sparse vectors ---
    print("Generating sparse BM25 vectors...")
    bm25_model = BM25Okapi(tokenized_corpus)
    vocab = list(set(token for doc in tokenized_corpus for token in doc))
    token_to_index = {token: idx for idx, token in enumerate(vocab)}

    points = []
    for idx, pid in enumerate(profile_id_map):
        dense_vector = clean_vector(dense_results[idx])

        doc_tokens = tokenized_corpus[idx]
        doc_scores = bm25_model.get_scores(doc_tokens)
        non_zero_indices = [i for i, score in enumerate(doc_scores) if score > 0]

        sparse_indices = [token_to_index[doc_tokens[i]] for i in non_zero_indices if doc_tokens[i] in token_to_index]
        sparse_values = [float(doc_scores[i]) for i in non_zero_indices if doc_tokens[i] in token_to_index]

        payload = structured_profiles_map[pid]

        points.append({
            "id": str(uuid.uuid4()),
            "vector": {"dense": dense_vector},
            "sparse_vector": {
                "indices": sparse_indices,
                "values": clean_vector(sparse_values)
            },
            "payload": payload
        })

    # --- Step 3: Create Collection ---
    if client.collection_exists(QDRANT_COLLECTION_NAME):
        print(f"Collection '{QDRANT_COLLECTION_NAME}' exists. Deleting it...")
        client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)

    print(f"Creating Qdrant collection '{QDRANT_COLLECTION_NAME}' with hybrid vector config...")
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=len(dense_vector), distance=Distance.COSINE),
            "sparse": VectorParams(size=len(vocab), distance=Distance.DOT)
        }
    )

    # --- Step 4: Upload to Qdrant via REST API ---
    print(f"Uploading {len(points)} hybrid points via HTTP...")
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{QDRANT_COLLECTION_NAME}/points?wait=true"
    headers = {"Content-Type": "application/json"}

    response = requests.put(url, json={"points": points}, headers=headers)
    if response.status_code == 200:
        print(f"✅ Successfully uploaded {len(points)} hybrid vectors to Qdrant.")
    else:
        print(f"❌ Upload failed ({response.status_code}): {response.text}")

    # --- Step 5: Upload BM25 model and token_to_index to MinIO directly ---
    print(f"Uploading BM25 model and token_to_index to MinIO bucket '{MINIO_BUCKET_NAME}'...")
    try:
        # Make sure the bucket exists
        found = minio_client.bucket_exists(MINIO_BUCKET_NAME)
        if not found:
            minio_client.make_bucket(MINIO_BUCKET_NAME)
            print(f"Created MinIO bucket '{MINIO_BUCKET_NAME}'")
        else:
            print(f"MinIO bucket '{MINIO_BUCKET_NAME}' already exists")

        # Serialize BM25 model
        bm25_model_bytes = pickle.dumps(bm25_model)
        minio_client.put_object(MINIO_BUCKET_NAME, "bm25_model.pkl", io.BytesIO(bm25_model_bytes), len(bm25_model_bytes))

        # Serialize token_to_index
        token_to_index_bytes = json.dumps(token_to_index).encode('utf-8')
        minio_client.put_object(MINIO_BUCKET_NAME, "token_to_index.json", io.BytesIO(token_to_index_bytes), len(token_to_index_bytes))

        print("✅ Successfully uploaded BM25 model and token_to_index to MinIO.")
    except S3Error as e:
        print(f"❌ MinIO S3 Error during model upload: {e}")
    except Exception as e:
        print(f"❌ Failed to upload BM25 model or token_to_index to MinIO: {e}")


# Synchronous wrapper for generate_qdrant_db_async
def generate_qdrant_db(silver_profiles, gold_profiles):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(generate_qdrant_db_async(silver_profiles, gold_profiles))

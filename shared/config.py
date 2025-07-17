import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Shared Configuration ---

# Directory for raw resume files (local temporary storage for downloaded files)
RAW_RESUMES_DIR = "./raw_resumes"

# Path for the structured JSON output file (Silver Layer) - local temporary storage
# UPDATED: Now includes 'outputjson' directory for local storage
OUTPUT_JSON_FILE = "./outputjson/employee_profiles.json"

# Path for the cleaned & enriched Parquet output file (Gold Layer)
GOLD_LAYER_PARQUET_FILE = "./gold_employee_profiles.parquet"

# Ollama LLM model name for both extraction and RAG
# Ensure Ollama is running and this model is pulled (`ollama pull <model_name>`)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "157.180.44.51:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == 'true' # Convert string "True"/"False" to boolean

# MinIO bucket for raw resumes input
RAW_RESUMES_BUCKET_NAME = os.getenv("RAW_RESUMES_BUCKET_NAME", "rawresumes")

# MinIO bucket for silver layer JSON output
SILVER_LAYER_BUCKET_NAME = os.getenv("SILVER_LAYER_BUCKET_NAME", "outputjson")

# MinIO bucket for gold layer parquet output (existing)
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "extracted")
# MinIO object name for the Gold Layer Parquet file
MINIO_OBJECT_NAME = os.getenv("MINIO_OBJECT_NAME", "gold_employee_profiles.parquet")

# New MinIO bucket for rejected profiles
REJECTED_PROFILES_BUCKET_NAME = os.getenv("REJECTED_PROFILES_BUCKET_NAME", "rejected-resumes")
# New MinIO bucket for rejected profiles
REJECTED_PROFILES_BUCKET_NAME = os.getenv("REJECTED_PROFILES_BUCKET_NAME", "rejected-resumes")

# New MinIO bucket for stale profiles (exist in JSON but missing in raw-resumes bucket)
STALE_PROFILES_BUCKET_NAME = os.getenv("STALE_PROFILES_BUCKET_NAME", "stale-profiles")


# Qdrant Configuration (New)
QDRANT_HOST = os.getenv("QDRANT_HOST", "157.180.44.51") # Hostname/IP of Qdrant service
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333)) # Port of Qdrant service (HTTP)
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", 6334)) # gRPC port (optional, but good to define)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # API key if Qdrant requires authentication
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "employee_profiles") # Default collection name

# RAG Chain Configuration
MAX_HISTORY_MESSAGES = 6 # Maximum number of chat history messages to send to the LLM

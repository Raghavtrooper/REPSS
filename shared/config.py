import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Shared Configuration ---

# Directory for raw resume files
RAW_RESUMES_DIR = "./raw_resumes" 

# Path for the structured JSON output file (Silver Layer)
OUTPUT_JSON_FILE = "./employee_profiles.json" 

# Path for the cleaned & enriched Parquet output file (Gold Layer)
GOLD_LAYER_PARQUET_FILE = "./gold_employee_profiles.parquet"

# Directory for ChromaDB vector store
CHROMA_DB_DIR = "./chroma_db" 

# Ollama LLM model name for both extraction and RAG
# Ensure Ollama is running and this model is pulled (`ollama pull <model_name>`)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3") 

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == 'true' # Convert string "True"/"False" to boolean
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "extracted")
# MinIO object name will now be the Parquet file
MINIO_OBJECT_NAME = os.getenv("MINIO_OBJECT_NAME", "gold_employee_profiles.parquet") 

# RAG Chain Configuration
MAX_HISTORY_MESSAGES = 6 # Maximum number of chat history messages to send to the LLM

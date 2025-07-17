+---------------------------------------------------+
|          ETL Pipeline for Resume Data Processing          |
+---------------------------------------------------+

This project implements an ETL (Extract, Transform, Load) pipeline for processing resume data, extracting structured information using Large Language Models (LLMs), and storing it in a vector database (Qdrant) for efficient search and retrieval. It also includes a FastAPI application to trigger the ETL process via an API endpoint.

+---------------------------------------------------+
|                Detailed Explanation               |
+---------------------------------------------------+

The ETL pipeline is designed to ingest raw resume files (PDF, DOCX, TXT), extract key information, normalize and deduplicate the data, and then store it in both a structured format (Parquet) and an indexed vector database (Qdrant) for semantic search capabilities.

+---------------------------------------------------+
|               ETL Process Overview                |
+---------------------------------------------------+

The main_etl.py script orchestrates the entire ETL process, which includes the following steps:

    MinIO Connection: Initializes a connection to MinIO for data storage and retrieval.

    Raw Resume Download: Downloads raw resume files (PDF, DOCX, TXT) from a specified MinIO bucket to a local directory.

    LLM and Prompt Initialization: Sets up the Ollama LLM and defines the prompt for structured data extraction.

    Silver Layer Processing:

        Loads existing "silver layer" (semi-structured) profiles from MinIO to avoid re-processing unchanged files.

        Compares downloaded raw resumes with existing silver layer data to identify new, modified, or deleted files.

        Processes new/modified resumes:

            Loads document content using data_loader.py, which supports PDF, DOCX, and TXT, and detects if a PDF contains images.

            Extracts structured data using the LLM (Ollama) and a predefined prompt (llm_extractor.py). This includes fields like name, email, phone, skills, experience summary, and more, with strict normalization rules.

            Handles rejected profiles by uploading them to a dedicated MinIO bucket.

        Consolidates newly extracted and retained existing profiles into a final "silver layer" dataset.

        Uploads the updated "silver layer" JSON to MinIO.

    Gold Layer Transformation:

        Transforms the "silver layer" data into a "gold layer" Pandas DataFrame using gold_layer_transformer.py.

        Performs data cleaning, normalization (e.g., skills, email, phone numbers), and annotations for duplicate records rather than merging or dropping them.

        Uploads the gold layer data as a Parquet file to MinIO.

    Qdrant DB Generation:

        Generates and persists the Qdrant vector database using qdrant_builder.py.

        Embeddings are generated from the "silver layer" data, while the "gold layer" data serves as the payload for each Qdrant point, enabling both semantic and keyword search.

        Uploads the BM25 model and token-to-index mapping to MinIO for keyword search functionality.

    Cleanup: Removes temporary local files and directories created during the ETL process.

+---------------------------------------------------+
|                 Module Breakdown                  |
+---------------------------------------------------+

    api_main.py:

        Implements a FastAPI application to expose an API endpoint (/trigger-etl) that initiates the ETL process.

        Provides a root endpoint (/) for a welcome message.

        Handles asynchronous execution of the ETL process to prevent blocking the API.

    data_loader.py:

        Responsible for loading text content from various document types, including PDF, DOCX, and TXT files.

        Utilizes langchain_community loaders and pypdf for content extraction and image detection in PDFs.

        Returns document content as a string along with metadata, including a has_photo flag for PDFs.

    gold_layer_transformer.py:

        Takes structured "silver layer" profiles and transforms them into a "gold layer" Pandas DataFrame.

        Performs data cleaning, normalization (e.g., standardizing skill names, cleaning email IDs and phone numbers), and deduplication annotation.

        Enriches profiles with metadata like duplicate group IDs and associated filenames.

    llm_extractor.py:

        Initializes the Ollama LLM for structured data extraction.

        Defines a comprehensive ChatPromptTemplate for extracting specific fields from resume text (e.g., name, contact info, skills, experience, education) in a standardized JSON format.

        Includes logic for strict normalization of extracted data (e.g., 10-digit phone numbers, standardized locations, normalized skills).

        Processes resume files, sends their content to the LLM for extraction, and merges document-level metadata.

    main_etl.py:

        The main script that orchestrates the entire ETL pipeline.

        Manages the flow of data through different stages: downloading raw resumes, extracting data, transforming to gold layer, and generating the Qdrant DB.

        Handles MinIO interactions for data persistence.

    minio_utils.py:

        Provides utility functions for interacting with MinIO.

        Includes functions to initialize the MinIO client, upload files to a specified bucket (creating the bucket if it doesn't exist), and download files from a bucket.

    qdrant_builder.py:

        Connects to a Qdrant service and generates the Qdrant database.

        Uses SentenceTransformerEmbeddings for dense vector embeddings and BM25Okapi for sparse vector (keyword) embeddings, combining them for hybrid search.

        Constructs a comprehensive text corpus from silver profiles for embedding generation and uses gold profiles as payload for Qdrant points.

        Uploads the trained BM25 model and token-to-index mapping to MinIO for later use in search.

    qdrant_builder(for_embeddings_only).py:

        A simplified version of the Qdrant builder focused solely on dense embeddings, without the hybrid search (BM25) components.

        Generates embeddings from silver_profiles and uses gold_profiles as the payload for Qdrant points.

    run_etl.py:

        A wrapper script to execute the main function from main_etl.py, providing a straightforward way to run the ETL process.

        Adds the project root to sys.path to ensure correct module imports.

    requirements.txt:

        Lists all the Python dependencies required for the project.

Data Flow

Raw resume files (PDF, DOCX, TXT) are downloaded from MinIO. These are then processed by data_loader.py to extract raw text content. llm_extractor.py uses an LLM to extract structured data (the "silver layer"). This silver layer data is then fed into gold_layer_transformer.py for cleaning, normalization, and deduplication annotation, creating the "gold layer." Both silver and gold layer data are used by qdrant_builder.py to generate embeddings and payloads for the Qdrant vector database, enabling advanced search functionalities. The gold layer data is also stored as a Parquet file in MinIO.

+---------------------------------------------------+
|                   Installation                    |
+---------------------------------------------------+

    Clone the Repository:
        git clone <repository_url>
        cd <repository_name>

    Create a Virtual Environment (recommended):
        python -m venv venv
        source venv/bin/activate # On Windows: venv\Scripts\activate

    Install Dependencies:
        pip install -r requirements.txt

    The requirements.txt includes:
    fastapi
    uvicorn
    python-multipart
    langchain-community
    langchain-core
    pypdf
    python-docx
    pandas
    numpy
    minio
    qdrant-client
    sentence-transformers
    tqdm
    requests
    rank_bm25
    nltk

    Set up Environment Variables:
    Configuration for the ETL pipeline is managed via environment variables, which are loaded by config.py. You should create a .env file in the project root to define these variables.

    Example .env file:
    MinIO Configuration

    MINIO_ENDPOINT="157.180.44.51:9000"
    MINIO_ACCESS_KEY="minioadmin"
    MINIO_SECRET_KEY="minioadmin"
    MINIO_SECURE="False" # Set to "True" for HTTPS
    MinIO Bucket Names

    RAW_RESUMES_BUCKET_NAME="rawresumes"
    SILVER_LAYER_BUCKET_NAME="outputjson" # Matches OUTPUT_JSON_FILE path
    REJECTED_PROFILES_BUCKET_NAME="rejected-resumes"
    MINIO_BUCKET_NAME="extracted" # For Gold Layer Parquet and Qdrant/BM25 models
    MinIO Object Names

    MINIO_OBJECT_NAME="gold_employee_profiles.parquet" # Gold Layer Parquet file name
    Local Directories and Files (defaults are set in config.py)
    RAW_RESUMES_DIR="./raw_resumes"
    OUTPUT_JSON_FILE="./outputjson/employee_profiles.json"
    GOLD_LAYER_PARQUET_FILE="./gold_employee_profiles.parquet"
    Ollama LLM Configuration

    OLLAMA_MODEL="llama3" # Or your preferred Ollama model
    Qdrant Configuration

    QDRANT_HOST="157.180.44.51"
    QDRANT_PORT=6333
    QDRANT_GRPC_PORT=6334 # gRPC port (optional)
    QDRANT_API_KEY="" # API key if Qdrant requires authentication
    QDRANT_COLLECTION_NAME="employee_profiles1" # Default collection name
    RAG Chain Configuration

    MAX_HISTORY_MESSAGES=6 # Maximum number of chat history messages to send to the LLM

    Note: Ensure MinIO and Qdrant services are running and accessible at the configured endpoints. The config.py file provides default values for these variables if they are not set in the .env file.

+---------------------------------------------------+
|                      Usage                        |
+---------------------------------------------------+

Running the ETL Process (Offline)

To run the ETL process directly, execute the run_etl.py script:
    python run_etl.py

This will start the data extraction, transformation, and Qdrant DB generation based on the files in your configured RAW_RESUMES_BUCKET_NAME MinIO bucket.

Running the ETL Trigger API

To start the FastAPI application:
    uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload

Once the API is running, you can trigger the ETL process by making a POST request to /trigger-etl. For example, using curl:
    curl -X POST http://localhost:8000/trigger-etl

You can also access the API documentation (Swagger UI) at http://localhost:8000/docs.

The ETL process triggered by the API will execute the main function from main_etl.py as a background task.
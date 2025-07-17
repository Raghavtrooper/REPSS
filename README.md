+-----------------------------------------------------------------------------+
|                                ETL Pipeline for                              |
|                          Resume Data Processing                             |
+-----------------------------------------------------------------------------+

+-----------------+
| 1. Introduction |
+-----------------+
This project implements an Extract, Transform, Load (ETL) pipeline designed
to process resume data from various document formats (PDF, DOCX, TXT).
It leverages Large Language Models (LLMs) for structured data extraction,
performs data cleaning and deduplication, and stores the processed
information in a vector database (Qdrant) for efficient search and retrieval.
MinIO is used for object storage across different stages of the pipeline.

+-----------------------+
| 2. Project Structure  |
+-----------------------+
.
├── etl/
│   ├── api_main.py                 # FastAPI application to trigger ETL
│   ├── data_loader.py              # Handles loading content from PDF, DOCX, TXT
│   ├── gold_layer_transformer.py   # Transforms silver layer to gold layer (deduplication, normalization)
│   ├── llm_extractor.py            # Extracts structured data using Ollama LLM
│   ├── main_etl.py                 # Main ETL orchestration script
│   ├── minio_utils.py              # Utility functions for MinIO interactions
│   ├── qdrant_builder.py           # Builds and populates the Qdrant vector database (hybrid search)
│   └── qdrant_builder(for_embeddings_only).py # Alternative Qdrant builder (dense embeddings only)
├── shared/
│   └── config.py                   # Configuration variables (MinIO, Qdrant, Ollama)
├── run_etl.py                      # Script to run the main ETL process
├── requirements.txt                # Python dependencies
└── utility.py                      # Utility functions (e.g., for path handling)


+-------------------------+
| 3. Key Features         |
+-------------------------+
* **Multi-format Support**: Processes resumes from PDF, DOCX, and TXT files.
* **LLM-Powered Extraction**: Uses Ollama to extract structured data (name, email, skills, experience, etc.).
* **Data Normalization**: Cleans and normalizes extracted data for consistency.
* **Duplicate Annotation**: Identifies and annotates duplicate profiles without dropping them.
* **MinIO Integration**: Stores raw, silver, gold, rejected, and stale data in MinIO buckets.
* **Qdrant Vector Database**: Builds a Qdrant collection with hybrid vectors (dense and sparse) for advanced search.
* **API Trigger**: Provides a FastAPI endpoint to trigger the ETL process programmatically.
* **Stale Profile Detection**: Identifies and handles profiles whose source files are no longer present.
* **Contact Info Validation**: Rejects profiles missing both email and phone number.

+--------------------------+
| 4. Setup and Installation|
+--------------------------+

Before running the ETL pipeline, ensure you have the following services
running and accessible:

* **Ollama**: For LLM inference.
* **MinIO**: For object storage.
* **Qdrant**: For the vector database.

You can typically run these using Docker.

+-------------------------------------+
| 4.1. Prerequisites (Docker Commands)|
+-------------------------------------+

+-----------------------------------------------------------------------------+
| Start Ollama (e.g., with 'llama2' model):                                   |
| --------------------------------------------------------------------------- |
| ```bash                                                                    |
| docker run -d -v ollama:/root/.ollama -p 127.0.0.1:11434:11434 --name ollama ollama/ollama |
| ollama pull llama2                                                          |
| # Or pull the model specified in shared/config.py (e.g., 'nomic-embed-text')|
| ollama pull nomic-embed-text                                                |
| ```                                                                         |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| Start MinIO:                                                                |
| --------------------------------------------------------------------------- |
| ```bash                                                                    |
| docker run -p 9000:9000 -p 9001:9001 --name minio1 \                         |
|   -v /mnt/data:/data \                                                      |
|   -e "MINIO_ROOT_USER=minioadmin" \                                         |
|   -e "MINIO_ROOT_PASSWORD=minioadmin" \                                     |
|   minio/minio server /data --console-address ":9001"                        |
| ```                                                                         |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| Start Qdrant:                                                               |
| --------------------------------------------------------------------------- |
| ```bash                                                                    |
| docker run -p 6333:6333 -p 6334:6334 \                                      |
|   -v $(pwd)/qdrant_storage:/qdrant/storage \                                |
|   qdrant/qdrant                                                             |
| ```                                                                         |
+-----------------------------------------------------------------------------+

+---------------------------------+
| 4.2. Python Environment Setup   |
+---------------------------------+

1.  **Clone the repository (if not already done):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download NLTK punkt tokenizer data (required by `rank_bm25`):**
    ```python
    import nltk
    nltk.download('punkt')
    ```
    (Run this in your Python environment once)

+-------------------------+
| 5. Configuration        |
+-------------------------+
Edit `shared/config.py` to adjust paths, MinIO credentials, Ollama model name,
and Qdrant connection details if they differ from the defaults.

+-------------------------+
| 6. Usage                |
+-------------------------+

+---------------------------------+
| 6.1. Running the ETL Pipeline   |
+---------------------------------+

To run the full ETL pipeline:

1.  **Place your raw resume files** (PDF, DOCX, TXT) into the MinIO bucket
    specified by `RAW_RESUMES_BUCKET_NAME` in `shared/config.py` (default:
    `raw-resumes`). You can use the MinIO UI (usually at `http://localhost:9001`)
    or MinIO client tools to upload.

2.  **Execute the ETL script:**
    ```bash
    python run_etl.py
    ```
    This script will:
    * Download new resumes from MinIO.
    * Load content from each resume.
    * Extract structured data using the configured Ollama LLM.
    * Perform gold layer transformation (cleaning, normalization, deduplication annotation).
    * Upload processed data to various MinIO buckets (silver, gold, rejected, stale).
    * Build and populate the Qdrant vector database with hybrid embeddings.

+---------------------------------+
| 6.2. Running the ETL Trigger API|
+---------------------------------+

The `api_main.py` file exposes a FastAPI endpoint to trigger the ETL process.

1.  **Start the FastAPI application:**
    ```bash
    uvicorn etl.api_main:app --host 0.0.0.0 --port 8000 --reload
    ```
    (The `--reload` flag is useful for development; remove for production.)

2.  **Access the API:**
    * **Root endpoint:** `http://localhost:8000/`
    * **Trigger ETL:** Send a POST request to `http://localhost:8000/trigger-etl`

    You can use `curl` or a tool like Postman/Insomnia to test:

    +-----------------------------------------------------------------------------+
    | Trigger ETL via curl:                                                       |
    | --------------------------------------------------------------------------- |
    | ```bash                                                                    |
    | curl -X POST http://localhost:8000/trigger-etl                              |
    | ```                                                                         |
    +-----------------------------------------------------------------------------+

+-------------------------+
| 7. Important Notes      |
+-------------------------+
* **Ollama Model**: Ensure the Ollama model specified in `shared/config.py`
    (`OLLAMA_MODEL`) is pulled and running in your Ollama instance.
* **MinIO Buckets**: The pipeline automatically creates necessary MinIO buckets
    if they don't exist.
* **Qdrant Collection**: The Qdrant collection is recreated on each run of
    `generate_qdrant_db` to ensure a clean state.
* **Error Handling**: The scripts include basic error handling and logging
    to the console. For production, consider more robust logging mechanisms.
* **Performance**: For very large datasets, consider optimizing the LLM
    extraction (e.g., batching, using a more powerful model) and MinIO/Qdrant
    interactions. The `asyncio.to_thread` in `api_main.py` helps prevent
    blocking the API for long-running ETL tasks.

+-------------------------+
| 8. License              |
+-------------------------+
[Specify your license here, e.g., MIT, Apache 2.0, etc.]

+-------------------------+
| 9. Contact              |
+-------------------------+
For any questions or issues, please contact [Your Name/Email/GitHub Profile].

+-------------------------+
| 10. Program Details     |
+-------------------------+

+-----------------------------------------------------------------------------+
| `etl/api_main.py`                                                           |
| --------------------------------------------------------------------------- |
| This FastAPI application provides a web interface to trigger the ETL        |
| process. It exposes a root endpoint (`/`) and a POST endpoint (`/trigger-etl`)|
| that asynchronously calls the main ETL function, preventing the API from    |
| blocking during long-running operations.                                    |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| `etl/data_loader.py`                                                        |
| --------------------------------------------------------------------------- |
| Responsible for loading textual content from various document formats.      |
| It supports PDF, DOCX, and TXT files. For PDFs, it also includes logic to   |
| detect if the document contains embedded images, which is stored as metadata.|
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| `etl/gold_layer_transformer.py`                                             |
| --------------------------------------------------------------------------- |
| This module takes the "silver layer" (raw extracted structured data) and    |
| transforms it into the "gold layer." This involves extensive data cleaning,|
| normalization (e.g., standardizing skill names, phone numbers, locations), |
| and crucially, annotating duplicate profiles based on email and name/phone |
| combinations, rather than simply removing them.                             |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| `etl/llm_extractor.py`                                                      |
| --------------------------------------------------------------------------- |
| Handles the core LLM-based data extraction. It initializes the Ollama LLM, |
| defines a detailed prompt for structured resume data extraction (including |
| strict normalization rules), and processes raw resume text to produce      |
| a clean, standardized JSON output. It also includes robust JSON parsing    |
| and healing logic.                                                          |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| `etl/main_etl.py`                                                           |
| --------------------------------------------------------------------------- |
| The central orchestration script for the entire ETL pipeline. It manages   |
| the flow from downloading raw resumes from MinIO, processing them through |
| LLM extraction, transforming them to the gold layer, filtering based on    |
| contact info, uploading intermediate and final data to MinIO, and finally,|
| building the Qdrant vector database.                                        |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| `etl/minio_utils.py`                                                        |
| --------------------------------------------------------------------------- |
| Provides utility functions for interacting with MinIO object storage.       |
| This includes initializing the MinIO client, uploading files to specified  |
| buckets (creating them if necessary), and downloading files from MinIO.     |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| `etl/qdrant_builder.py`                                                     |
| --------------------------------------------------------------------------- |
| This module is responsible for building and populating the Qdrant vector   |
| database. It generates *hybrid* embeddings (dense using Sentence Transformers|
| and sparse using BM25) from the processed resume data. It handles collection|
| creation/recreation and upserting points, and also uploads the BM25 model  |
| and token-to-index mapping to MinIO for later retrieval during search.      |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| `etl/qdrant_builder(for_embeddings_only).py`                                |
| --------------------------------------------------------------------------- |
| An alternative Qdrant builder that focuses solely on generating and        |
| populating the Qdrant database with *dense* embeddings. It uses Sentence   |
| Transformers to create vectors from the silver layer profiles and stores   |
| the gold layer profiles as payloads.                                        |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| `shared/config.py`                                                          |
| --------------------------------------------------------------------------- |
| This file centralizes all configuration settings for the ETL pipeline.     |
| It loads environment variables from a `.env` file and defines critical     |
| parameters such as:                                                         |
| - **File Paths**: Local directories for raw resumes, and output JSON/Parquet.|
| - **Ollama Model**: The name of the LLM model used for extraction.         |
| - **MinIO Configuration**: Endpoint, access keys, secret keys, and bucket  |
|   names for various stages (raw, silver, gold, rejected, stale profiles).   |
| - **Qdrant Configuration**: Host, port, API key, and collection name for   |
|   the vector database.                                                      |
| - **RAG Chain Configuration**: Parameters like the maximum chat history    |
|   messages for Retrieval Augmented Generation (RAG) chains.                 |
| This centralized approach ensures easy management and modification of      |
| pipeline settings.                                                          |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| `run_etl.py`                                                                |
| --------------------------------------------------------------------------- |
| A simple wrapper script to execute the `main` function of the ETL pipeline.|
| It ensures that the project root is correctly added to the Python path for |
| module imports.                                                             |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| `requirements.txt`                                                          |
| --------------------------------------------------------------------------- |
| Lists all Python package dependencies required for the project. These      |
| packages can be installed using `pip install -r requirements.txt`.         |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| `utility.py`                                                                |
| --------------------------------------------------------------------------- |
| Contains general utility functions that might be used across different     |
| parts of the project, such as functions to correctly determine and add the |
| project root to `sys.path`.                                                 |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+


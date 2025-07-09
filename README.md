# Resume RAG System

This project implements a Retrieval Augmented Generation (RAG) system for querying employee profiles (resumes). It consists of an ETL (Extract, Transform, Load) pipeline to process raw resume documents into a structured format, enrich them, and store them in a vector database (ChromaDB) and a gold layer (Parquet file). A Streamlit application then uses this data to answer questions about employee skills and profiles using an LLM.
Project Structure

### The project is organized into logical directories to separate concerns:

```text
.
|──  api/
|   |── api_helper.py        # Functions provided for the rest_controller
|   └── rest_controller.py   # Server module
├── app/
│   ├── main_app.py           # Main Streamlit application
│   ├── rag_chain.py          # Defines the RAG chain using LangChain
│   ├── ui_components.py      # Contains reusable UI components for Streamlit
│   └── vector_utils.py       # Initializes and loads the ChromaDB vector store
├── etl/
│   ├── chroma_builder.py     # Handles ChromaDB creation and persistence
│   ├── data_loader.py        # Utility for loading content from various document types (PDF, DOCX, TXT)
│   ├── gold_layer_transformer.py # Transforms silver layer data to gold layer, including duplicate annotation
│   ├── llm_extractor.py      # Uses LLM to extract structured data from raw resumes
│   ├── main_etl.py           # Orchestrates the entire ETL pipeline
│   └── minio_utils.py        # Utilities for interacting with MinIO storage
├── raw_resumes/              # Directory to store raw resume files (PDF, DOCX, TXT)
│   └── example_resume.txt    # (Example placeholder)
├── shared/
│   └── config.py             # Centralized configuration file for environment variables and paths
├── .env                      # Environment variables for configuration (e.g., MinIO credentials, Ollama model)
├── employee_profiles.json    # (Generated) Silver layer: structured data after LLM extraction
├── gold_employee_profiles.parquet # (Generated) Gold layer: cleaned, enriched, and annotated data
├── chroma_db/                # (Generated) Persistent directory for ChromaDB vector store
|── run_api.py                # Server launcher
├── run_app.py                # Script to launch the Streamlit application
└── run_etl.py                # Script to execute the ETL pipeline
```

## Functionality
### ETL Pipeline (etl/)

The ETL pipeline is responsible for preparing the data for the RAG system:

    Data Loading (data_loader.py): Reads raw resume files (PDF, DOCX, TXT) from the raw_resumes/ directory and extracts their text content.

    LLM Extraction (llm_extractor.py): Uses an Ollama-based LLM to parse the raw text content of resumes and extract structured information (e.g., name, title, skills, experience, education, contact info) into a JSON format (Silver Layer).

    Gold Layer Transformation (gold_layer_transformer.py): Takes the extracted structured data, performs cleaning, normalization, and importantly, annotates duplicate records instead of merging or dropping them. This creates the "Gold Layer" data.

    MinIO Upload (minio_utils.py): Uploads the generated Gold Layer Parquet file to a MinIO object storage bucket for persistence and accessibility.

    ChromaDB Building (chroma_builder.py): Creates a vector database (ChromaDB) from the Gold Layer profiles using SentenceTransformerEmbeddings. The textual content and metadata from the profiles are embedded and stored, enabling semantic search.

    Orchestration (main_etl.py): This script ties together all the ETL steps, managing the flow from raw documents to the persistent ChromaDB and Gold Layer Parquet file.

### RAG Application (app/)

The Streamlit application provides a user interface for interacting with the processed data:

    Main Application (main_app.py): The core Streamlit application. It loads the pre-built ChromaDB, initializes the RAG chain, manages chat history, and displays responses and retrieved documents.

    RAG Chain (rag_chain.py): Defines the LangChain-based RAG chain. This chain takes a user's query and chat history, retrieves relevant documents from ChromaDB, and then feeds them to an Ollama LLM to generate a coherent and informative response. It includes a custom retriever logic to handle explicit context or use the vector store as needed.

    UI Components (ui_components.py): Contains helper functions for common Streamlit UI elements, such as clearing chat history.

    Vector Utilities (vector_utils.py): Initializes or loads the ChromaDB vector store for the Streamlit application.

### Tech Stack

    Python 3.x: The primary programming language.

    Streamlit: For building the interactive web application.

    LangChain: Framework for developing applications powered by language models, used for the RAG pipeline.

    Ollama: For running open-source large language models locally (e.g., Llama3).

    ChromaDB: A lightweight, open-source vector database used for storing and querying document embeddings.

    Sentence Transformers: For generating embeddings (specifically all-MiniLM-L6-v2).

    Pandas: For data manipulation and creating the Gold Layer DataFrame.

    MinIO: Open-source object storage suite compatible with Amazon S3 APIs, used for persisting the Gold Layer Parquet file.

    python-dotenv: For managing environment variables.

    PyPDFLoader, Docx2txtLoader, TextLoader (from langchain_community): For loading content from various document formats.

### Dependencies

You can install the Python dependencies using pip and the requirements.txt file (which you would typically generate or create based on the above tech stack).

Example requirements.txt:

streamlit
langchain
langchain-community
ollama
chromadb
sentence-transformers
pandas
python-dotenv
minio
pydantic
fastapi
uvicorn

### Setup and Running the Project
### Prerequisites

    Python 3.x: Ensure Python is installed.

    Ollama: Download and install Ollama from https://ollama.com/.

        Pull the required LLM model (e.g., llama3): ollama pull llama3

    MinIO (Optional but Recommended): Set up a MinIO server. You can run it locally via Docker:

    docker run -p 9000:9000 -p 9001:9001 --name minio1 \
      -e "MINIO_ROOT_USER=admin" \
      -e "MINIO_ROOT_PASSWORD=password123" \
      quay.io/minio/minio server /data --console-address ":9001"

### Configuration

    Create a .env file in the project root with your configuration:

    OLLAMA_MODEL=llama3
    MINIO_ENDPOINT=localhost:9000
    MINIO_ACCESS_KEY=admin
    MINIO_SECRET_KEY=password123
    MINIO_SECURE=False
    MINIO_BUCKET_NAME=extracted
    MINIO_OBJECT_NAME=gold_employee_profiles.parquet

    Adjust values as per your setup. MINIO_SECURE should be True if using HTTPS.

### Steps to Run

    1. Install Dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    (Assuming you create a requirements.txt from the listed dependencies)

    2. Place Raw Resumes:
    Put your PDF, DOCX, or TXT resume files into the raw_resumes/ directory.

    3. Run the ETL Pipeline:
    This step processes the raw resumes, extracts data, transforms it, and builds the ChromaDB and Gold Layer Parquet file.

    ```bash
    python run_etl.py
    ```

    You should see console output indicating the progress of extraction, transformation, and ChromaDB creation. This will also upload the gold_employee_profiles.parquet to your MinIO instance.

    4. Run the Streamlit Application:
    Once the ETL is complete, launch the RAG application:

    ```bash
    python run_app.py
    ```

    This will open the Streamlit application in your web browser, typically at http://localhost:8501.

### Sever setup:

* Make sure the server dependencies fastapi, uvicorn and python-multipart are present
* Simply execute the run_apy.py

### Pipeline Flow

    Input: Raw resume files are placed in the raw_resumes/ directory.

    ETL Execution (run_etl.py -> etl/main_etl.py):

        data_loader.py reads content from each raw resume.

        llm_extractor.py (using Ollama) extracts structured data (Silver Layer) and saves it to employee_profiles.json.

        gold_layer_transformer.py cleans, normalizes, and annotates duplicates in the Silver Layer data, creating the Gold Layer DataFrame.

        The Gold Layer DataFrame is saved as gold_employee_profiles.parquet.

        minio_utils.py uploads gold_employee_profiles.parquet to the configured MinIO bucket.

        chroma_builder.py creates and persists chroma_db/ from the Gold Layer profiles, embedding their content.

    Application Launch (run_app.py -> app/main_app.py):

        app/main_app.py loads the persistent chroma_db/ using app/vector_utils.py.

        It initializes the RAG chain from app/rag_chain.py.

    User Interaction (Streamlit UI):

        Users enter queries in the Streamlit chat interface.

        The query and chat history are sent to the RAG chain.

    RAG Chain Processing (app/rag_chain.py):

        The chain determines if external context is provided or if it needs to retrieve documents.

        It retrieves relevant documents (employee profiles) from chroma_db/ based on the user's query.

        The retrieved documents and the user's query (and chat history) are sent to the Ollama LLM.

        The LLM generates a comprehensive answer based on the provided context.

    Output: The LLM's answer and the source documents (employee profiles) are displayed in the Streamlit UI.

This structured approach ensures efficient data processing, maintainable code, and a robust RAG system for querying employee information.


Development note: as of 1/7/2025(dd/mm/yyyy) haystack implemntation in the ETL pipeline could not be done due to version compatability issues.
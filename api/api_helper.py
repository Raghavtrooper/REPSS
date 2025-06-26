# api_helper.py
import threading
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from typing import List, Set # Import Set for previously_shown_doc_ids
from pathlib import Path
from pydantic import BaseModel

# Correctly import functions from app.main_app
# Assuming main_app is located at 'app/main_app.py' relative to project root
from app.main_app import (
    process_employee_query,
    load_gold_layer_data,
    get_llm_and_embeddings,
    get_vectorstore_instance,
    get_rag_chain_instance,
    get_show_more_detector
)

# Assuming etl.main_etl.main is needed for /upload endpoint
from etl.main_etl import main as etl_main # Renamed to avoid conflict if any


UPLOAD_DIR = Path("raw_resumes")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global variables for loaded resources
gold_df_global = None
parser_llm_global = None
embedding_function_global = None
vectorstore_instance_global = None
rag_chain_instance_global = None
show_more_detector_data_global = None

# Function to initialize resources
def initialize_api_resources():
    """Initializes global resources for the API."""
    global gold_df_global, parser_llm_global, embedding_function_global, \
           vectorstore_instance_global, rag_chain_instance_global, show_more_detector_data_global
    
    print("Initializing API resources...")
    gold_df_global = load_gold_layer_data()
    parser_llm_global, embedding_function_global = get_llm_and_embeddings()
    vectorstore_instance_global = get_vectorstore_instance()
    rag_chain_instance_global = get_rag_chain_instance(vectorstore_instance_global)
    show_more_detector_data_global = get_show_more_detector(embedding_function_global)
    print("API resources initialized.")


class QueryInput(BaseModel):
    query: str
    chat_history: List[dict] = []  # To store previous messages (role, content)
    previously_shown_doc_ids: List[str] = [] # To store IDs of previously shown documents


# This function now uses the new process_employee_query from main_app.py
def process_query(data: QueryInput) -> dict:
    if not data.query.strip():
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Query string is empty"
        )
    print(f"Received query: {data.query}")

    # Ensure resources are loaded. In a real app, you might have a check
    # or rely on the startup event to guarantee loading.
    if gold_df_global is None:
        # This case should ideally not happen if startup event works, but as a fallback
        print("Warning: API resources not loaded. Attempting to load now.")
        initialize_api_resources()

    # Call the refactored core processing function
    response_data = process_employee_query(
        user_query=data.query,
        gold_df=gold_df_global,
        vectorstore_instance=vectorstore_instance_global,
        rag_chain_instance=rag_chain_instance_global,
        ollama_parser_llm=parser_llm_global,
        embedding_function=embedding_function_global,
        show_more_detector_data=show_more_detector_data_global,
        chat_history=data.chat_history, # Pass history from request
        previously_shown_doc_ids=set(data.previously_shown_doc_ids) # Convert to set for function
    )
    
    # Ensure previously_shown_doc_ids is returned as a list for Pydantic
    response_data["updated_previously_shown_doc_ids"] = list(response_data.get("updated_previously_shown_doc_ids", []))

    return response_data


async def handle_upload(files: List[UploadFile]) -> dict:
    saved_files = []

    for file in files:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"{file.filename} is empty"
            )

        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(contents)

        saved_files.append({
            "filename": file.filename,
            "type": file.content_type,
            "size": file_path.stat().st_size
        })

    return {
        "total": len(saved_files),
        "uploaded_files": saved_files
    }


async def upload_files_helper(files: List[UploadFile]) -> dict:
    try:
        res = await handle_upload(files)
        # Call the ETL main function in a separate thread
        threading.Thread(target=etl_main, daemon=True).start()
        return res
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while saving the file: {str(e)}"
        )


async def receive_query_helper(data: QueryInput):
    try:
        return process_query(data) # Pass the full QueryInput object
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        print(f"Error in receive_query_helper: {e}")
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Unexpected error: {str(e)}"}
        )
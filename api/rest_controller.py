# rest_controller.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_400_BAD_REQUEST
from typing import List

from .api_helper import (
    QueryInput,
    upload_files_helper,
    receive_query_helper,
    initialize_api_resources # Import the new initialization function
)

app = FastAPI()

# Add an event handler to initialize resources at startup
@app.on_event("startup")
async def startup_event():
    initialize_api_resources()


# Exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": "Invalid file or request format", "errors": exc.errors()},
    )

@app.post("/query")
async def receive_query(data: QueryInput):
    # The data object now contains query, chat_history, and previously_shown_doc_ids
    return await receive_query_helper(data)


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    return await upload_files_helper(files)
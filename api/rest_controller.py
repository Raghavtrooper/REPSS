from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_400_BAD_REQUEST
from typing import List

from .api_helper import (
    QueryInput,
    upload_files_helper,
    receive_query_helper
)

app = FastAPI()

# Exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": "Invalid file or request format", "errors": exc.errors()},
    )

@app.post("/query")
async def receive_query(data: QueryInput):
    return await receive_query_helper(data)


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    return await upload_files_helper(files)
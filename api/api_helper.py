
import threading
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from typing import List
from pathlib import Path
from pydantic import BaseModel
from ..etl.main_etl import main  

UPLOAD_DIR = Path("raw_resumes")
UPLOAD_DIR.mkdir(exist_ok=True)

class QueryInput(BaseModel):
    query: str


# later used in the recieve_query_helper
# used to check the query
def process_query(query: str) -> dict:
    if not query.strip():
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Query string is empty"
        )
    print(query)
    return {
        "received_query": query,
        "length": len(query)
    }



# later used in the upload_file_helper, 
# used to save files on the sys
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


# endpoint function
async def upload_files_helper(files: List[UploadFile]) -> dict:
    try:
        res = await handle_upload(files)
        threading.Thread(target=main, daemon=True).start()
        return res
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while saving the file: {str(e)}"
        )


# endpoint function
# ? logical cache loading and off-loading required
async def receive_query_helper(data: QueryInput):
    try:
        return process_query(data.query)
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Unexpected error: {str(e)}"}
        )

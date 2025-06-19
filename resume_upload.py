from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
from pathlib import Path
import shutil
import zipfile

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx"}


def save_file(file: UploadFile) -> str:
    """Save individual file if it has allowed extension."""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file.filename


def process_zip(file: UploadFile) -> List[str]:
    """Extract zip file and save allowed files."""
    saved_files = []
    with zipfile.ZipFile(file.file) as zip_ref:
        for zip_info in zip_ref.infolist():
            filename = zip_info.filename
            file_ext = Path(filename).suffix.lower()

            if file_ext in ALLOWED_EXTENSIONS:
                extracted_path = UPLOAD_DIR / Path(filename).name
                with zip_ref.open(zip_info) as source, open(extracted_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                saved_files.append(Path(filename).name)

    return saved_files


@router.post("/upload-resumes")
async def upload_resumes(files: List[UploadFile] = File(...)):
    saved_files = []

    for file in files:
        file_ext = Path(file.filename).suffix.lower()

        if file_ext == ".zip":
            extracted_files = process_zip(file)
            saved_files.extend(extracted_files)
        elif file_ext in ALLOWED_EXTENSIONS:
            filename = save_file(file)
            saved_files.append(filename)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

    return {
        "message": "Resumes uploaded successfully.",
        "files": saved_files
    }

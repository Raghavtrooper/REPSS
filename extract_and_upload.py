import os
import re
import pdfplumber
from docx import Document
import shutil
from minio import Minio
from minio.error import S3Error

# Regex pattern for 10-digit Indian phone numbers
phone_pattern = re.compile(r'(\+91[-\s]?)?\b\d{10}\b')

# MinIO config
minio_client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket_name = "new-resume"

# Create bucket if it doesn't exist
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)
    print(f"Bucket '{bucket_name}' created.")
else:
    print(f"Bucket '{bucket_name}' already exists.")

def sanitize_filename(name):
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "")
    return name

def extract_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_name_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

    lines = text.strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(keyword in line.lower() for keyword in ["objective", "summary", "experience", "career", "contact", "email", "phone"]):
            continue
        if any(char.isdigit() for char in line):
            continue
        if len(line.split()) <= 1:
            continue
        return sanitize_filename(line.replace(" ", "_"))

    return "UnknownName"

def extract_from_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_name_from_docx(text):
    lines = text.strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(keyword in line.lower() for keyword in ["summary", "experience", "profile", "career", "contact", "phone", "email"]):
            continue
        if any(char.isdigit() for char in line):
            continue
        if len(line.split()) <= 1:
            continue
        return sanitize_filename(line.replace(" ", "_"))

    return "UnknownName"

def rename_and_upload_file(old_path, new_name):
    directory = os.path.dirname(old_path)
    extension = os.path.splitext(old_path)[1]
    new_path = os.path.join(directory, f"{new_name}{extension}")

    # Avoid filename conflicts
    counter = 1
    while os.path.exists(new_path):
        new_path = os.path.join(directory, f"{new_name}_{counter}{extension}")
        counter += 1

    shutil.move(old_path, new_path)
    print(f"Renamed: {old_path} → {new_path}")

    # Upload to MinIO
    try:
        minio_client.fput_object(
            bucket_name,
            os.path.basename(new_path),
            new_path,
            content_type="application/pdf" if new_path.lower().endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        print(f"Uploaded to MinIO: {new_path}")
    except S3Error as err:
        print(f"Failed to upload {new_path}: {err}")

def process_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        text = extract_from_pdf(file_path)
        name = extract_name_from_pdf(file_path)
    elif ext == ".docx":
        text = extract_from_docx(file_path)
        name = extract_name_from_docx(text)
    else:
        print(f"Unsupported file format: {file_path}")
        return

    phone_match = phone_pattern.search(text)
    phone = phone_match.group(0) if phone_match else "UnknownNumber"

    new_name = f"{name}_{phone}"
    rename_and_upload_file(file_path, new_name)

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            process_file(file_path)

# 🔴 Update this to your resumes folder
folder_path = r"C:\Users\Aryan Raj Chauhan\REPSS\raw_resumes"
process_folder(folder_path)

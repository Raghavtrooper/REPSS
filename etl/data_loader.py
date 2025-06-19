import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader

def load_document_content(file_path):
    """
    Loads text content from various document types.
    Returns the content as a single string.
    """
    _, file_extension = os.path.splitext(file_path)
    content = ""
    try:
        if file_extension.lower() == ".pdf":
            print(f"    Loading PDF: {os.path.basename(file_path)}")
            loader = PyPDFLoader(file_path)
            # PyPDFLoader returns a list of Document objects, one per page.
            # We join their page_content to get the full text.
            docs = loader.load()
            content = "\n".join([doc.page_content for doc in docs])
        elif file_extension.lower() == ".docx":
            print(f"    Loading DOCX: {os.path.basename(file_path)}")
            loader = Docx2txtLoader(file_path)
            # Docx2txtLoader also returns a list of Document objects
            docs = loader.load()
            content = "\n".join([doc.page_content for doc in docs])
        elif file_extension.lower() == ".txt":
            print(f"    Loading TXT: {os.path.basename(file_path)}")
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            content = "\n".join([doc.page_content for doc in docs])
        else:
            print(f"    Skipping unsupported file type: {file_extension} for {os.path.basename(file_path)}")
            return None
        return content
    except Exception as e:
        print(f"    Error loading content from {os.path.basename(file_path)}: {e}")
        return None

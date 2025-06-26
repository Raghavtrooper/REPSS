import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
import pypdf # New import for PDF image detection

def load_document_content(file_path):
    """
    Loads text content from various document types and detects if a PDF contains images.
    Returns a tuple: (content_as_string, metadata_dict).
    metadata_dict includes 'has_photo' for PDFs, otherwise an empty dict.
    """
    _, file_extension = os.path.splitext(file_path)
    content = ""
    metadata = {} # Initialize metadata dictionary

    try:
        if file_extension.lower() == ".pdf":
            print(f"    Loading PDF: {os.path.basename(file_path)}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            content = "\n".join([doc.page_content for doc in docs])

            # --- Image Detection for PDFs ---
            has_image = False
            try:
                with open(file_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        if '/XObject' in page['/Resources']:
                            xObjects = page['/Resources']['/XObject'].get_object()
                            for obj in xObjects:
                                # Check if the XObject is an image
                                if xObjects[obj]['/Subtype'] == '/Image':
                                    has_image = True
                                    break # Found an image, no need to check further objects on this page
                        if has_image:
                            break # Found an image, no need to check further pages
            except Exception as e:
                # Log a warning if image detection fails, but don't stop processing
                print(f"    Warning: Could not detect images in PDF {os.path.basename(file_path)}: {e}")
            metadata['has_photo'] = has_image # Store the flag
            # --- End Image Detection ---

        elif file_extension.lower() == ".docx":
            print(f"    Loading DOCX: {os.path.basename(file_path)}")
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            content = "\n".join([doc.page_content for doc in docs])
            # Default to False for DOCX as image detection is more complex and not implemented here
            metadata['has_photo'] = False 

        elif file_extension.lower() == ".txt":
            print(f"    Loading TXT: {os.path.basename(file_path)}")
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            content = "\n".join([doc.page_content for doc in docs])
            # TXT files do not contain embedded images
            metadata['has_photo'] = False 

        else:
            print(f"    Skipping unsupported file type: {file_extension} for {os.path.basename(file_path)}")
            return None, {} # Return None content and empty metadata for unsupported types
        
        return content, metadata
    except Exception as e:
        print(f"    Error loading content from {os.path.basename(file_path)}: {e}")
        return None, {} # Return None content and empty metadata on error

import os
import shutil
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def generate_chroma_db(structured_profiles, chroma_dir):
    """
    Generates and persists the ChromaDB from structured employee profiles.
    """
    if os.path.exists(chroma_dir):
        print(f"Clearing existing ChromaDB directory: {chroma_dir}")
        shutil.rmtree(chroma_dir)
    os.makedirs(chroma_dir, exist_ok=True)
    
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    documents = []
    for profile in structured_profiles:
        # Combine relevant fields into a single text string for the embedding model to process
        # This page_content is what the vector store uses for similarity search
        skills_str = ", ".join(profile.get('skills', [])) if profile.get('skills') else "Not Found"
        
        content = f"---BEGIN EMPLOYEE PROFILE---\n" \
                  f"ID: {profile.get('id', 'N/A')}\n" \
                  f"Name: {profile.get('name', 'N/A')}\n" \
                  f"Title: {profile.get('title', 'N/A')}\n" \
                  f"Department: {profile.get('department', 'N/A')}\n" \
                  f"Skills: {skills_str}\n" \
                  f"Experience: {profile.get('experience', 'N/A')}\n" \
                  f"Education: {profile.get('education', 'N/A')}\n" \
                  f"Contact Info: {profile.get('contact_info', 'N/A')}\n" \
                  f"---END EMPLOYEE PROFILE---\n"
        
        # Store all original structured profile details as metadata.
        # Ensure all metadata values are simple types (str, int, float, bool, None).
        metadata = {
            "id": profile.get("id", "N/A"),
            "name": profile.get("name", "N/A"),
            "title": profile.get("title", "N/A"),
            "department": profile.get("department", "N/A"),
            "skills": skills_str, # Store as string for metadata
            "experience": profile.get("experience", "N/A"),
            "education": profile.get("education", "N/A"),
            "contact_info": profile.get("contact_info", "N/A"),
            # Store the original filename if it was carried through, useful for debugging
            "original_filename": profile.get("_original_filename", "N/A") 
        }
        documents.append(Document(page_content=content, metadata=metadata))

    if not documents:
        print("No documents to add to ChromaDB. Aborting ChromaDB creation.")
        return

    print(f"Creating ChromaDB with {len(documents)} documents...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=chroma_dir
    )
    vectorstore.persist() # Save the collection to disk
    print(f"ChromaDB created and persisted to '{chroma_dir}' with {len(documents)} documents.")

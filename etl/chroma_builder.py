import os
import shutil
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata # ADDED THIS IMPORT

def generate_chroma_db(structured_profiles, chroma_dir):
    """
    Generates and persists the ChromaDB from structured employee profiles.
    Updated to use new fields (location, objective, qualifications_summary, experience_summary, has_photo)
    and remove old ones (title, department, experience, education).
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
        
        # Include a note about duplicates in the page_content for LLM awareness
        duplicate_status = ""
        if profile.get('_is_master_record') and profile.get('_duplicate_count', 1) > 1:
            duplicate_status = f" (NOTE: This person has {profile['_duplicate_count']} associated resumes. Group ID: {profile['_duplicate_group_id']})"
        elif not profile.get('_is_master_record') and profile.get('_duplicate_group_id'):
            duplicate_status = f" (NOTE: This is an associated resume for a person with multiple entries. Group ID: {profile['_duplicate_group_id']})"

        content = f"---BEGIN EMPLOYEE PROFILE---\n" \
                  f"ID: {profile.get('id', 'N/A')}\n" \
                  f"Name: {profile.get('name', 'N/A')}{duplicate_status}\n" \
                  f"Email: {profile.get('email_id', 'N/A')}\n" \
                  f"Phone: {profile.get('phone_number', 'N/A')}\n" \
                  f"Location: {profile.get('location', 'N/A')}\n" \
                  f"Objective: {profile.get('objective', 'N/A')}\n" \
                  f"Skills: {skills_str}\n" \
                  f"Qualifications Summary: {profile.get('qualifications_summary', 'N/A')}\n" \
                  f"Experience Summary: {profile.get('experience_summary', 'N/A')}\n" \
                  f"Has Photo: {profile.get('has_photo', False)}\n" \
                  f"---END EMPLOYEE PROFILE---\n"
        
        # Store all original structured profile details as metadata.
        # Ensure all metadata values are simple types (str, int, float, bool, None).
        metadata = {
            "id": profile.get("id", "N/A"),
            "name": profile.get("name", "N/A"),
            "email_id": profile.get("email_id", "N/A"),
            "phone_number": profile.get("phone_number", "N/A"),
            "location": profile.get("location", "N/A"),
            "objective": profile.get("objective", "N/A"),
            "skills": skills_str, # Store as string for metadata
            "qualifications_summary": profile.get('qualifications_summary', 'N/A'),
            "experience_summary": profile.get('experience_summary', 'N/A'),
            "has_photo": profile.get('has_photo', False),
            "original_filename": profile.get("_original_filename", "N/A"),
            # Duplicate annotation metadata
            "_is_master_record": profile.get("_is_master_record", False),
            "_duplicate_group_id": profile.get("_duplicate_group_id"),
            "_duplicate_count": profile.get("_duplicate_count", 1),
            "_associated_original_filenames": profile.get("_associated_original_filenames"),
            "_associated_ids": profile.get("_associated_ids")
        }
        documents.append(Document(page_content=content, metadata=metadata))

    if not documents:
        print("No documents to add to ChromaDB. Aborting ChromaDB creation.")
        return

    # Filter complex metadata before sending to ChromaDB
    # This will convert lists/dicts in metadata to string representations
    print("Filtering complex metadata for ChromaDB compatibility...")
    filtered_documents = filter_complex_metadata(documents)

    print(f"Creating ChromaDB with {len(filtered_documents)} documents...")
    vectorstore = Chroma.from_documents(
        documents=filtered_documents, # Use the filtered documents here
        embedding=embedding_function,
        persist_directory=chroma_dir
    )
    vectorstore.persist() # Save the collection to disk
    print(f"ChromaDB created and persisted to '{chroma_dir}' with {len(filtered_documents)} documents.")

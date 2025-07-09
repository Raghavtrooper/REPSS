import sys
import os
import pandas as pd
import json
import re
from typing import List, Set, Union # Import List, Set, Union for type hints
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time # Import time for time.ctime()

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary LangChain components
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Import modules from the current project structure
try:
    from shared.config import MAX_HISTORY_MESSAGES, OLLAMA_MODEL, GOLD_LAYER_PARQUET_FILE
except ImportError:
    print("Warning: shared/config.py not found or incomplete. Using default values.")
    MAX_HISTORY_MESSAGES = 10
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
    GOLD_LAYER_PARQUET_FILE = os.path.join(project_root, "data", "gold_layer.parquet")


from app.rag_chain import get_rag_chain
from app.vector_utils import get_vectorstore # This now gets Qdrant vectorstore


# --- Global Instances (initialized once) ---
vectorstore_instance = None
rag_chain_instance = None
embedding_function = None

# --- Constants for Retrieval and Reranking ---
INITIAL_RETRIEVAL_K = 20 # Number of documents to retrieve initially from vectorstore
FINAL_LLM_CONTEXT_K = 7 # Number of top documents to pass to the LLM after re-ranking/deduplication
PENALTY_FACTOR = 0.7 # Factor to penalize already shown documents (0.0 to 1.0, lower means more penalty)

# --- Initialize RAG chain components ---
def get_rag_chain_instance(_vectorstore):
    """Initializes the RAG chain if it hasn't been already."""
    global rag_chain_instance
    if rag_chain_instance is None:
        rag_chain_instance = get_rag_chain(_vectorstore)
    return rag_chain_instance

def get_embedding_function():
    global embedding_function
    if embedding_function is None:
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_function

# --- Document Re-ranking (to prioritize relevant and new documents) ---
def re_rank_documents(query: str, documents: List[Document], embedding_function,
                      previously_shown_doc_ids: Set[str], top_k: int = FINAL_LLM_CONTEXT_K,
                      penalty_factor: float = PENALTY_FACTOR) -> List[Document]:
    """
    Re-ranks documents based on similarity to query and penalizes already shown documents.
    """
    if not documents:
        return []

    # Get embeddings for the query and documents
    query_embedding = embedding_function.embed_query(query)
    doc_embeddings = embedding_function.embed_documents([doc.page_content for doc in documents])

    # Calculate cosine similarities
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    # Create a list of (similarity, Document, original_index)
    scored_documents = []
    for i, doc in enumerate(documents):
        # Apply penalty for previously shown documents
        penalty = penalty_factor if doc.metadata.get('_id') in previously_shown_doc_ids else 1.0
        adjusted_similarity = similarities[i] * penalty
        scored_documents.append((adjusted_similarity, doc, i))

    # Sort by adjusted similarity in descending order
    scored_documents.sort(key=lambda x: x[0], reverse=True)

    # Return top K documents
    return [doc for score, doc, original_index in scored_documents[:top_k]]

# --- Display Profile Details (moved from ui_components.py and adapted for terminal) ---
def display_profile_details(profile_metadata: dict, use_inner_expanders: bool = False):
    """
    Displays the details of an employee profile in a neatly formatted way for the terminal.
    The 'use_inner_expanders' argument is ignored as it's not applicable for terminal output.
    """
    print(f"\n--- Profile: {profile_metadata.get('name', 'N/A')} ---")
    print(f"  Email: {profile_metadata.get('email_id', 'N/A')}")
    print(f"  Phone: {profile_metadata.get('phone_number', 'N/A')}")
    print(f"  Location: {profile_metadata.get('location', 'N/A')}")
    print(f"  Has Photo: {'Yes' if profile_metadata.get('has_photo') else 'No'}")

    # Helper function to display content for terminal
    def render_section_terminal(title, content):
        if not content:
            return
        # Ensure content is a string for display; if it's a list (like skills), join it.
        display_content = ", ".join(content) if isinstance(content, list) else str(content)
        print(f"  {title}: {display_content}")

    render_section_terminal("Objective", profile_metadata.get('objective'))
    render_section_terminal("Skills", profile_metadata.get('skills'))
    render_section_terminal("Qualifications Summary", profile_metadata.get('qualifications_summary'))
    render_section_terminal("Experience Summary", profile_metadata.get('experience_summary'))

    # Display duplicate status if available
    if profile_metadata.get('_duplicate_count', 1) > 1:
        status = "Master Record" if profile_metadata.get('_is_master_record') else "Associated Resume"
        print(f"  Duplicate Status: This person has {profile_metadata['_duplicate_count']} associated resumes. "
              f"Group ID: {profile_metadata.get('_duplicate_group_id', 'N/A')[:8]} ({status}).")
        if profile_metadata.get('_associated_original_filenames'):
            print(f"  Associated Original Files: {', '.join(profile_metadata['_associated_original_filenames'])}")

    # Add original filename and download link if available
    if 'original_file_minio_url' in profile_metadata:
        minio_url = profile_metadata['original_file_minio_url']
        original_filename = profile_metadata.get('original_filename', 'Resume')
        print(f"  Download Original {original_filename}: {minio_url}")

    print("------------------------------------------")


# --- Main Query Processing ---
def process_employee_query(
    user_query: str,
    vectorstore_instance,
    rag_chain_instance,
    embedding_function,
    show_more_detector_data: bool,
    chat_history: List[dict],
    previously_shown_doc_ids: Set[str]
) -> dict:
    """
    Processes a user query by retrieving from vector store,
    re-ranking, and invoking the RAG chain.
    """
    try:
        newly_shown_doc_ids = set()

        # Rely solely on vector store retrieval
        retrieved_docs = vectorstore_instance.similarity_search(user_query, k=INITIAL_RETRIEVAL_K)
        
        # DEBUG: Inspect metadata from similarity_search
        if retrieved_docs:
            print(f"DEBUG: Metadata from similarity_search: {retrieved_docs[0].metadata}")

        final_context_for_llm = re_rank_documents(
            query=user_query,
            documents=retrieved_docs, # `retrieved_docs` is the source here
            embedding_function=embedding_function,
            previously_shown_doc_ids=previously_shown_doc_ids,
            top_k=FINAL_LLM_CONTEXT_K,
            penalty_factor=PENALTY_FACTOR
        )

        print(f"Final context for LLM (after vector retrieval and re-ranking): {len(final_context_for_llm)} documents.")
        print(f"DEBUG: Number of documents passed to LLM context: {len(final_context_for_llm)}")


        # Update newly shown doc IDs for the re-ranking penalty
        for doc in final_context_for_llm:
            doc_id = doc.metadata.get('_id')
            if doc_id:
                newly_shown_doc_ids.add(doc_id)

        # Invoke RAG chain with the determined context
        # It also now returns 'retrieved_profiles' with full metadata due to rag_chain.py changes.
        response = rag_chain_instance.invoke({
            "input": user_query,
            "chat_history": chat_history[-MAX_HISTORY_MESSAGES:], # Pass recent chat history
            "context": final_context_for_llm # Pass the curated context
        })

        llm_answer = response["answer"]
        # Directly use 'retrieved_profiles' from the RAG chain's output
        retrieved_profiles = response.get("retrieved_profiles", [])
        
        print(f"DEBUG: Retrieved profiles metadata before return: {retrieved_profiles}")

        return {
            "answer": llm_answer,
            "retrieved_profiles": retrieved_profiles,
            "updated_previously_shown_doc_ids": list(previously_shown_doc_ids.union(newly_shown_doc_ids))
        }

    except Exception as e:
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return {
            "answer": "I encountered an error while processing your request. Please check the logs.",
            "error": str(e),
            "retrieved_profiles": [],
            "updated_previously_shown_doc_ids": []
        }

# --- Terminal Application ---
def main_terminal():
    global vectorstore_instance, embedding_function, rag_chain_instance

    # 1. Initialize vector store and embedding function
    try:
        vectorstore_instance = get_vectorstore()
        embedding_function = get_embedding_function()
        rag_chain_instance = get_rag_chain_instance(vectorstore_instance)
    except Exception as e:
        print(f"Initialization failed: {e}")
        sys.exit(1)

    chat_history = []
    previously_shown_doc_ids = set() # To keep track of documents shown in the current session

    print("Welcome to the Employee Profile Search Assistant (Terminal Edition)!")
    print("Type your questions about employee skills, experience, or roles. Type 'exit' or 'quit' to end.")

    while True:
        user_query = input("\nEnter your question (or 'exit'/'quit' to end): ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting application. Goodbye!")
            break

        # For demo purposes, we're not using show_more_detector_data in terminal
        response_data = process_employee_query(
            user_query=user_query,
            vectorstore_instance=vectorstore_instance,
            rag_chain_instance=rag_chain_instance,
            embedding_function=embedding_function,
            show_more_detector_data=False, # Not applicable for terminal
            chat_history=chat_history,
            previously_shown_doc_ids=previously_shown_doc_ids
        )

        if "error" in response_data:
            print(f"\nAssistant: {response_data['answer']}")
            print(f"Error details: {response_data['error']}")
            print(f"Please ensure Ollama is running and the model '{OLLAMA_MODEL}' is pulled and running.")
            chat_history.append({"role": "assistant", "content": response_data["answer"]})
        else:
            llm_answer = response_data["answer"]
            retrieved_profiles = response_data["retrieved_profiles"]
            previously_shown_doc_ids.clear() # Clear previous for each turn to avoid excessive accumulation if not intended to persist
            previously_shown_doc_ids.update(response_data["updated_previously_shown_doc_ids"])

            print(f"\nAssistant: {llm_answer}")
            chat_history.append({"role": "assistant", "content": llm_answer})

            if retrieved_profiles:
                print("\n--- Retrieved Profiles ---")
                for profile_metadata in retrieved_profiles:
                    print(f"DEBUG: Data passed to display_profile_details: {profile_metadata}") # Added for debugging
                    # Call the locally defined display_profile_details
                    display_profile_details(profile_metadata, use_inner_expanders=False)
            else:
                print("\nNo relevant documents were retrieved for this query. This might mean your query is too specific or not semantically close to any profiles.")


if __name__ == "__main__":
    main_terminal()

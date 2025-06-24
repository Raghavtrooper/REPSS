import streamlit as st
import sys
import os
import pandas as pd
import json
import re
from pydantic import BaseModel, Field

# Add project root to sys.path (handled by run_app.py, but kept for standalone testing)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import SentenceTransformerEmbeddings # Import embeddings

# Import modules from the current project structure
# Expecting MAX_HISTORY_MESSAGES, OLLAMA_MODEL, GOLD_LAYER_PARQUET_FILE from shared.config
try:
    from shared.config import MAX_HISTORY_MESSAGES, OLLAMA_MODEL, GOLD_LAYER_PARQUET_FILE
except ImportError:
    # Fallback if shared.config is not available or missing variables (for standalone testing)
    print("Warning: shared/config.py not found or incomplete. Using default values.")
    MAX_HISTORY_MESSAGES = 10  # Default value if not from config
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2") # Default model if not from config
    GOLD_LAYER_PARQUET_FILE = os.path.join(project_root, "data", "gold_layer.parquet")


from app.rag_chain import get_rag_chain
from app.vector_utils import get_vectorstore
from app.ui_components import clear_chat_callback, display_profile_details # Import the new function

# --- Global Components ---
@st.cache_data
def load_gold_layer_data():
    """Loads the gold layer Parquet file into a Pandas DataFrame."""
    if not os.path.exists(GOLD_LAYER_PARQUET_FILE):
        st.error(f"Gold layer data not found at '{GOLD_LAYER_PARQUET_FILE}'.")
        st.markdown(
            "**Please run the `python run_etl.py` script first "
            "to create the gold layer Parquet file and ChromaDB before running this Streamlit application.**"
        )
        st.stop()
    
    print(f"Loading gold layer data from {GOLD_LAYER_PARQUET_FILE}...")
    df = pd.read_parquet(GOLD_LAYER_PARQUET_FILE)
    print(f"Loaded {len(df)} records from gold layer.")
    return df

@st.cache_resource(show_spinner=False)
def get_ollama_query_parser_llm():
    """Initializes a dedicated Ollama LLM for query parsing."""
    return Ollama(model=OLLAMA_MODEL, temperature=0.0) # Low temperature for consistent JSON output

# Pydantic model to define the expected structure of the parsed filter conditions
class FilterConditions(BaseModel):
    skills: list[str] = Field(default_factory=list, description="List of skills mentioned in the query (e.g., Python, AWS).")
    experience_years_min: int | None = Field(default=None, description="Minimum years of experience mentioned in the query.")
    education_level: str | None = Field(default=None, description="Education level (e.g., Bachelor's, Master's, PhD).")
    title: str | None = Field(default=None, description="Job title or role mentioned in the query.")
    department: str | None = Field(default=None, description="Department mentioned in the query.")
    email_id: str | None = Field(default=None, description="Specific email address mentioned in the query.")
    phone_number: str | None = Field(default=None, description="Specific phone number mentioned in the query.")

def parse_query_for_filters(query: str, llm: Ollama) -> FilterConditions:
    """
    Uses an LLM to extract structured filter conditions from a natural language query.
    Updated prompt for email_id and phone_number.
    """
    print(f"Parsing query for filters: '{query}')")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at extracting structured filter conditions from user queries about employee profiles.
        Extract relevant keywords for skills, minimum experience in years, education level, job title, department, email address, and phone number.
        If a condition is not explicitly mentioned or is ambiguous, leave it as null or an empty list.
        For phone numbers, extract digits and symbols like '+' only.
        Provide your response as a JSON object strictly following this schema:
        {
          "skills": ["skill1", "skill2"],
          "experience_years_min": 5,
          "education_level": "Master's",
          "title": "Software Engineer",
          "department": "Engineering",
          "email_id": "example@domain.com",
          "phone_number": "+1234567890"
        }
        Do not include any other text or explanation, just the JSON object.
        Normalize education levels to: Bachelor's, Master's, PhD.
        """),
        ("user", "{query}")
    ])

    parser_chain = prompt_template | llm 

    try:
        raw_json_str = parser_chain.invoke({"query": query})
        raw_json_str = raw_json_str.replace("```json", "").replace("```", "").strip()
        
        parsed_data = json.loads(raw_json_str)
        filters = FilterConditions(**parsed_data)
        print(f"Extracted filters: {filters.model_dump_json(indent=2)}")
        return filters
    except json.JSONDecodeError as e:
        print(f"Warning: LLM did not return valid JSON for filter extraction: {raw_json_str}. Error: {e}")
        return FilterConditions()
    except Exception as e:
        print(f"Error in parsing query for filters: {e}")
        return FilterConditions()

def apply_structured_filters_and_convert_to_docs(gold_df: pd.DataFrame, filters: FilterConditions) -> list[Document]:
    """
    Applies extracted filter conditions to the gold layer DataFrame and converts 
    matching records into LangChain Document objects.
    Now works with all records, including duplicates.
    """
    print("Applying structured filters to gold layer data...")
    filtered_df = gold_df.copy()

    if filters.skills:
        filter_skills_lower = [s.lower() for s in filters.skills]
        filtered_df = filtered_df[
            filtered_df['skills'].apply(lambda x: any(skill in x for skill in filter_skills_lower) if isinstance(x, list) else False)
        ]
        print(f"  Filtered by skills {filters.skills}: {len(filtered_df)} records remaining.")

    if filters.experience_years_min is not None:
        if 'experience' in filtered_df.columns:
            # Need to re-extract numerical experience as it's a string in gold layer
            filtered_df['experience_years_num'] = filtered_df['experience'].astype(str).str.extract(r'(\d+)\s*year').astype(float)
            filtered_df = filtered_df[
                (filtered_df['experience_years_num'].notna()) & 
                (filtered_df['experience_years_num'] >= filters.experience_years_min)
            ]
            filtered_df = filtered_df.drop(columns=['experience_years_num'])
            print(f"  Filtered by min experience {filters.experience_years_min} years: {len(filtered_df)} records remaining.")
    
    if filters.education_level:
        filter_edu_lower = filters.education_level.lower()
        if 'education' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['education'].astype(str).str.lower().str.contains(filter_edu_lower, na=False)
            ]
        print(f"  Filtered by education level '{filters.education_level}': {len(filtered_df)} records remaining.")

    if filters.title:
        filter_title_lower = filters.title.lower()
        if 'title' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['title'].astype(str).str.lower().str.contains(filter_title_lower, na=False)
            ]
        print(f"  Filtered by title '{filters.title}': {len(filtered_df)} records remaining.")

    if filters.department:
        filter_dept_lower = filters.department.lower()
        if 'department' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['department'].astype(str).str.lower().str.contains(filter_dept_lower, na=False)
            ]
        print(f"  Filtered by department '{filters.department}': {len(filtered_df)} records remaining.")
    
    # Filter by email_id
    if filters.email_id:
        filter_email_lower = filters.email_id.lower().strip()
        if 'email_id' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['email_id'].astype(str).str.lower().str.strip() == filter_email_lower
            ]
        print(f"  Filtered by email '{filters.email_id}': {len(filtered_df)} records remaining.")

    # Filter by phone_number
    if filters.phone_number:
        cleaned_filter_phone = re.sub(r'[^\d+]', '', filters.phone_number).strip()
        if 'phone_number' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['phone_number'].astype(str).str.strip() == cleaned_filter_phone
            ]
        print(f"  Filtered by phone number '{filters.phone_number}': {len(filtered_df)} records remaining.")


    # Convert filtered DataFrame rows to LangChain Document objects
    # Include new duplicate metadata in content and metadata
    documents = []
    for _, row in filtered_df.iterrows():
        # Include a note about duplicates in the page_content for LLM awareness
        duplicate_status = ""
        if row.get('_is_master_record') and row.get('_duplicate_count', 1) > 1:
            duplicate_status = f" (NOTE: This person has {row['_duplicate_count']} associated resumes. Group ID: {row['_duplicate_group_id']})"
        elif not row.get('_is_master_record') and row.get('_duplicate_group_id'):
            duplicate_status = f" (NOTE: This is an associated resume for a person with multiple entries. Group ID: {row['_duplicate_group_id']})"


        content = f"Name: {row.get('name', 'Unknown')}{duplicate_status}\n" \
                  f"Title: {row.get('title', 'Unknown')}\n" \
                  f"Department: {row.get('department', 'Unknown')}\n" \
                  f"Experience: {row.get('experience', 'Unknown')}\n" \
                  f"Skills: {', '.join(row['skills']) if isinstance(row.get('skills'), list) else 'Unknown'}\n" \
                  f"Education: {row.get('education', 'Unknown')}\n" \
                  f"Email: {row.get('email_id', 'Unknown')}\n" \
                  f"Phone: {row.get('phone_number', 'Unknown')}"
        
        # Add all original row data as metadata
        metadata = row.to_dict()
        metadata.pop('skills', None)
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    print(f"Structured filtering complete. Converted {len(documents)} matching profiles to Documents.")
    return documents

# --- Functions for embedding and re-ranking with penalization ---
@st.cache_resource(show_spinner=False)
def get_embedding_function():
    """Initializes and activates the SentenceTransformerEmbeddings function."""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def re_rank_documents(
    query: str,
    documents: list[Document],
    embedding_function: SentenceTransformerEmbeddings,
    previously_shown_doc_ids: set[str],
    top_k: int = 7,
    penalty_factor: float = 0.5 # Factor to reduce score for previously shown documents
) -> list[Document]:
    """
    Re-ranks a list of documents based on similarity to the query, applying a penalty
    to documents that have been previously shown.
    """
    if not documents:
        return []

    print(f"Re-ranking {len(documents)} documents. Previously shown: {len(previously_shown_doc_ids)}.")

    # 1. Embed the query
    query_embedding = embedding_function.embed_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # 2. Embed the documents and calculate initial similarity scores
    doc_embeddings = embedding_function.embed_documents([doc.page_content for doc in documents])
    doc_embeddings = np.array(doc_embeddings)

    # Calculate cosine similarity between query and each document
    # Ensure doc_embeddings is 2D
    if doc_embeddings.ndim == 1:
        doc_embeddings = doc_embeddings.reshape(1, -1)

    # Handle cases where query_embedding or doc_embeddings might be empty/invalid
    if query_embedding.shape[0] == 0 or doc_embeddings.shape[0] == 0:
        print("Warning: Empty embeddings for query or documents during re-ranking. Returning empty list.")
        return []

    similarity_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    # 3. Apply penalty
    penalized_scores = []
    for i, doc in enumerate(documents):
        score = similarity_scores[i]
        doc_id = doc.metadata.get('id')

        if doc_id and doc_id in previously_shown_doc_ids:
            # Apply penalty: reduce the score
            penalized_score = score * penalty_factor
            print(f"  Penalized document {doc_id} (score {score:.4f} -> {penalized_score:.4f})")
            penalized_scores.append(penalized_score)
        else:
            penalized_scores.append(score)

    # 4. Sort documents by penalized scores
    scored_documents = sorted(
        zip(penalized_scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    # 5. Return top_k documents
    final_ranked_docs = [doc for score, doc in scored_documents[:top_k]]
    print(f"Re-ranking complete. Returning {len(final_ranked_docs)} documents.")
    return final_ranked_docs

# --- New function to detect "show me more" intent ---
@st.cache_resource(show_spinner=False)
def get_show_more_detector(_embedding_function: SentenceTransformerEmbeddings): # Changed parameter name
    """
    Pre-computes embeddings for common 'show me more' phrases to detect intent.
    """
    show_more_phrases = [
        "show me more",
        "next",
        "more",
        "tell me more",
        "next one",
        "show more",
        "any others",
        "what else",
        "another one",
        "more profiles", # Added
        "next profiles", # Added
        "show more employees" # Added
    ]
    phrase_embeddings = _embedding_function.embed_documents(show_more_phrases) # Changed usage
    return {
        "phrases": show_more_phrases,
        "embeddings": np.array(phrase_embeddings),
        "embedding_function": _embedding_function # Store for use in is_show_more_query
    }

def is_show_more_query(query: str, detector_data: dict, similarity_threshold: float = 0.6) -> bool:
    """
    Checks if a query semantically indicates a "show me more" intent.
    """
    query_embedding = detector_data["embedding_function"].embed_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    if query_embedding.shape[0] == 0 or detector_data["embeddings"].shape[0] == 0:
        return False # Cannot compare if embeddings are empty

    similarities = cosine_similarity(query_embedding, detector_data["embeddings"])[0]
    max_similarity = np.max(similarities)

    print(f"Show me more intent detection: Max similarity for '{query}' is {max_similarity:.4f}")
    return max_similarity >= similarity_threshold


def main():
    st.set_page_config(layout="centered", page_title="Employee Profile AI Retriever")
    st.title("Employee Profile AI Retriever")
    st.markdown("Ask me anything about employee profiles in the organization! Try asking for 'Python developers with 5 years experience', 'someone from HR with a Master\'s degree', or 'the person with email jane.doe@example.com'.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm ready to help you find information about employees. What are you looking for?"})

    # Initialize session state for previously shown documents
    if "previously_shown_doc_ids" not in st.session_state:
        st.session_state.previously_shown_doc_ids = set() # Store unique IDs

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Load global components
    gold_df = load_gold_layer_data()
    vectorstore_instance = get_vectorstore()
    rag_chain_instance = get_rag_chain(vectorstore_instance)
    ollama_parser_llm = get_ollama_query_parser_llm()
    embedding_function = get_embedding_function() # Get the embedding function
    # Pass embedding function to the detector
    show_more_detector_data = get_show_more_detector(embedding_function)


    # Constants for retrieval and re-ranking
    INITIAL_RETRIEVAL_K = 20 # Fetch more documents than needed for diversity
    FINAL_LLM_CONTEXT_K = 7 # Number of documents to pass to the LLM after re-ranking/initial fetch
    PENALTY_FACTOR = 0.5 # Factor to reduce score for previously shown documents
    SHOW_MORE_SIMILARITY_THRESHOLD = 0.6 # Threshold to detect "show me more" intent


    if user_query := st.chat_input("Enter your question:"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Detect if the current query is a "show me more" intent
        current_query_is_show_more = is_show_more_query(user_query.lower(), show_more_detector_data, SHOW_MORE_SIMILARITY_THRESHOLD)
        print(f"Query '{user_query}' detected as 'show me more' intent: {current_query_is_show_more}")


        formatted_chat_history = []
        history_to_process = st.session_state.messages[:-1]
        limited_history = history_to_process[-MAX_HISTORY_MESSAGES:]

        for msg in limited_history:
            if msg["role"] == "user":
                formatted_chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_chat_history.append(AIMessage(content=msg["content"]))

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                try:
                    extracted_filters = parse_query_for_filters(user_query, ollama_parser_llm)

                    has_filters = any(
                        getattr(extracted_filters, field) not in (None, [], '')
                        for field in extracted_filters.model_fields.keys()
                    )

                    final_context_for_llm = []
                    
                    if has_filters and not gold_df.empty:
                        structured_filter_docs = apply_structured_filters_and_convert_to_docs(gold_df, extracted_filters)
                        if structured_filter_docs:
                            print(f"Query filters applied: {len(structured_filter_docs)} profiles matched by structured criteria.")

                            # For structured filtered results, we still combine with vector search 
                            # to get a broader pool for potential re-ranking, and for semantic richness.
                            retrieved_docs_from_vectorstore = vectorstore_instance.similarity_search(user_query, k=INITIAL_RETRIEVAL_K)
                            combined_context_docs = structured_filter_docs + retrieved_docs_from_vectorstore

                            # Deduplicate combined documents based on 'id'
                            seen_ids_for_dedupe = set() 
                            deduplicated_context = []
                            for doc in combined_context_docs:
                                doc_id = doc.metadata.get('id')
                                if doc_id and doc_id not in seen_ids_for_dedupe:
                                    deduplicated_context.append(doc)
                                    seen_ids_for_dedupe.add(doc_id)
                                elif not doc_id: 
                                    if doc.page_content not in [d.page_content for d in deduplicated_context]:
                                        deduplicated_context.append(doc)

                            if current_query_is_show_more:
                                # Apply re-ranking with penalization for 'show me more' intent
                                final_context_for_llm = re_rank_documents(
                                    query=user_query,
                                    documents=deduplicated_context,
                                    embedding_function=embedding_function,
                                    previously_shown_doc_ids=st.session_state.previously_shown_doc_ids,
                                    top_k=FINAL_LLM_CONTEXT_K,
                                    penalty_factor=PENALTY_FACTOR
                                )
                                # Update previously shown IDs ONLY when re-ranking is applied for 'show more'
                                for doc in final_context_for_llm:
                                    doc_id = doc.metadata.get('id')
                                    if doc_id:
                                        st.session_state.previously_shown_doc_ids.add(doc_id)
                            else:
                                # For initial query with filters, just take top relevant from deduplicated pool
                                # Sort by original similarity (higher is better)
                                initial_scores = []
                                # Only embed query once for this path
                                current_query_embedding_for_sort = embedding_function.embed_query(user_query)
                                current_query_embedding_for_sort = np.array(current_query_embedding_for_sort).reshape(1, -1)

                                for doc in deduplicated_context:
                                    doc_embedding = embedding_function.embed_documents([doc.page_content])[0]
                                    score = cosine_similarity(current_query_embedding_for_sort, np.array(doc_embedding).reshape(1,-1))[0][0]
                                    initial_scores.append((score, doc))
                                sorted_initial_docs = sorted(initial_scores, key=lambda x: x[0], reverse=True)
                                final_context_for_llm = [doc for score, doc in sorted_initial_docs[:FINAL_LLM_CONTEXT_K]]
                                
                                # Reset previously shown IDs for a fresh start on a new topic
                                st.session_state.previously_shown_doc_ids = set() 
                                # Add current top results to the set
                                for doc in final_context_for_llm:
                                    doc_id = doc.metadata.get('id')
                                    if doc_id:
                                        st.session_state.previously_shown_doc_ids.add(doc_id)

                            print(f"Final context for LLM (after structured filter, deduplication, and conditional retrieval): {len(final_context_for_llm)} documents.")

                        else: # Structured filters extracted but yielded no results (empty structured_filter_docs)
                            print("Structured filters extracted but yielded no results. Proceeding with vector search.")
                            initial_docs = vectorstore_instance.similarity_search(user_query, k=INITIAL_RETRIEVAL_K)
                            if current_query_is_show_more:
                                final_context_for_llm = re_rank_documents(
                                    query=user_query,
                                    documents=initial_docs,
                                    embedding_function=embedding_function,
                                    previously_shown_doc_ids=st.session_state.previously_shown_doc_ids,
                                    top_k=FINAL_LLM_CONTEXT_K,
                                    penalty_factor=PENALTY_FACTOR
                                )
                                for doc in final_context_for_llm:
                                    doc_id = doc.metadata.get('id')
                                    if doc_id:
                                        st.session_state.previously_shown_doc_ids.add(doc_id)
                            else:
                                final_context_for_llm = initial_docs[:FINAL_LLM_CONTEXT_K] # Just take top relevant
                                st.session_state.previously_shown_doc_ids = set() # Reset for new topic
                                for doc in final_context_for_llm:
                                    doc_id = doc.metadata.get('id')
                                    if doc_id:
                                        st.session_state.previously_shown_doc_ids.add(doc_id)

                    else: # No filters detected from query, proceed with pure vector search
                        print("No specific filters detected in query or gold layer empty. Performing pure vector search.")
                        initial_docs = vectorstore_instance.similarity_search(user_query, k=INITIAL_RETRIEVAL_K)

                        if current_query_is_show_more:
                            final_context_for_llm = re_rank_documents(
                                query=user_query,
                                documents=initial_docs,
                                embedding_function=embedding_function,
                                previously_shown_doc_ids=st.session_state.previously_shown_doc_ids,
                                top_k=FINAL_LLM_CONTEXT_K,
                                penalty_factor=PENALTY_FACTOR
                            )
                            for doc in final_context_for_llm:
                                doc_id = doc.metadata.get('id')
                                if doc_id:
                                    st.session_state.previously_shown_doc_ids.add(doc_id)
                        else:
                            final_context_for_llm = initial_docs[:FINAL_LLM_CONTEXT_K] # Just take top relevant
                            st.session_state.previously_shown_doc_ids = set() # Reset for new topic
                            for doc in final_context_for_llm:
                                doc_id = doc.metadata.get('id')
                                if doc_id:
                                    st.session_state.previously_shown_doc_ids.add(doc_id)

                    response = rag_chain_instance.invoke({
                        "input": user_query,
                        "chat_history": formatted_chat_history,
                        "context": final_context_for_llm
                    })
                    
                    llm_answer = response["answer"]
                    st.markdown(llm_answer)
                    st.session_state.messages.append({"role": "assistant", "content": llm_answer})

                    with st.expander("See Retrieved Profiles"): # Renamed for clarity
                        if response.get("context"):
                            for i, doc in enumerate(response["context"]):
                                # Call display_profile_details with use_inner_expanders=False
                                display_profile_details(doc.metadata, use_inner_expanders=False) 
                        else:
                            st.info("No relevant documents were retrieved for this query. This might mean your query is too specific or not semantically close to any profiles.")
                        
                        # Removed "Full RAG Chain Response Object:" and st.json(response) for cleaner output

                except Exception as e:
                    error_message = f"An unexpected error occurred during RAG chain invocation: {e}"
                    st.error(error_message)
                    st.warning(f"Please ensure Ollama is running and the model '{OLLAMA_MODEL}' is pulled and running.")
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

    with st.sidebar:
        st.header("Admin Tools")
        st.markdown("The vector database and gold layer are managed by the ETL process.")
        st.markdown(f"**Run `python run_etl.py` to build/rebuild the vector database and gold layer.**")
        
        # When clearing chat, also clear the previously shown documents
        if st.button("Clear Chat", on_click=clear_chat_callback):
             st.session_state.previously_shown_doc_ids = set() # Clear IDs when chat is cleared


if __name__ == "__main__":
    main()

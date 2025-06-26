import streamlit as st
import sys
import os
import pandas as pd
import json
import re
from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Add project root to sys.path (handled by run_app.py, but kept for standalone testing)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary LangChain components
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import SentenceTransformerEmbeddings # Import embeddings

# Import modules from the current project structure
try:
    from shared.config import MAX_HISTORY_MESSAGES, OLLAMA_MODEL, GOLD_LAYER_PARQUET_FILE
except ImportError:
    print("Warning: shared/config.py not found or incomplete. Using default values.")
    MAX_HISTORY_MESSAGES = 10
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
    GOLD_LAYER_PARQUET_FILE = os.path.join(project_root, "data", "gold_layer.parquet")


from app.rag_chain import get_rag_chain
from app.vector_utils import get_vectorstore
from app.ui_components import clear_chat_callback, display_profile_details

# --- Global Components Loading ---
def load_gold_layer_data():
    """Loads the gold layer Parquet file into a Pandas DataFrame."""
    if not os.path.exists(GOLD_LAYER_PARQUET_FILE):
        return None 
    
    print(f"Loading gold layer data from {GOLD_LAYER_PARQUET_FILE}...")
    df = pd.read_parquet(GOLD_LAYER_PARQUET_FILE)
    print(f"Loaded {len(df)} records from gold layer.")
    return df

def get_llm_and_embeddings():
    """Initializes and returns Ollama LLM for parsing and embedding function."""
    parser_llm = Ollama(model=OLLAMA_MODEL, temperature=0.0) # Low temperature for consistent JSON output
    embedding_func = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return parser_llm, embedding_func

def get_vectorstore_instance(): # Renamed to avoid clash with cached version
    """Initializes or loads the ChromaDB vector store."""
    return get_vectorstore()

def get_rag_chain_instance(_vectorstore): # Renamed
    """Initializes and returns the RAG (Retrieval Augmented Generation) chain."""
    return get_rag_chain(_vectorstore)

def get_show_more_detector(_embedding_function: SentenceTransformerEmbeddings): # Renamed
    """Pre-computes embeddings for common 'show me more' phrases to detect intent."""
    show_more_phrases = [
        "show me more", "next", "more", "tell me more", "next one",
        "show more", "any others", "what else", "another one",
        "more profiles", "next profiles", "show more employees"
    ]
    phrase_embeddings = _embedding_function.embed_documents(show_more_phrases)
    return {
        "phrases": show_more_phrases,
        "embeddings": np.array(phrase_embeddings),
        "embedding_function": _embedding_function
    }

# --- Pydantic model for parsed filter conditions ---
class FilterConditions(BaseModel):
    skills: list[str] = Field(default_factory=list, description="List of skills mentioned in the query (e.g., Python, AWS).")
    experience_years_min: int | None = Field(default=None, description="Minimum years of experience mentioned in the query (inferred from experience_summary).")
    education_level: str | None = Field(default=None, description="Education level (e.g., Bachelor's, Master's, PhD, inferred from qualifications_summary).")
    email_id: str | None = Field(default=None, description="Specific email address mentioned in the query.")
    phone_number: str | None = Field(default=None, description="Specific phone number mentioned in the query.")
    location: str | None = Field(default=None, description="Location mentioned in the query.") # New field
    has_photo: bool | None = Field(default=None, description="Boolean indicating if the profile should have a photo (e.g., 'with a photo', 'no photo').") # New field
    objective_keywords: list[str] = Field(default_factory=list, description="Keywords related to career objective.") # New field


# --- Helper functions for query processing ---

def parse_query_for_filters(query: str, llm: Ollama) -> FilterConditions:
    """
    Uses an LLM to extract structured filter conditions from a natural language query.
    Updated for new fields and to remove 'title'/'department'.
    """
    print(f"Parsing query for filters: '{query}')")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at extracting structured filter conditions from user queries about employee profiles.
        Extract relevant keywords for skills, minimum experience in years, education level, email address, phone number, location,
        whether a photo is required, and keywords for career objective.
        If a condition is not explicitly mentioned or is ambiguous, leave it as null or an empty list.
        For phone numbers, extract digits and symbols like '+' only.
        For 'has_photo', infer true if phrases like 'with a photo', 'has picture', 'image' are used. Infer false if 'no photo', 'without picture' are used. Otherwise, leave null.
        Provide your response as a JSON object strictly following this schema:
        {
          "skills": ["skill1", "skill2"],
          "experience_years_min": 5,
          "education_level": "Master's",
          "email_id": "example@domain.com",
          "phone_number": "+1234567890",
          "location": "City, State",
          "has_photo": true,
          "objective_keywords": ["leadership", "innovation"]
        }
        Do not include any other text or explanation, just the JSON object.
        Normalize education levels to: Bachelor's, Master's, PhD, Diploma, Certificate, Other.
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
    Updated for new fields (location, objective, has_photo) and renamed/removed fields.
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
        if 'experience_summary' in filtered_df.columns: # Use new field name
            filtered_df['experience_str'] = filtered_df['experience_summary'].astype(str)
            # This regex attempts to find "X year" or "X years"
            filtered_df['experience_years_num'] = filtered_df['experience_str'].str.extract(r'(\d+)\s*year').astype(float)
            filtered_df = filtered_df[
                (filtered_df['experience_years_num'].notna()) & 
                (filtered_df['experience_years_num'] >= filters.experience_years_min)
            ]
            filtered_df = filtered_df.drop(columns=['experience_years_num', 'experience_str'])
            print(f"  Filtered by min experience {filters.experience_years_min} years: {len(filtered_df)} records remaining.")
    
    if filters.education_level:
        filter_edu_lower = filters.education_level.lower()
        if 'qualifications_summary' in filtered_df.columns: # Use new field name
            filtered_df = filtered_df[
                filtered_df['qualifications_summary'].astype(str).str.lower().str.contains(filter_edu_lower, na=False)
            ]
        print(f"  Filtered by education level '{filters.education_level}': {len(filtered_df)} records remaining.")
    
    if filters.email_id:
        filter_email_lower = filters.email_id.lower().strip()
        if 'email_id' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['email_id'].astype(str).str.lower().str.strip() == filter_email_lower
            ]
        print(f"  Filtered by email '{filters.email_id}': {len(filtered_df)} records remaining.")

    if filters.phone_number:
        cleaned_filter_phone = re.sub(r'[^\d+]', '', filters.phone_number).strip()
        if 'phone_number' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['phone_number'].astype(str).str.strip() == cleaned_filter_phone
            ]
        print(f"  Filtered by phone number '{filters.phone_number}': {len(filtered_df)} records remaining.")

    if filters.location: # New filter for location
        filter_location_lower = filters.location.lower()
        if 'location' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['location'].astype(str).str.lower().str.contains(filter_location_lower, na=False)
            ]
        print(f"  Filtered by location '{filters.location}': {len(filtered_df)} records remaining.")
    
    if filters.has_photo is not None: # New filter for has_photo
        if 'has_photo' in filtered_df.columns:
            # Ensure the column is boolean before direct comparison
            filtered_df = filtered_df[filtered_df['has_photo'].astype(bool) == filters.has_photo]
        print(f"  Filtered by has_photo '{filters.has_photo}': {len(filtered_df)} records remaining.")

    if filters.objective_keywords: # New filter for objective keywords
        filter_objective_lower = [kw.lower() for kw in filters.objective_keywords]
        if 'objective' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['objective'].astype(str).str.lower().apply(
                    lambda x: any(kw in x for kw in filter_objective_lower)
                )
            ]
        print(f"  Filtered by objective keywords {filters.objective_keywords}: {len(filtered_df)} records remaining.")


    documents = []
    for _, row in filtered_df.iterrows():
        duplicate_status = ""
        if row.get('_is_master_record') and row.get('_duplicate_count', 1) > 1:
            duplicate_status = f" (NOTE: This person has {row['_duplicate_count']} associated resumes. Group ID: {row['_duplicate_group_id']})"
        elif not row.get('_is_master_record') and row.get('_duplicate_group_id'):
            duplicate_status = f" (NOTE: This is an associated resume for a person with multiple entries. Group ID: {row['_duplicate_group_id']})"

        # Constructing the content string for the document, including new fields
        content = f"Name: {row.get('name', 'Unknown')}{duplicate_status}\n" \
                  f"Email: {row.get('email_id', 'Unknown')}\n" \
                  f"Phone: {row.get('phone_number', 'Unknown')}\n" \
                  f"Location: {row.get('location', 'Unknown')}\n" \
                  f"Objective: {row.get('objective', 'Unknown')}\n" \
                  f"Skills: {', '.join(row['skills']) if isinstance(row.get('skills'), list) else 'Unknown'}\n" \
                  f"Qualifications Summary: {row.get('qualifications_summary', 'Unknown')}\n" \
                  f"Experience Summary: {row.get('experience_summary', 'Unknown')}\n" \
                  f"Has Photo: {row.get('has_photo', False)}"
        
        metadata = row.to_dict()
        metadata.pop('skills', None) # Remove skills from metadata to avoid redundancy if already in page_content

        documents.append(Document(page_content=content, metadata=metadata))
    
    print(f"Structured filtering complete. Converted {len(documents)} matching profiles to Documents.")
    return documents

def re_rank_documents(
    query: str,
    documents: list[Document],
    embedding_function: SentenceTransformerEmbeddings,
    previously_shown_doc_ids: set[str],
    top_k: int,
    penalty_factor: float
) -> list[Document]:
    """
    Re-ranks a list of documents based on similarity to the query, applying a penalty
    to documents that have been previously shown.
    """
    if not documents:
        return []

    print(f"Re-ranking {len(documents)} documents. Previously shown: {len(previously_shown_doc_ids)}.")

    query_embedding = embedding_function.embed_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    doc_embeddings = embedding_function.embed_documents([doc.page_content for doc in documents])
    doc_embeddings = np.array(doc_embeddings)

    if doc_embeddings.ndim == 1:
        doc_embeddings = doc_embeddings.reshape(1, -1)

    if query_embedding.shape[0] == 0 or doc_embeddings.shape[0] == 0:
        print("Warning: Empty embeddings for query or documents during re-ranking. Returning empty list.")
        return []

    similarity_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    penalized_scores = []
    for i, doc in enumerate(documents):
        score = similarity_scores[i]
        doc_id = doc.metadata.get('id')

        if doc_id and doc_id in previously_shown_doc_ids:
            penalized_score = score * penalty_factor
            print(f"  Penalized document {doc_id} (score {score:.4f} -> {penalized_score:.4f})")
            penalized_scores.append(penalized_score)
        else:
            penalized_scores.append(score)

    scored_documents = sorted(
        zip(penalized_scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    final_ranked_docs = [doc for score, doc in scored_documents[:top_k]]
    print(f"Re-ranking complete. Returning {len(final_ranked_docs)} documents.")
    return final_ranked_docs

def is_show_more_query(query: str, detector_data: dict, similarity_threshold: float) -> bool:
    """
    Checks if a query semantically indicates a "show me more" intent.
    """
    query_embedding = detector_data["embedding_function"].embed_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    if query_embedding.shape[0] == 0 or detector_data["embeddings"].shape[0] == 0:
        return False

    similarities = cosine_similarity(query_embedding, detector_data["embeddings"])[0]
    max_similarity = np.max(similarities)

    print(f"Show me more intent detection: Max similarity for '{query}' is {max_similarity:.4f}")
    return max_similarity >= similarity_threshold


# --- Core Query Processing Function (API friendly) ---
def process_employee_query(
    user_query: str,
    gold_df: pd.DataFrame,
    vectorstore_instance,
    rag_chain_instance,
    ollama_parser_llm,
    embedding_function,
    show_more_detector_data,
    chat_history: list, # Pass chat history as an argument
    previously_shown_doc_ids: set # Pass previously shown IDs as an argument
):
    """
    Core function to process a user query and retrieve employee profiles.
    This function is designed to be independent of Streamlit's session state.
    """
    
    # Constants for retrieval and re-ranking
    INITIAL_RETRIEVAL_K = 20
    FINAL_LLM_CONTEXT_K = 7
    PENALTY_FACTOR = 0.5
    SHOW_MORE_SIMILARITY_THRESHOLD = 0.6

    try:
        # 1. Detect "show me more" intent
        current_query_is_show_more = is_show_more_query(user_query.lower(), show_more_detector_data, SHOW_MORE_SIMILARITY_THRESHOLD)
        print(f"Query '{user_query}' detected as 'show me more' intent: {current_query_is_show_more}")

        # 2. Parse query for structured filters
        extracted_filters = parse_query_for_filters(user_query, ollama_parser_llm)
        has_filters = any(
            getattr(extracted_filters, field) not in (None, [], '')
            for field in extracted_filters.model_fields.keys()
        )

        final_context_for_llm = []
        newly_shown_doc_ids = set() # To track docs shown in this specific query

        # 3. Determine document retrieval strategy
        if has_filters and not gold_df.empty:
            structured_filter_docs = apply_structured_filters_and_convert_to_docs(gold_df, extracted_filters)
            if structured_filter_docs:
                print(f"Query filters applied: {len(structured_filter_docs)} profiles matched by structured criteria.")
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
                        # If a doc has no ID, assume unique for deduplication for now based on content
                        if doc.page_content not in [d.page_content for d in deduplicated_context]:
                            deduplicated_context.append(doc)

                if current_query_is_show_more:
                    final_context_for_llm = re_rank_documents(
                        query=user_query,
                        documents=deduplicated_context,
                        embedding_function=embedding_function,
                        previously_shown_doc_ids=previously_shown_doc_ids,
                        top_k=FINAL_LLM_CONTEXT_K,
                        penalty_factor=PENALTY_FACTOR
                    )
                    for doc in final_context_for_llm:
                        doc_id = doc.metadata.get('id')
                        if doc_id:
                            newly_shown_doc_ids.add(doc_id)
                else:
                    initial_scores = []
                    current_query_embedding_for_sort = embedding_function.embed_query(user_query)
                    current_query_embedding_for_sort = np.array(current_query_embedding_for_sort).reshape(1, -1)

                    for doc in deduplicated_context:
                        doc_embedding = embedding_function.embed_documents([doc.page_content])[0]
                        score = cosine_similarity(current_query_embedding_for_sort, np.array(doc_embedding).reshape(1,-1))[0][0]
                        initial_scores.append((score, doc))
                    sorted_initial_docs = sorted(initial_scores, key=lambda x: x[0], reverse=True)
                    final_context_for_llm = [doc for score, doc in sorted_initial_docs[:FINAL_LLM_CONTEXT_K]]
                    
                    # For a new query (not "show more"), clear previous IDs and add current ones
                    previously_shown_doc_ids.clear() 
                    for doc in final_context_for_llm:
                        doc_id = doc.metadata.get('id')
                        if doc_id:
                            newly_shown_doc_ids.add(doc_id)


                print(f"Final context for LLM (after structured filter, deduplication, and conditional retrieval): {len(final_context_for_llm)} documents.")

            else: # Structured filters extracted but yielded no results
                print("Structured filters extracted but yielded no results. Proceeding with vector search.")
                initial_docs = vectorstore_instance.similarity_search(user_query, k=INITIAL_RETRIEVAL_K)
                if current_query_is_show_more:
                    final_context_for_llm = re_rank_documents(
                        query=user_query,
                        documents=initial_docs,
                        embedding_function=embedding_function,
                        previously_shown_doc_ids=previously_shown_doc_ids,
                        top_k=FINAL_LLM_CONTEXT_K,
                        penalty_factor=PENALTY_FACTOR
                    )
                    for doc in final_context_for_llm:
                        doc_id = doc.metadata.get('id')
                        if doc_id:
                            newly_shown_doc_ids.add(doc_id)
                else:
                    final_context_for_llm = initial_docs[:FINAL_LLM_CONTEXT_K]
                    previously_shown_doc_ids.clear()
                    for doc in final_context_for_llm:
                        doc_id = doc.metadata.get('id')
                        if doc_id:
                            newly_shown_doc_ids.add(doc_id)

        else: # No filters detected from query, proceed with pure vector search
            print("No specific filters detected in query or gold layer empty. Performing pure vector search.")
            initial_docs = vectorstore_instance.similarity_search(user_query, k=INITIAL_RETRIEVAL_K)

            if current_query_is_show_more:
                final_context_for_llm = re_rank_documents(
                    query=user_query,
                    documents=initial_docs,
                    embedding_function=embedding_function,
                    previously_shown_doc_ids=previously_shown_doc_ids,
                    top_k=FINAL_LLM_CONTEXT_K,
                    penalty_factor=PENALTY_FACTOR
                )
                for doc in final_context_for_llm:
                    doc_id = doc.metadata.get('id')
                    if doc_id:
                        newly_shown_doc_ids.add(doc_id)
            else:
                final_context_for_llm = initial_docs[:FINAL_LLM_CONTEXT_K]
                previously_shown_doc_ids.clear()
                for doc in final_context_for_llm:
                    doc_id = doc.metadata.get('id')
                    if doc_id:
                        newly_shown_doc_ids.add(doc_id)

        # Update the master previously_shown_doc_ids set
        previously_shown_doc_ids.update(newly_shown_doc_ids)
        
        # Format chat history for LangChain (ensure it's not empty)
        formatted_history = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    formatted_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_history.append(AIMessage(content=msg["content"]))

        response = rag_chain_instance.invoke({
            "input": user_query,
            "chat_history": formatted_history,
            "context": final_context_for_llm
        })
        
        llm_answer = response["answer"]
        retrieved_profiles_metadata = [doc.metadata for doc in response.get("context", [])]

        return {
            "answer": llm_answer,
            "retrieved_profiles": retrieved_profiles_metadata,
            "updated_previously_shown_doc_ids": list(previously_shown_doc_ids) # Return for API to manage
        }

    except Exception as e:
        print(f"An unexpected error occurred during RAG chain invocation: {e}")
        return {
            "answer": f"An unexpected error occurred: {e}. Please ensure Ollama is running and the model '{OLLAMA_MODEL}' is pulled and running.",
            "retrieved_profiles": [],
            "error": str(e)
        }

# --- Streamlit Application Functions (remain mostly the same, now calling the core function) ---

def setup_streamlit_ui():
    """Sets up the basic Streamlit page configuration and title."""
    st.set_page_config(layout="centered", page_title="Employee Profile AI Retriever")
    st.title("Employee Profile AI Retriever")
    st.markdown("Ask me anything about employee profiles in the organization! Try asking for 'Python developers with 5 years experience', 'someone with a Master\'s degree in Computer Science', 'the person with email jane.doe@example.com', 'employees in New York', 'someone looking for leadership roles', or 'profiles with a photo'.")

def initialize_session_state():
    """Initializes Streamlit session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm ready to help you find information about employees. What are you looking for?"}]
    if "previously_shown_doc_ids" not in st.session_state:
        st.session_state.previously_shown_doc_ids = set()

def display_chat_history():
    """Displays all messages currently in the session state chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def get_formatted_chat_history_streamlit(): # Renamed to avoid clash
    """Prepares chat history in LangChain message format, limited by MAX_HISTORY_MESSAGES."""
    formatted_history = []
    # Exclude the last (current user) message for history context
    history_to_process = st.session_state.messages[:-1]
    limited_history = history_to_process[-MAX_HISTORY_MESSAGES:]

    for msg in limited_history:
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_history.append(AIMessage(content=msg["content"]))
    return formatted_history

def handle_query_processing_streamlit(user_query, gold_df, vectorstore_instance, rag_chain_instance,
                            ollama_parser_llm, embedding_function, show_more_detector_data):
    """
    Handles the entire process of taking a user query, processing it, and
    displaying the response and retrieved profiles. This function is for Streamlit UI.
    """
    # Add user query to messages and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating response..."):
            # Call the core processing function
            response_data = process_employee_query(
                user_query=user_query,
                gold_df=gold_df,
                vectorstore_instance=vectorstore_instance,
                rag_chain_instance=rag_chain_instance,
                ollama_parser_llm=ollama_parser_llm,
                embedding_function=embedding_function,
                show_more_detector_data=show_more_detector_data,
                chat_history=st.session_state.messages, # Pass current messages for history
                previously_shown_doc_ids=st.session_state.previously_shown_doc_ids
            )
            
            if "error" in response_data:
                st.error(response_data["error"])
                st.warning(f"Please ensure Ollama is running and the model '{OLLAMA_MODEL}' is pulled and running.")
                st.session_state.messages.append({"role": "assistant", "content": response_data["answer"]})
            else:
                llm_answer = response_data["answer"]
                retrieved_profiles = response_data["retrieved_profiles"]
                st.session_state.previously_shown_doc_ids.clear()
                st.session_state.previously_shown_doc_ids.update(response_data["updated_previously_shown_doc_ids"])

                st.markdown(llm_answer)
                st.session_state.messages.append({"role": "assistant", "content": llm_answer})

                with st.expander("See Retrieved Profiles"):
                    if retrieved_profiles:
                        for i, profile_metadata in enumerate(retrieved_profiles):
                            display_profile_details(profile_metadata, use_inner_expanders=False)
                    else:
                        st.info("No relevant documents were retrieved for this query. This might mean your query is too specific or not semantically close to any profiles.")


def display_admin_tools():
    """Displays the admin tools section in the sidebar."""
    with st.sidebar:
        st.header("Admin Tools")
        st.markdown("The vector database and gold layer are managed by the ETL process.")
        st.markdown(f"**Run `python run_etl.py` to build/rebuild the vector database and gold layer.**")
        
        if st.button("Clear Chat", on_click=clear_chat_callback):
             st.session_state.previously_shown_doc_ids = set() # Clear IDs when chat is cleared


def main():
    """Main function to run the Streamlit employee profile retriever application."""
    # 1. Setup Streamlit UI
    setup_streamlit_ui()

    # 2. Initialize Session State
    initialize_session_state()

    # 3. Load Global Components
    gold_df = st.cache_data(load_gold_layer_data)()
    parser_llm, embedding_function = st.cache_resource(get_llm_and_embeddings)()
    vectorstore_instance = st.cache_resource(get_vectorstore_instance)()
    rag_chain_instance = st.cache_resource(get_rag_chain_instance)(vectorstore_instance)
    show_more_detector_data = st.cache_resource(get_show_more_detector)(embedding_function)


    # 4. Display Current Chat History
    display_chat_history()

    # 5. Process User Input
    if user_query := st.chat_input("Enter your question:"):
        handle_query_processing_streamlit(user_query, gold_df, vectorstore_instance, rag_chain_instance,
                                parser_llm, embedding_function, show_more_detector_data)

    # 6. Display Admin Tools in Sidebar
    display_admin_tools()


if __name__ == "__main__":
    main()

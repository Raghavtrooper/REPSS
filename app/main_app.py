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
    # Removed 'skills' as they are handled by semantic search only
    experience_years_min: int | None = Field(default=None, description="Minimum years of experience mentioned in the query (inferred from experience_summary).")
    education_level: str | None = Field(default=None, description="Education level (e.g., Bachelor's, Master's, PhD, inferred from qualifications_summary).")
    email_id: str | None = Field(default=None, description="Specific email address mentioned in the query.")
    phone_number: str | None = Field(default=None, description="Specific phone number mentioned in the query.")
    location: str | None = Field(default=None, description="Location mentioned in the query.") # New field
    has_photo: bool | None = Field(default=None, description="Boolean indicating if the profile should have a photo (e.g., 'with a photo', 'no photo').") # New field
    objective_keywords: list[str] = Field(default_factory=list, description="Keywords related to career objective.") # New field
    title: str | None = Field(default=None, description="Job title mentioned in the query.")
    department: str | None = Field(default=None, description="Department mentioned in the query.")


# --- Helper functions for query processing ---

def parse_query_for_filters(query: str, primary_llm: Ollama) -> FilterConditions:
    """
    Extracts structured filters from the user query.
    Uses lightweight regex for common fields and falls back to LLM (phi2) if needed.
    """
    print(f"Parsing query for filters (hybrid mode): '{query}'")
    query_lower = query.lower()
    
    # --- Basic regex-based extraction ---
    # Removed skills regex - handled by semantic search
    # Title: look for common job titles
    title_match = re.search(r'(engineer|manager|scientist|developer|analyst|lead|hr)', query_lower)
    # Email: standard email pattern
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', query)
    # Phone number: allows for optional '+' and various separators
    phone_match = re.search(r'(\+?\d[\d\s\-\(\)]{6,})', query)
    # Experience: captures number before "years"
    years_match = re.search(r'(\d+)\s*(\+)?\s*years?', query_lower)
    # Education: captures common education levels
    edu_match = re.search(r"(bachelor|master|phd|mba|diploma|certificate)", query_lower)
    # Location: Simple regex for common city/state names, could be expanded
    location_match = re.search(r'\b(new york|london|paris|tokyo|seattle|san francisco|chicago|bangalore|delhi|mumbai)\b', query_lower)
    # Has Photo: detect intent for photo
    has_photo_true_match = re.search(r'\b(with a photo|has picture|image|photo)\b', query_lower)
    has_photo_false_match = re.search(r'\b(no photo|without picture)\b', query_lower)

    # Initialize filters with regex extracted values
    filters = FilterConditions(
        # skills=[] no longer extracted by regex
        experience_years_min=int(years_match.group(1)) if years_match else None,
        education_level=edu_match.group(1).title() + "'s" if edu_match else None, # Format as "Master's"
        title=title_match.group(1).title() if title_match else None,
        department=None,  # Regex can't reliably infer department, rely on LLM fallback
        email_id=email_match.group(0) if email_match else None,
        phone_number=re.sub(r"[^\d+]", "", phone_match.group(1)) if phone_match else None, # Clean phone number
        location=location_match.group(1).title() if location_match else None,
        has_photo=True if has_photo_true_match else (False if has_photo_false_match else None),
        objective_keywords=[] # Regex generally not suited for general keywords
    )

    # Check if any filters were extracted by regex.
    # We now exclude 'skills' from this check as it's no longer a structured filter.
    regex_extracted_any = any(
        (isinstance(v, list) and v) or # Checks for non-empty lists like objective_keywords
        (isinstance(v, str) and v) or # Checks for non-empty strings
        (isinstance(v, bool) and v is not None) or # Checks for True/False
        (isinstance(v, int) and v is not None) # Checks for non-None integers
        for k, v in filters.model_dump().items()
    )

    # If regex extraction yielded nothing, fall back to LLM extraction
    if not regex_extracted_any:
        print("Regex extraction empty. Falling back to LLM (phi2)...")
        try:
            # System prompt for LLM fallback. Note: It specifically tells LLM *not* to extract skills.
            system_prompt_llm = """You are an expert at extracting structured filter conditions from user queries about employee profiles.
            Extract relevant keywords for minimum experience in years, education level, email address, phone number, location,
            whether a photo is required, keywords for career objective, job title, and department.
            Do NOT extract skills as a structured filter; they will be handled by other means.
            If a condition is not explicitly mentioned or is ambiguous, leave it as null or an empty list.
            For phone numbers, extract digits and symbols like '+' only.
            For 'has_photo', infer true if phrases like 'with a photo', 'has picture', 'image' are used. Infer false if 'no photo', 'without picture' are used. Otherwise, leave null.
            Provide your response as a JSON object strictly following this schema:
            {
              "experience_years_min": 5,
              "education_level": "Master's",
              "email_id": "example@domain.com",
              "phone_number": "+1234567890",
              "location": "City, State",
              "has_photo": true,
              "objective_keywords": ["leadership", "innovation"],
              "title": "Software Engineer",
              "department": "Engineering"
            }
            Do not include any other text or explanation, just the JSON object.
            Normalize education levels to: Bachelor's, Master's, PhD, Diploma, Certificate, Other.
            """
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt_llm),
                ("user", "{query}")
            ])
            
            # Use phi2 as the lightweight fallback LLM
            phi2_llm = Ollama(model="phi2", temperature=0.0) 
            parser_chain = prompt_template | phi2_llm
            raw_json_str = parser_chain.invoke({"query": query})
            raw_json_str = raw_json_str.replace("```json", "").replace("```", "").strip()
            
            parsed_data = json.loads(raw_json_str)
            
            # Create a new FilterConditions object from LLM output
            llm_filters = FilterConditions(**parsed_data)
            
            # Merge LLM results into 'filters' (which is currently empty from regex)
            filters.experience_years_min = llm_filters.experience_years_min
            filters.education_level = llm_filters.education_level
            filters.email_id = llm_filters.email_id
            filters.phone_number = llm_filters.phone_number
            filters.location = llm_filters.location
            filters.has_photo = llm_filters.has_photo
            filters.objective_keywords = llm_filters.objective_keywords # LLM will provide these
            filters.title = llm_filters.title
            filters.department = llm_filters.department

        except Exception as e:
            print(f"LLM fallback failed: {e}. Returning empty filters.")
            # If LLM fallback fails, just return the empty FilterConditions object initialized earlier.
            return FilterConditions()

    print(f"Extracted filters (hybrid): {filters.model_dump_json(indent=2)}")
    return filters

def apply_structured_filters_and_convert_to_docs(gold_df: pd.DataFrame, filters: FilterConditions) -> list[Document]:
    """
    Applies structured filters to the gold layer DataFrame and converts 
    matching records into LangChain Document objects.
    """
    print("Applying structured filters to gold layer data...")
    filtered_df = gold_df.copy()

    if filters.experience_years_min is not None:
        if 'experience' in filtered_df.columns:
            # Safely convert experience to string before extracting
            filtered_df['experience_str'] = filtered_df['experience'].astype(str)
            # Extract numerical part for comparison, assuming format like "X year(s)"
            filtered_df['experience_years_num'] = filtered_df['experience_str'].str.extract(r'(\d+)\s*year').astype(float)
            filtered_df = filtered_df[
                (filtered_df['experience_years_num'].notna()) & 
                (filtered_df['experience_years_num'] >= filters.experience_years_min)
            ]
            # Drop the temporary columns
            filtered_df = filtered_df.drop(columns=['experience_years_num', 'experience_str'])
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

    if filters.email_id:
        filter_email_lower = filters.email_id.lower().strip()
        if 'email_id' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['email_id'].astype(str).str.lower().str.strip() == filter_email_lower
            ]
        print(f"  Filtered by email '{filters.email_id}': {len(filtered_df)} records remaining.")

    if filters.phone_number:
        # Clean phone number for exact match (remove non-digit/non-plus characters)
        cleaned_filter_phone = re.sub(r'[^\d+]', '', filters.phone_number).strip()
        if 'phone_number' in filtered_df.columns:
            # Ensure the DataFrame column is also cleaned for consistent comparison
            filtered_df = filtered_df[
                filtered_df['phone_number'].astype(str).apply(lambda x: re.sub(r'[^\d+]', '', x).strip()) == cleaned_filter_phone
            ]
        print(f"  Filtered by phone number '{filters.phone_number}': {len(filtered_df)} records remaining.")

    if filters.location:
        filter_location_lower = filters.location.lower()
        if 'location' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['location'].astype(str).str.lower().str.contains(filter_location_lower, na=False)
            ]
        print(f"  Filtered by location '{filters.location}': {len(filtered_df)} records remaining.")

    if filters.has_photo is not None:
        if 'has_photo' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['has_photo'] == filters.has_photo]
        print(f"  Filtered by has_photo '{filters.has_photo}': {len(filtered_df)} records remaining.")

    if filters.objective_keywords:
        # Assuming 'objective' column exists and contains text for keywords search
        if 'objective' in filtered_df.columns:
            # Create a regex pattern to match any of the keywords
            keywords_pattern = '|'.join(re.escape(k.lower()) for k in filters.objective_keywords)
            filtered_df = filtered_df[
                filtered_df['objective'].astype(str).str.lower().str.contains(keywords_pattern, na=False)
            ]
        print(f"  Filtered by objective keywords '{filters.objective_keywords}': {len(filtered_df)} records remaining.")

    # Removed skills filtering as per user request to handle skills via semantic search only.
    # if filters.skills: ...

    documents = []
    for _, row in filtered_df.iterrows():
        # Add duplicate status note if applicable
        duplicate_status = ""
        if row.get('_is_master_record') and row.get('_duplicate_count', 1) > 1:
            duplicate_status = f" (NOTE: This person has {row['_duplicate_count']} associated resumes. Group ID: {row['_duplicate_group_id']})"
        elif not row.get('_is_master_record') and row.get('_duplicate_group_id'):
            duplicate_status = f" (NOTE: This is an associated resume for a person with multiple entries. Group ID: {row['_duplicate_group_id']})"

        # Constructing the content for the Document object
        # Note: Skills are still included in the content for the LLM to process semantically.
        content = f"Name: {row.get('name', 'Unknown')}{duplicate_status}\n" \
                  f"Title: {row.get('title', 'Unknown')}\n" \
                  f"Department: {row.get('department', 'Unknown')}\n" \
                  f"Experience: {row.get('experience', 'Unknown')}\n" \
                  f"Skills: {', '.join(row['skills']) if isinstance(row.get('skills'), list) else 'Unknown'}\n" \
                  f"Education: {row.get('education', 'Unknown')}\n" \
                  f"Email: {row.get('email_id', 'Unknown')}\n" \
                  f"Phone: {row.get('phone_number', 'Unknown')}\n" \
                  f"Location: {row.get('location', 'Unknown')}\n" \
                  f"Has Photo: {'Yes' if row.get('has_photo') else 'No'}\n" \
                  f"Objective: {row.get('objective', 'Unknown')}"

        metadata = row.to_dict()
        # Skills are included in content, so optionally remove from metadata to avoid redundancy
        metadata.pop('skills', None) 

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
        # Pass the primary LLM configured from shared.config or defaults
        extracted_filters = parse_query_for_filters(user_query, ollama_parser_llm)
        # Check if any structured filters were actually extracted (excluding default empty list/None values)
        has_filters = False
        # Corrected Pydantic model_fields access for newer versions
        for field_name in FilterConditions.model_fields.keys(): # Access from class, not instance
            value = getattr(extracted_filters, field_name)
            # A value is considered a filter if it's not None, not an empty list, and not an empty string.
            if value is not None and \
               not (isinstance(value, list) and not value) and \
               not (isinstance(value, str) and not value):
                has_filters = True
                break
        print(f"Structured filters detected: {has_filters}")


        final_context_for_llm = []
        newly_shown_doc_ids = set() # To track docs shown in this specific query

        # 3. Determine document retrieval strategy
        if has_filters and not gold_df.empty:
            # Apply structured filters first
            structured_filter_docs = apply_structured_filters_and_convert_to_docs(gold_df, extracted_filters)
            if structured_filter_docs:
                print(f"Query filters applied: {len(structured_filter_docs)} profiles matched by structured criteria.")
                
                # Perform vector similarity search on the original user query for broader context
                retrieved_docs_from_vectorstore = vectorstore_instance.similarity_search(user_query, k=INITIAL_RETRIEVAL_K)
                
                # Combine structured filter results with vector store results
                combined_context_docs = structured_filter_docs + retrieved_docs_from_vectorstore

                # Deduplicate combined documents based on 'id' to avoid redundant profiles
                seen_ids_for_dedupe = set() 
                deduplicated_context = []
                for doc in combined_context_docs:
                    doc_id = doc.metadata.get('id')
                    if doc_id and doc_id not in seen_ids_for_dedupe:
                        deduplicated_context.append(doc)
                        seen_ids_for_dedupe.add(doc_id)
                    elif not doc_id: 
                        # If a doc has no ID (e.g., malformed data), treat it as unique based on content for deduplication in this step
                        if doc.page_content not in [d.page_content for d in deduplicated_context]:
                            deduplicated_context.append(doc)

                # Apply re-ranking or simple top-k based on "show me more" intent
                if current_query_is_show_more:
                    final_context_for_llm = re_rank_documents(
                        query=user_query,
                        documents=deduplicated_context,
                        embedding_function=embedding_function,
                        previously_shown_doc_ids=previously_shown_doc_ids,
                        top_k=FINAL_LLM_CONTEXT_K,
                        penalty_factor=PENALTY_FACTOR
                    )
                    # Add newly shown documents to the set
                    for doc in final_context_for_llm:
                        doc_id = doc.metadata.get('id')
                        if doc_id:
                            newly_shown_doc_ids.add(doc_id)
                else:
                    # For a new query, clear previously shown IDs and populate with current results
                    # Sort by initial similarity before taking top_k
                    initial_scores = []
                    current_query_embedding_for_sort = embedding_function.embed_query(user_query)
                    current_query_embedding_for_sort = np.array(current_query_embedding_for_sort).reshape(1, -1)

                    for doc in deduplicated_context:
                        doc_embedding = embedding_function.embed_documents([doc.page_content])[0]
                        score = cosine_similarity(current_query_embedding_for_sort, np.array(doc_embedding).reshape(1,-1))[0][0]
                        initial_scores.append((score, doc))
                    sorted_initial_docs = sorted(initial_scores, key=lambda x: x[0], reverse=True)
                    final_context_for_llm = [doc for score, doc in sorted_initial_docs[:FINAL_LLM_CONTEXT_K]]
                    
                    previously_shown_doc_ids.clear() 
                    for doc in final_context_for_llm:
                        doc_id = doc.metadata.get('id')
                        if doc_id:
                            newly_shown_doc_ids.add(doc_id)

                print(f"Final context for LLM (after structured filter, deduplication, and conditional retrieval): {len(final_context_for_llm)} documents.")

            else: # Structured filters extracted but yielded no results from gold_df
                print("Structured filters extracted but yielded no results from gold layer. Proceeding with pure vector search.")
                # Fallback to pure vector search if structured filters yield nothing
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

        else: # No filters detected from query, or gold layer is empty. Proceed with pure vector search.
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

        # Update the master previously_shown_doc_ids set with all new IDs from this query's results
        previously_shown_doc_ids.update(newly_shown_doc_ids)
        
        # Format chat history for LangChain (ensure it's not empty)
        formatted_history = []
        if chat_history:
            # The chat_history passed here from Streamlit includes the *current* user query.
            # LangChain's ChatPromptTemplate expects chat_history to be previous turns.
            # So, we should exclude the very last message (the current user query)
            # when constructing the `chat_history` for the prompt.
            for msg in chat_history[:-1]: # Exclude the current user's message
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
    # Updated example prompt to reflect skill filter and other new capabilities
    st.markdown("Ask me anything about employee profiles in the organization! Try asking for 'Python developers with 5 years experience', 'someone with a Master\'s degree in Computer Science', 'the person with email jane.doe@example.com', 'employees in New York', 'someone looking for leadership roles', 'profiles with a photo', or 'employees in the Engineering department'.")

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
                st.warning(f"Please ensure Ollama is running and the model '{OLLAMA_MODEL}' is pulled and running, and also 'phi2' for fallback.")
                st.session_state.messages.append({"role": "assistant", "content": response_data["answer"]})
            else:
                llm_answer = response_data["answer"]
                retrieved_profiles = response_data["retrieved_profiles"]
                # Clear and update the session state's previously_shown_doc_ids
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

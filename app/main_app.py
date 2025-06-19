import streamlit as st
import sys
import os
import pandas as pd # Import pandas for data filtering
import json # For parsing LLM output
from pydantic import BaseModel, Field # For structured output parsing

# --- Add project root to sys.path ---
# Get the absolute path of the directory where main_app.py resides (e.g., E:/Projects/employee2/app)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (e.g., E:/Projects/employee2) which is the project root
project_root = os.path.dirname(current_dir)
# Add the project root to sys.path so Python can find 'shared' and 'etl' modules
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End sys.path modification ---

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document # To create LangChain Document objects from filtered data
from langchain_community.llms import Ollama # For query parsing LLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # <--- ADDED THESE IMPORTS

# Import modules from the current project structure
from shared.config import MAX_HISTORY_MESSAGES, OLLAMA_MODEL, GOLD_LAYER_PARQUET_FILE
from app.rag_chain import get_rag_chain
from app.vector_utils import get_vectorstore
from app.ui_components import clear_chat_callback

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
    # Add other fields as needed for your gold layer structure

def parse_query_for_filters(query: str, llm: Ollama) -> FilterConditions:
    """
    Uses an LLM to extract structured filter conditions from a natural language query.
    """
    print(f"Parsing query for filters: '{query}'")
    # Define the prompt for the LLM to extract filter conditions
    # This prompt asks the LLM to output a JSON object adhering to the FilterConditions schema
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at extracting structured filter conditions from user queries about employee profiles.
        Extract relevant keywords for skills, minimum experience in years, education level, job title, and department.
        If a condition is not explicitly mentioned or is ambiguous, leave it as null or an empty list.
        Provide your response as a JSON object strictly following this schema:
        {
          "skills": ["skill1", "skill2"],
          "experience_years_min": 5,
          "education_level": "Master's",
          "title": "Software Engineer",
          "department": "Engineering"
        }
        Do not include any other text or explanation, just the JSON object.
        Normalize education levels to: Bachelor's, Master's, PhD.
        """),
        ("user", "{query}")
    ])

    parser_chain = prompt_template | llm 

    try:
        raw_json_str = parser_chain.invoke({"query": query})
        # Clean up the string to ensure it's valid JSON (remove any ```json or backticks)
        raw_json_str = raw_json_str.replace("```json", "").replace("```", "").strip()
        
        parsed_data = json.loads(raw_json_str)
        # Validate the parsed data against the Pydantic model
        filters = FilterConditions(**parsed_data)
        print(f"Extracted filters: {filters.model_dump_json(indent=2)}")
        return filters
    except json.JSONDecodeError as e:
        print(f"Warning: LLM did not return valid JSON for filter extraction: {raw_json_str}. Error: {e}")
        return FilterConditions() # Return empty filters on error
    except Exception as e:
        print(f"Error in parsing query for filters: {e}")
        return FilterConditions()

def apply_structured_filters_and_convert_to_docs(gold_df: pd.DataFrame, filters: FilterConditions) -> list[Document]:
    """
    Applies extracted filter conditions to the gold layer DataFrame and converts 
    matching records into LangChain Document objects.
    """
    print("Applying structured filters to gold layer data...")
    filtered_df = gold_df.copy() # Work on a copy

    if filters.skills:
        # Filter where ANY of the required skills are in the profile's skills list
        # Skills are normalized to lowercase in gold_layer_transformer, so convert filter skills to lowercase too
        filter_skills_lower = [s.lower() for s in filters.skills]
        filtered_df = filtered_df[
            filtered_df['skills'].apply(lambda x: any(skill in x for skill in filter_skills_lower) if isinstance(x, list) else False)
        ]
        print(f"  Filtered by skills {filters.skills}: {len(filtered_df)} records remaining.")

    if filters.experience_years_min is not None:
        # Assuming 'experience' column in gold_df contains a string like "5 years", extract numerical value
        if 'experience' in filtered_df.columns:
            filtered_df['experience_years_num'] = filtered_df['experience'].astype(str).str.extract(r'(\d+)\s*year').astype(float)
            filtered_df = filtered_df[
                (filtered_df['experience_years_num'].notna()) & 
                (filtered_df['experience_years_num'] >= filters.experience_years_min)
            ]
            filtered_df = filtered_df.drop(columns=['experience_years_num']) # Clean up temp column
            print(f"  Filtered by min experience {filters.experience_years_min} years: {len(filtered_df)} records remaining.")
    
    if filters.education_level:
        # Normalize filter education level for comparison
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
    
    # Convert filtered DataFrame rows to LangChain Document objects
    # Combine relevant fields into page_content for LLM context
    documents = []
    for _, row in filtered_df.iterrows():
        # Create a concise summary for the document content
        content = f"Name: {row.get('name', 'Unknown')}\n" \
                  f"Title: {row.get('title', 'Unknown')}\n" \
                  f"Department: {row.get('department', 'Unknown')}\n" \
                  f"Experience: {row.get('experience', 'Unknown')}\n" \
                  f"Skills: {', '.join(row['skills']) if isinstance(row.get('skills'), list) else 'Unknown'}\n" \
                  f"Education: {row.get('education', 'Unknown')}\n" \
                  f"Contact: {row.get('contact_info', 'Unknown')}"
        
        # Add all original row data as metadata
        metadata = row.to_dict()
        # Remove any large/unnecessary columns from metadata if they exist
        metadata.pop('skills', None) # Skills are in content, can be omitted from metadata if too long
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    print(f"Structured filtering complete. Converted {len(documents)} matching profiles to Documents.")
    return documents


def main():
    st.set_page_config(layout="centered", page_title="Employee Profile AI Retriever")
    st.title("👨‍💼 Employee Profile AI Retriever")
    st.markdown("Ask me anything about employee profiles in the organization! Try asking for 'Python developers with 5 years experience' or 'someone from HR with a Master\'s degree'.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm ready to help you find information about employees. What are you looking for?"})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Load global components
    gold_df = load_gold_layer_data()
    vectorstore_instance = get_vectorstore() 
    rag_chain_instance = get_rag_chain(vectorstore_instance)
    ollama_parser_llm = get_ollama_query_parser_llm()


    if user_query := st.chat_input("Enter your question:"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        formatted_chat_history = []
        history_to_process = st.session_state.messages[:-1] # Exclude the current user query
        limited_history = history_to_process[-MAX_HISTORY_MESSAGES:] # Limit the history length

        for msg in limited_history:
            if msg["role"] == "user":
                formatted_chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_chat_history.append(AIMessage(content=msg["content"]))

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                try:
                    # --- Query Refinement (Structured Filtering) ---
                    extracted_filters = parse_query_for_filters(user_query, ollama_parser_llm)
                    
                    # Check if any filters were successfully extracted
                    has_filters = any(
                        getattr(extracted_filters, field) not in (None, [], '') 
                        for field in extracted_filters.model_fields.keys()
                    )

                    structured_filter_docs = []
                    if has_filters and not gold_df.empty:
                        structured_filter_docs = apply_structured_filters_and_convert_to_docs(gold_df, extracted_filters)
                        if structured_filter_docs:
                            print(f"Query filters applied: {len(structured_filter_docs)} profiles matched by structured criteria.")
                            
                            # Perform vector search regardless, then combine
                            retrieved_docs_from_vectorstore = vectorstore_instance.similarity_search(user_query)
                            
                            combined_context_docs = structured_filter_docs + retrieved_docs_from_vectorstore
                            
                            # Deduplicate documents based on content or a unique ID if available in metadata
                            seen_ids = set()
                            deduplicated_context = []
                            for doc in combined_context_docs:
                                doc_id = doc.metadata.get('id')
                                if doc_id and doc_id not in seen_ids:
                                    deduplicated_context.append(doc)
                                    seen_ids.add(doc_id)
                                elif not doc_id: # If no ID, add it if its content is somewhat unique
                                    # Fallback for docs without an ID, might need a more robust check
                                    if doc.page_content not in [d.page_content for d in deduplicated_context]:
                                        deduplicated_context.append(doc)
                            
                            final_context_for_llm = deduplicated_context
                            print(f"Final context for LLM (after structured filter and deduplication): {len(final_context_for_llm)} documents.")

                        else:
                            print("Structured filters extracted but yielded no results. Falling back to pure vector search.")
                            final_context_for_llm = vectorstore_instance.similarity_search(user_query)
                    else:
                        # No explicit filters or gold_df is empty, proceed with pure vector search
                        print("No specific filters detected in query or gold layer empty. Performing pure vector search.")
                        final_context_for_llm = vectorstore_instance.similarity_search(user_query)
                    # --- End Query Refinement ---

                    response = rag_chain_instance.invoke({
                        "input": user_query,
                        "chat_history": formatted_chat_history,
                        "context": final_context_for_llm # Pass the prepared context
                    })
                    
                    llm_answer = response["answer"]
                    st.markdown(llm_answer)
                    st.session_state.messages.append({"role": "assistant", "content": llm_answer})

                    with st.expander("Advanced Debugging: Retrieved Context"):
                        st.subheader("Raw Context Sent to LLM:")
                        if response.get("context"): # Use .get to safely access context from response
                            for i, doc in enumerate(response["context"]):
                                st.markdown(f"**Document {i+1} (ID: {doc.metadata.get('id', 'N/A')}, Name: {doc.metadata.get('name', 'N/A')}, Title: {doc.metadata.get('title', 'N/A')}):**")
                                st.text(doc.page_content)
                                st.markdown(f"**Metadata:**")
                                st.json(doc.metadata)
                        else:
                            st.info("No relevant documents were retrieved for this query. This might mean your query is too specific or not semantically close to any profiles.")
                        
                        st.subheader("Full RAG Chain Response Object:")
                        st.json(response)
                except Exception as e:
                    error_message = f"An unexpected error occurred during RAG chain invocation: {e}"
                    st.error(error_message)
                    st.warning(f"Please ensure Ollama is running and the model '{OLLAMA_MODEL}' is pulled and running.")
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

    with st.sidebar:
        st.header("Admin Tools")
        st.markdown("The vector database is managed by the ETL process. If you need to update or reset it, run the ETL script.")
        st.markdown(f"**Run `python run_etl.py` to build/rebuild the vector database and gold layer.**")
        
        st.button("Clear Chat", on_click=clear_chat_callback)


if __name__ == "__main__":
    main()

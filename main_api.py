from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama

# Project imports
from rag_chain import get_rag_chain
from vector_utils import get_vectorstore
from main_app import parse_query_for_filters, apply_structured_filters_and_convert_to_docs, re_rank_documents, is_show_more_query
from main_app import FilterConditions

# Load config fallback
try:
    from shared.config import GOLD_LAYER_PARQUET_FILE, OLLAMA_MODEL
except ImportError:
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
    GOLD_LAYER_PARQUET_FILE = os.path.join("data", "gold_layer.parquet")

# --- FastAPI App ---
app = FastAPI(title="Employee RAG Search API")

# --- Pydantic Models ---

class QueryRequest(BaseModel):
    query: str
    previously_shown_doc_ids: Optional[List[str]] = []

class ProfileMetadata(BaseModel):
    name: str
    title: str
    department: str
    email_id: Optional[str]
    phone_number: Optional[str]
    skills: Optional[List[str]] = []
    experience: Optional[str] = None
    education: Optional[str] = None
    duplicate_count: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    profiles: List[ProfileMetadata]


# --- Load Components at Startup ---

vectorstore = get_vectorstore()
rag_chain = get_rag_chain(vectorstore)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Ollama(model=OLLAMA_MODEL, temperature=0.0)

if not os.path.exists(GOLD_LAYER_PARQUET_FILE):
    raise RuntimeError(f"Gold layer not found at {GOLD_LAYER_PARQUET_FILE}")

gold_df = pd.read_parquet(GOLD_LAYER_PARQUET_FILE)

# --- Constants ---
INITIAL_RETRIEVAL_K = 20
FINAL_LLM_CONTEXT_K = 7
PENALTY_FACTOR = 0.5
SHOW_MORE_SIMILARITY_THRESHOLD = 0.6

# --- API Endpoint ---

@app.post("/query", response_model=QueryResponse)
def query_profiles(payload: QueryRequest):
    query = payload.query
    previously_shown_doc_ids = set(payload.previously_shown_doc_ids or [])

    try:
        extracted_filters = parse_query_for_filters(query, llm)

        has_filters = any(
            getattr(extracted_filters, field) not in (None, [], '')
            for field in extracted_filters.model_fields.keys()
        )

        final_context_for_llm = []

        if has_filters and not gold_df.empty:
            structured_docs = apply_structured_filters_and_convert_to_docs(gold_df, extracted_filters)
            vector_docs = vectorstore.similarity_search(query, k=INITIAL_RETRIEVAL_K)
            combined_docs = structured_docs + vector_docs

            seen_ids = set()
            deduped_docs = []
            for doc in combined_docs:
                doc_id = doc.metadata.get("id")
                if doc_id and doc_id not in seen_ids:
                    deduped_docs.append(doc)
                    seen_ids.add(doc_id)
                elif not doc_id and doc.page_content not in [d.page_content for d in deduped_docs]:
                    deduped_docs.append(doc)

            final_context_for_llm = re_rank_documents(
                query=query,
                documents=deduped_docs,
                embedding_function=embedding_function,
                previously_shown_doc_ids=previously_shown_doc_ids,
                top_k=FINAL_LLM_CONTEXT_K,
                penalty_factor=PENALTY_FACTOR
            )

        else:
            vector_docs = vectorstore.similarity_search(query, k=INITIAL_RETRIEVAL_K)
            final_context_for_llm = re_rank_documents(
                query=query,
                documents=vector_docs,
                embedding_function=embedding_function,
                previously_shown_doc_ids=previously_shown_doc_ids,
                top_k=FINAL_LLM_CONTEXT_K,
                penalty_factor=PENALTY_FACTOR
            )

        response = rag_chain.invoke({
            "input": query,
            "chat_history": [],
            "context": final_context_for_llm
        })

        profiles = []
        for doc in final_context_for_llm:
            meta = doc.metadata
            profiles.append(ProfileMetadata(
                name=meta.get("name", "Unknown"),
                title=meta.get("title", "Unknown"),
                department=meta.get("department", "Unknown"),
                email_id=meta.get("email_id"),
                phone_number=meta.get("phone_number"),
                skills=meta.get("skills"),
                experience=meta.get("experience"),
                education=meta.get("education"),
                duplicate_count=meta.get("_duplicate_count")
            ))

        return QueryResponse(answer=response["answer"], profiles=profiles)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Run Locally ---
if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)

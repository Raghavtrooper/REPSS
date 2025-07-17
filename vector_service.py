"""
Vector Service for MVP
Handles embeddings, dense search, and re-ranking using Qdrant and Cross-Encoder.
"""
import logging
import httpx
import json
import uuid
from typing import List, Dict, Optional
import asyncio
import os
import numpy as np # Import numpy

# Conditional imports for sentence-transformers and CrossEncoder
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    logging.warning("sentence-transformers or CrossEncoder not available. Using mock models.")
    SentenceTransformer = None
    CrossEncoder = None

# Configure logging level to DEBUG for more detailed output during debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set to DEBUG for more verbose output

class VectorService:
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL", "http://157.180.44.51:6333")
        self.collection_name = "employee_profiles1"
        self.embedding_model = None  # For dense embeddings
        self.reranker_model = None   # For cross-encoder re-ranking

    async def initialize_collections(self):
        """Initialize Qdrant collections. Checks if collection exists."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.qdrant_url}/collections/{self.collection_name}")
                if response.status_code == 200:
                    logger.info(f"Qdrant collection {self.collection_name} already exists")
                else:
                    logger.info(f"Note: {self.collection_name} collection not found on the server. It should be pre-created.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collections: {e}")

    async def get_embedding_model(self):
        """Get or initialize the dense embedding model."""
        if self.embedding_model is None:
            if SentenceTransformer:
                try:
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Loaded dense embedding model: all-MiniLM-L6-v2")
                except Exception as e:
                    logger.warning(f"Failed to load SentenceTransformer: {e}. Using mock embeddings.")
                    self.embedding_model = "mock"
            else:
                self.embedding_model = "mock"
        return self.embedding_model

    async def get_reranker_model(self):
        """Get or initialize the Cross-Encoder re-ranking model."""
        if self.reranker_model is None:
            if CrossEncoder:
                try:
                    self.reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                    logger.info("Loaded Cross-Encoder reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
                except Exception as e:
                    logger.warning(f"Failed to load CrossEncoder: {e}. Re-ranking will be skipped.")
                    self.reranker_model = "mock"
            else:
                self.reranker_model = "mock"
        return self.reranker_model

    async def create_dense_embedding(self, text: str) -> List[float]:
        """Create dense embedding for text."""
        try:
            model = await self.get_embedding_model()
            if model == "mock":
                # Deterministic mock embedding for testing (384 dimensions)
                import hashlib
                import struct
                text_hash = hashlib.md5(text.encode()).digest()
                embedding = []
                for i in range(0, len(text_hash), 4):
                    chunk = text_hash[i:i+4]
                    if len(chunk) == 4:
                        val = struct.unpack('f', chunk)[0]
                        embedding.append(val)
                while len(embedding) < 384:
                    embedding.extend(embedding[:min(len(embedding), 384 - len(embedding))])
                return embedding[:384]
            else:
                embedding = model.encode(text)
                return embedding.tolist()
        except Exception as e:
            logger.error(f"Dense embedding creation failed: {e}")
            return [0.0] * 384 # Return zero vector as fallback

    async def add_document(self, text: str, metadata: Dict) -> str:
        """Add document to vector database with dense embedding."""
        try:
            # Create dense embedding
            embedding = await self.create_dense_embedding(text)
            
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            
            # Prepare point for Qdrant
            point = {
                "id": doc_id,
                "vector": embedding, # Store dense vector directly (unnamed vector)
                "payload": {
                    **metadata,
                    "text_content": text[:1000],  # Store first 1000 chars of original text
                    "text_length": len(text)
                }
            }
            
            # Insert into Qdrant
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.qdrant_url}/collections/{self.collection_name}/points",
                    json={
                        "points": [point]
                    }
                )
                if response.status_code == 200:
                    logger.info(f"Added document to vector DB: {doc_id}")
                    return doc_id
                else:
                    logger.error(f"Failed to add document to vector DB: {response.text}")
                    raise Exception(f"Vector DB insertion failed: {response.text}")
        except Exception as e:
            logger.error(f"Document addition failed: {e}")
            raise Exception(f"Vector DB error: {str(e)}")

    async def search_resumes(self, query_text: str, job_category: Optional[str] = None, 
                           limit: int = 10, similarity_threshold: float = 0.7, # Threshold for initial retrieval
                           initial_retrieval_limit: int = 40, # MODIFIED: Default changed to 40
                           enhance_query: bool = True) -> List[Dict]:
        """
        Search for similar resumes in employee_profiles collection using dense search and re-ranking.
        `limit` refers to the final number of results after re-ranking.
        `initial_retrieval_limit` is the number of candidates fetched from Qdrant for re-ranking.
        """
        try:
            logger.info(f"Search request: query='{query_text}', category='{job_category}', final_limit={limit}, initial_limit={initial_retrieval_limit}, threshold={similarity_threshold}")
            
            # Enhance query if enabled
            final_query = query_text
            if enhance_query:
                try:
                    # Assuming query_enhancer is in the same package or accessible
                    # Note: This import might need adjustment based on your project structure
                    from .query_enhancer import enhance_search_query
                    context = {"job_category": job_category} if job_category else None
                    final_query = await enhance_search_query(query_text, context)
                    if final_query != query_text:
                        logger.info(f"Query enhanced: '{query_text}' -> '{final_query}'")
                except ImportError:
                    logger.info("Query enhancer not available, using original query.")
                except Exception as e:
                    logger.warning(f"Query enhancement failed: {e}, using original query.")
            
            # Create dense query embedding
            dense_query_embedding = await self.create_dense_embedding(final_query)
            logger.info(f"Dense query embedding created, length: {len(dense_query_embedding)}")
            
            # Prepare dense search request for Qdrant
            search_request = {
                "vector": dense_query_embedding,
                "limit": initial_retrieval_limit, # Fetch more candidates for re-ranking
                "score_threshold": similarity_threshold, # Apply threshold for initial dense retrieval
                "with_payload": True,
                "with_vector": False
            }
            
            # Add filter for job category if specified
            if job_category:
                search_request["filter"] = {
                    "must": [
                        {
                            "key": "job_category",
                            "match": {
                                "value": job_category
                            }
                        }
                    ]
                }
            
            logger.info(f"Qdrant search request payload: {json.dumps(search_request, indent=2)}")
            
            initial_qdrant_results = []
            async with httpx.AsyncClient() as client:
                logger.info(f"Searching in {self.collection_name} collection with initial limit {initial_retrieval_limit}")
                response = await client.post(
                    f"{self.qdrant_url}/collections/{self.collection_name}/points/search",
                    json=search_request
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    # Log the raw Qdrant response data for debugging
                    logger.debug(f"Raw Qdrant response data: {json.dumps(response_data, indent=2)}")
                    initial_qdrant_results = response_data.get("result", [])
                    
                    logger.info(f"Found {len(initial_qdrant_results)} initial results from Qdrant.")
                    logger.debug(f"Type of initial_qdrant_results: {type(initial_qdrant_results)}")
                    if initial_qdrant_results:
                        logger.debug(f"Type of first element in initial_qdrant_results: {type(initial_qdrant_results[0])}")
                        logger.debug(f"First element in initial_qdrant_results: {initial_qdrant_results[0]}")
                else:
                    logger.error(f"Error searching {self.collection_name}: {response.text}")
                    # Fallback to empty results if Qdrant search fails
                    return []

            final_results = []
            reranker = await self.get_reranker_model()

            if reranker and initial_qdrant_results:
                logger.info("Performing re-ranking with Cross-Encoder...")
                rerank_pairs = []
                valid_initial_qdrant_results = [] # To store only valid dict results for re-ranking
                for result in initial_qdrant_results:
                    # Defensive check: Ensure 'result' is a dictionary before processing
                    if not isinstance(result, dict):
                        logger.error(f"Unexpected item type in initial_qdrant_results: Expected dict, got {type(result)}. Skipping item: {result}")
                        continue # Skip this malformed item
                    
                    valid_initial_qdrant_results.append(result) # Add to valid list
                    payload = result.get("payload", {})
                    
                    # Handle 'projects' as a list of strings
                    projects_list_for_reranker = []
                    projects_data = payload.get('projects', [])
                    if isinstance(projects_data, list):
                        for p_item in projects_data:
                            if isinstance(p_item, str):
                                projects_list_for_reranker.append(p_item)
                            else:
                                logger.warning(f"Unexpected item type in 'projects' list: Expected str, got {type(p_item)}. Skipping item: {p_item}")
                    else:
                        logger.warning(f"Unexpected type for 'projects' payload: Expected list, got {type(projects_data)}. Using empty list.")
                    
                    doc_text = f"Title: {payload.get('current_job_title', '')}. " \
                               f"Skills: {', '.join(payload.get('skills', []))}. " \
                               f"Experience: {payload.get('experience_summary', '')}. " \
                               f"Objective: {payload.get('objective', '')}. " \
                               f"Qualifications: {payload.get('qualifications_summary', '')}. " \
                               f"Projects: {', '.join(projects_list_for_reranker)}. " \
                               f"Location: {payload.get('location', '')}." # Added location
                    rerank_pairs.append((final_query, doc_text))
                
                # Only proceed with prediction if rerank_pairs is not empty
                if rerank_pairs:
                    scores = reranker.predict(rerank_pairs)

                    # Combine scores with original valid results and sort
                    # Ensure scores length matches valid_initial_qdrant_results length
                    if len(scores) == len(valid_initial_qdrant_results):
                        scored_results = sorted(zip(scores, valid_initial_qdrant_results), key=lambda x: x[0], reverse=True)
                        logger.info(f"Re-ranking complete. Top score: {scored_results[0][0]:.4f}" if scored_results else "No scored results after re-ranking.")
                        final_results_for_mapping = [item[1] for item in scored_results[:limit]] # 'limit' determines final count
                    else:
                        logger.error(f"Mismatch between reranker scores ({len(scores)}) and valid initial Qdrant results ({len(valid_initial_qdrant_results)}). Returning top Qdrant results without re-ranking.")
                        final_results_for_mapping = valid_initial_qdrant_results[:limit]
                else:
                    logger.warning("No valid items for re-ranking after filtering. Returning top Qdrant results without re-ranking.")
                    final_results_for_mapping = valid_initial_qdrant_results[:limit] # Use the filtered valid results
            else:
                logger.warning("Cross-Encoder reranker not available or no initial results. Returning top Qdrant results without re-ranking.")
                # If no reranker, just take the top 'limit' results from initial Qdrant results, ensuring they are dicts
                final_results_for_mapping = [r for r in initial_qdrant_results if isinstance(r, dict)][:limit]
            
            # Format and map the final results to the expected output structure
            results_to_return = []
            for result in final_results_for_mapping:
                # Another defensive check, though less likely to hit here if previous filtering works
                if not isinstance(result, dict):
                    logger.error(f"Unexpected item type in final_results_for_mapping: Expected dict, got {type(result)}. Skipping item: {result}")
                    continue
                
                payload = result.get("payload", {})
                # Use the re-ranker score if available, otherwise Qdrant score
                # Find the re-ranker score for the current result if it was re-ranked
                reranker_score = None
                if 'scored_results' in locals() and scored_results: # Check if scored_results was populated
                    for s, r in scored_results:
                        if r.get('id') == result.get('id'):
                            reranker_score = s
                            break

                similarity_score = reranker_score if reranker_score is not None else result.get("score", 0.0)
                
                # --- MODIFIED: Convert numpy.float32 to standard float ---
                if isinstance(similarity_score, np.float32):
                    similarity_score = float(similarity_score)

                results_to_return.append({
                    "id": result["id"],
                    "similarity_score": similarity_score, # Use re-ranker score or Qdrant score
                    "collection": self.collection_name,
                    "name": payload.get("name", "Unknown"),
                    "email_id": payload.get("email_id", "Unknown"),
                    "phone_number": payload.get("phone_number", "Unknown"),
                    "location": payload.get("location", "Unknown"),
                    "skills": payload.get("skills", []),
                    "experience_summary": payload.get("experience_summary", ""),
                    "qualifications_summary": payload.get("qualifications_summary", ""),
                    "companies_worked_with_duration": payload.get("companies_worked_with_duration", []),
                    "current_job_title": payload.get("current_job_title", ""),
                    "objective": payload.get("objective", ""),
                    "projects": payload.get("projects", []), # Keep original projects list in final output
                    "certifications": payload.get("certifications", []),
                    "awards_achievements": payload.get("awards_achievements", []),
                    "languages": payload.get("languages", []),
                    "linkedin_url": payload.get("linkedin_url", "Unknown"),
                    "github_url": payload.get("github_url", "Unknown"),
                    "availability_status": payload.get("availability_status"),
                    "work_authorization_status": payload.get("work_authorization_status"),
                    "has_photo": payload.get("has_photo", False),
                    "_original_filename": payload.get("_original_filename", ""),
                    "personal_details": payload.get("personal_details"),
                    "personal_info": payload.get("personal_info"),
                    "_is_master_record": payload.get("_is_master_record", False),
                    "_duplicate_group_id": payload.get("_duplicate_group_id"),
                    "_duplicate_count": payload.get("_duplicate_count", 0),
                    "_associated_original_filenames": payload.get("_associated_original_filenames", []),
                    "_associated_ids": payload.get("_associated_ids", [])
                })
            
            logger.info(f"Returning {len(results_to_return)} final results after re-ranking.")
            return results_to_return
        
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True) # Log full traceback
            return []
    
    async def force_indexing(self) -> bool:
        """Force immediate indexing by updating optimizer config."""
        try:
            async with httpx.AsyncClient() as client:
                update_config = {
                    "optimizer_config": {
                        "indexing_threshold": 0  # Force immediate indexing
                    }
                }
                response = await client.patch(
                    f"{self.qdrant_url}/collections/{self.collection_name}",
                    json=update_config
                )
                logger.info(f"Force indexing response: {response.status_code}, {response.text}")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Force indexing failed: {e}")
            return False

    async def rebuild_index(self) -> bool:
        """Rebuild Qdrant index."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.qdrant_url}/collections/{self.collection_name}/index",
                    json={"wait": True}
                )
                logger.info(f"Index rebuild response: {response.status_code}")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
            return False
    
    async def get_collection_info(self) -> dict:
        """Get collection information."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.qdrant_url}/collections/{self.collection_name}")
                if response.status_code == 200:
                    info = response.json()
                    logger.info(f"Collection info: {info}")
                    return info
                return {}
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    async def health_check(self) -> bool:
        """Check Qdrant health."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.qdrant_url}/collections")
                if response.status_code == 200:
                    collections = response.json()
                    collection_names = [col["name"] for col in collections.get("result", {}).get("collections", [])]
                    logger.info(f"Available Qdrant collections: {collection_names}")
                    if self.collection_name in collection_names:
                        logger.info(f"Collection {self.collection_name} is available")
                    else:
                        logger.warning(f"Collection {self.collection_name} is NOT available")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    async def get_collection_stats(self) -> Dict:
        """Get statistics for the employee_profiles collection, including point count."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.qdrant_url}/collections/{self.collection_name}/points")
                if response.status_code == 200:
                    data = response.json()
                    point_count = data.get("result", {}).get("points_count", 0)
                    return {
                        "collection_name": self.collection_name,
                        "point_count": point_count,
                        "status": "active",
                    }
                logger.error(f"Failed to get collection stats: {response.text}")
                return {"collection_name": self.collection_name, "status": "error", "message": response.text}
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"collection_name": self.collection_name, "status": "error", "message": str(e)}

    def search_similar_resumes(self, query_text: str, job_role: Optional[str] = None, 
                              limit: int = 10, score_threshold: float = 0.7, # score_threshold now applies to initial Qdrant retrieval
                              initial_retrieval_limit: int = 40) -> List[Dict]: # MODIFIED: Default changed to 40
        """
        Synchronous wrapper for search_resumes method.
        `limit` is the final number of results after re-ranking.
        `initial_retrieval_limit` is the number of candidates for re-ranking.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(
            self.search_resumes(
                query_text=query_text,
                job_category=job_role,
                limit=limit, # Final limit (determines re-ranking limit)
                similarity_threshold=score_threshold, # Initial Qdrant threshold
                initial_retrieval_limit=initial_retrieval_limit # Candidates for re-ranking
            )
        )
        
        return results

# Global instance
vector_service = VectorService()

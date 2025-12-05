"""
Simplified 2-level Qdrant vector database client.
- doc_level: Coarse document-level search
- chunk_level: Fine-grained chunk-level search
No sentence-level (can be added later if needed for very long chunks)
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny
)
from qdrant_client.http.exceptions import UnexpectedResponse
from typing import List, Dict, Any, Optional
import numpy as np
from app.config import get_settings
from uuid import UUID

settings = get_settings()


class VectorDBService:
    """Qdrant vector database service with 2-level hierarchy."""
    
    # Collection names
    DOC_COLLECTION = "doc_level"
    CHUNK_COLLECTION = "chunk_level"
    
    def __init__(self):
        """Initialize Qdrant client and create collections if needed."""
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
            timeout=30
        )
        
        self._ensure_collections()
        print(f"✓ Connected to Qdrant: {settings.QDRANT_URL}")
    
    def _ensure_collections(self):
        # Doc-level
        if not self._collection_exists(self.DOC_COLLECTION):
            self._create_doc_collection()
        # Chunk-level
        if not self._collection_exists(self.CHUNK_COLLECTION):
            self._create_chunk_collection()

    def _collection_exists(self, name: str) -> bool:
        try:
            self.client.get_collection(name)
            return True
        except Exception:
            return False

    def _create_doc_collection(self):
        """Create document-level collection."""
        dimension = settings.EMBEDDING_DIMENSION
        try:
            self.client.create_collection(
                collection_name=self.DOC_COLLECTION,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            print(f"  Created collection: {self.DOC_COLLECTION}")
        except UnexpectedResponse as e:
            if getattr(e, "status_code", None) == 409:
                # Already exists, ignore
                print(f"  Collection exists: {self.DOC_COLLECTION}")
            else:
                raise

    def _create_chunk_collection(self):
        """Create chunk-level collection."""
        dimension = settings.EMBEDDING_DIMENSION
        try:
            self.client.create_collection(
                collection_name=self.CHUNK_COLLECTION,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            print(f"  Created collection: {self.CHUNK_COLLECTION}")
        except UnexpectedResponse as e:
            if getattr(e, "status_code", None) == 409:
                print(f"  Collection exists: {self.CHUNK_COLLECTION}")
            else:
                raise
    
    # ========================================================================
    # DOCUMENT-LEVEL OPERATIONS
    # ========================================================================
    
    def upsert_document(
        self, 
        doc_id: str,
        tenant_id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any] = None
    ):
        """
        Upsert document-level vector.
        
        Args:
            doc_id: Document ID
            tenant_id: Tenant ID for multi-tenancy
            vector: Embedding vector
            metadata: Additional metadata
        """
        if metadata is None:
            metadata = {}
        
        payload = {
            "doc_id": doc_id,
            "tenant_id": tenant_id,
            **metadata
        }
        
        # Clean payload (remove None values)
        payload = {k: v for k, v in payload.items() if v is not None}
        
        point = PointStruct(
            id=doc_id,
            vector=vector.tolist(),
            payload=payload
        )
        
        self.client.upsert(
            collection_name=self.DOC_COLLECTION,
            points=[point]
        )
    
    def search_documents(
        self,
        query_vector: np.ndarray,
        tenant_id: str,
        top_k: int = 8,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search documents by vector similarity.
        
        Args:
            query_vector: Query embedding
            tenant_id: Tenant ID for filtering
            top_k: Number of results
            score_threshold: Minimum score threshold
        
        Returns:
            List of search results with metadata and scores
        """
        # Filter by tenant_id
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=tenant_id)
                )
            ]
        )
        
        results = self.client.search(
            collection_name=self.DOC_COLLECTION,
            query_vector=query_vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold
        )
        
        # Format results
        formatted = []
        for result in results:
            formatted.append({
                "doc_id": result.payload.get("doc_id"),
                "score": result.score,
                **result.payload
            })
        
        return formatted
    
    def delete_document(self, doc_id: str):
        """Delete document vector."""
        self.client.delete(
            collection_name=self.DOC_COLLECTION,
            points_selector=[doc_id]
        )
    
    # ========================================================================
    # CHUNK-LEVEL OPERATIONS
    # ========================================================================
    
    def upsert_chunks_batch(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Batch upsert chunk vectors."""
        points = [
            PointStruct(
                id=chunk["chunk_id"],
                vector=emb,
                payload={
                    "chunk_id": chunk["chunk_id"],  # Store chunk_id in payload too
                    "doc_id": chunk["doc_id"],
                    "tenant_id": chunk["tenant_id"],
                    "chunk_index": chunk["idx"],  # Changed from chunk["chunk_index"]
                    "page": chunk.get("page", 1),
                    "text_preview": chunk.get("preview", ""),
                }
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        
        self.client.upsert(
            collection_name=self.CHUNK_COLLECTION,
            points=points,
            wait=True
        )
    
    def search_chunks(
        self,
        query_vector: np.ndarray,
        tenant_id: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 32,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search chunks by vector similarity.
        
        Args:
            query_vector: Query embedding
            tenant_id: Tenant ID for filtering
            doc_ids: Optional list of doc IDs to filter by
            top_k: Number of results
            score_threshold: Minimum score threshold
        
        Returns:
            List of search results with metadata and scores
        """
        # Build filter
        must_conditions = [
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id)
            )
        ]
        
        # Add doc_id filter if provided
        if doc_ids:
            if len(doc_ids) == 1:
                must_conditions.append(
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_ids[0])
                    )
                )
            else:
                must_conditions.append(
                    FieldCondition(
                        key="doc_id",
                        match=MatchAny(any=doc_ids)
                    )
                )
        
        query_filter = Filter(must=must_conditions)
        
        results = self.client.search(
            collection_name=self.CHUNK_COLLECTION,
            query_vector=query_vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold
        )
        
        # Format results
        formatted = []
        for result in results:
            formatted.append({
                "chunk_id": result.payload.get("chunk_id"),
                "doc_id": result.payload.get("doc_id"),
                "score": result.score,
                **result.payload
            })
        
        return formatted
    
    def delete_chunks_by_doc(self, doc_id: str):
        """Delete all chunks for a document."""
        # Filter by doc_id
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        )
        
        self.client.delete(
            collection_name=self.CHUNK_COLLECTION,
            points_selector=query_filter
        )
    
    # ========================================================================
    # COLLECTION MANAGEMENT
    # ========================================================================
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics."""
        info = self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status
        }
    
    def delete_collection(self, collection_name: str):
        """Delete entire collection (use with caution!)."""
        self.client.delete_collection(collection_name)
    
    def recreate_collections(self):
        """Delete and recreate all collections (for fresh start)."""
        self.client.delete_collection(self.DOC_COLLECTION)
        self.client.delete_collection(self.CHUNK_COLLECTION)
        self._ensure_collections()


# Global vector DB instance
_vector_db: VectorDBService = None


def get_vector_db() -> VectorDBService:
    """Get or create global vector database service."""
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDBService()
    return _vector_db

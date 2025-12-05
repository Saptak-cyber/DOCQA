"""
Reranker service abstraction layer.
Supports multiple reranking models and providers.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from sentence_transformers import CrossEncoder
from app.config import get_settings

settings = get_settings()


class RerankService(ABC):
    """Abstract base class for reranking services."""
    
    @abstractmethod
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        text_key: str = "text",
        top_k: int = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank candidates based on query relevance.
        
        Args:
            query: Query string
            candidates: List of candidate documents (dicts with text)
            text_key: Key in dict to extract text from
            top_k: Return only top k results (optional)
        
        Returns:
            List of (candidate, score) tuples sorted by relevance
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name."""
        pass


class CrossEncoderReranker(RerankService):
    """
    Cross-Encoder based reranking service.
    Supports MS-MARCO, SBERT rerankers, etc.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        self._model_name = model_name or settings.RERANKER_MODEL
        self._device = device or "cpu"
        self._model = CrossEncoder(self._model_name, device=self._device)
        
        print(f"✓ Loaded reranker model: {self._model_name}")
        print(f"  Device: {self._device}")
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        text_key: str = "text",
        top_k: int = None,
        batch_size: int = 32
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Query string
            candidates: List of candidate documents
            text_key: Key to extract text from candidates
            top_k: Return only top k results
            batch_size: Batch size for scoring
        
        Returns:
            List of (candidate, score) tuples sorted by score descending
        """
        if not candidates:
            return []
        
        # Extract texts
        texts = [c.get(text_key, "") for c in candidates]
        
        # Create query-document pairs
        pairs = [[query, text] for text in texts]
        
        # Score all pairs
        scores = self._model.predict(pairs, batch_size=batch_size)
        
        # Combine candidates with scores
        results = list(zip(candidates, scores))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k if specified
        if top_k:
            results = results[:top_k]
        
        return results
    
    @property
    def model_name(self) -> str:
        return self._model_name


class CohereReranker(RerankService):
    """
    Cohere reranking service (for future use).
    Uses Cohere's rerank API.
    """
    
    def __init__(self, model_name: str = "rerank-english-v2.0", api_key: str = None):
        self._model_name = model_name
        self._api_key = api_key
        
        print(f"✓ Configured Cohere reranker: {self._model_name}")
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        text_key: str = "text",
        top_k: int = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Rerank using Cohere API."""
        # TODO: Implement Cohere API call
        raise NotImplementedError("Cohere reranking not yet implemented")
    
    @property
    def model_name(self) -> str:
        return self._model_name


# Global reranker instance
_rerank_service: RerankService = None


def get_rerank_service() -> RerankService:
    """
    Get or create the global reranker service instance.
    Singleton to avoid loading model multiple times.
    """
    global _rerank_service
    
    if _rerank_service is None:
        model_name = settings.RERANKER_MODEL
        
        if model_name.startswith("rerank-"):
            _rerank_service = CohereReranker(model_name=model_name)
        else:
            _rerank_service = CrossEncoderReranker(model_name=model_name)
    
    return _rerank_service


def rerank_candidates(
    query: str, 
    candidates: List[Dict[str, Any]], 
    text_key: str = "text",
    top_k: int = None,
    use_cache: bool = True
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Convenience function to rerank candidates.
    Optionally uses cache to avoid re-reranking.
    
    Args:
        query: Query string
        candidates: List of candidates
        text_key: Key to extract text
        top_k: Number of results to return
        use_cache: Whether to use caching
    
    Returns:
        List of (candidate, score) tuples
    """
    if use_cache:
        from app.services.cache_service import get_cached_rerank, cache_rerank
        
        # Try cache
        cached = get_cached_rerank(query, candidates, text_key)
        if cached is not None:
            return cached[:top_k] if top_k else cached
        
        # Rerank and cache
        service = get_rerank_service()
        results = service.rerank(query, candidates, text_key, top_k)
        cache_rerank(query, candidates, text_key, results)
        
        return results
    else:
        service = get_rerank_service()
        return service.rerank(query, candidates, text_key, top_k)

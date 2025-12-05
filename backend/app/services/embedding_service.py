"""
Embedding service abstraction layer.
Supports multiple embedding models and providers.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import get_settings

settings = get_settings()


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        Returns: numpy array of shape (n, embedding_dim)
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name."""
        pass


class SentenceTransformerEmbedding(EmbeddingService):
    """
    SentenceTransformer-based embedding service.
    Supports models like all-mpnet-base-v2, bge-*, e5-*, etc.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._device = device or "cpu"
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        
        print(f"✓ Loaded embedding model: {self._model_name}")
        print(f"  Device: {self._device}, Dimension: {self._dimension}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings with batching and normalization.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings (recommended for cosine similarity)
        
        Returns:
            numpy array of shape (n, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model_name


class OpenAIEmbedding(EmbeddingService):
    """
    OpenAI embedding service (for future use).
    Supports text-embedding-ada-002, text-embedding-3-small, etc.
    """
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None):
        self._model_name = model_name
        self._api_key = api_key
        # Dimensions for different models
        self._dimension_map = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        self._dimension = self._dimension_map.get(model_name, 1536)
        
        print(f"✓ Configured OpenAI embedding: {self._model_name}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        # TODO: Implement OpenAI API call
        raise NotImplementedError("OpenAI embedding not yet implemented")
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model_name


# Global embedding service instance
_embedding_service: EmbeddingService = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create the global embedding service instance.
    This is a singleton to avoid loading the model multiple times.
    """
    global _embedding_service
    
    if _embedding_service is None:
        # Determine which service to use based on config
        model_name = settings.EMBEDDING_MODEL
        
        if model_name.startswith("text-embedding-"):
            _embedding_service = OpenAIEmbedding(model_name=model_name)
        else:
            _embedding_service = SentenceTransformerEmbedding(model_name=model_name)
    
    return _embedding_service


def embed_texts(texts: List[str], use_cache: bool = True) -> np.ndarray:
    """
    Convenience function to embed texts.
    Optionally uses cache to avoid re-embedding.
    
    Args:
        texts: List of text strings
        use_cache: Whether to check/store in cache
    
    Returns:
        numpy array of embeddings
    """
    if use_cache:
        from app.services.cache_service import get_cached_embeddings, cache_embeddings
        
        # Try to get from cache
        cached = get_cached_embeddings(texts)
        if cached is not None:
            return cached
        
        # Generate and cache
        service = get_embedding_service()
        embeddings = service.embed_texts(texts)
        cache_embeddings(texts, embeddings)
        
        return embeddings
    else:
        service = get_embedding_service()
        return service.embed_texts(texts)

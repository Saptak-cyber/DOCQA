"""
Enhanced caching service with support for:
- Embedding cache
- Rerank cache  
- Answer cache
- General key-value cache
"""

import redis
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from app.config import get_settings

settings = get_settings()


class CacheService:
    """Redis-based caching service."""
    
    def __init__(self):
        self.redis_client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=False  # We'll handle encoding ourselves
        )
        print(f"✓ Connected to Redis: {settings.REDIS_URL}")
    
    def _make_key(self, prefix: str, data: str) -> str:
        """Create stable cache key using SHA-256 hash."""
        hash_obj = hashlib.sha256(data.encode('utf-8'))
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get(self, key: str) -> Optional[bytes]:
        """Get raw value from cache."""
        try:
            return self.redis_client.get(key)
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: bytes, ttl: int = None):
        """Set raw value in cache with optional TTL."""
        try:
            if ttl:
                self.redis_client.setex(key, ttl, value)
            else:
                self.redis_client.set(key, value)
        except Exception as e:
            print(f"Cache set error: {e}")
    
    def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value from cache."""
        value = self.get(key)
        if value:
            try:
                return json.loads(value.decode('utf-8'))
            except Exception as e:
                print(f"Cache JSON decode error: {e}")
                return None
        return None
    
    def set_json(self, key: str, value: Any, ttl: int = None):
        """Set JSON value in cache."""
        try:
            json_bytes = json.dumps(value).encode('utf-8')
            self.set(key, json_bytes, ttl)
        except Exception as e:
            print(f"Cache JSON encode error: {e}")
    
    def delete(self, key: str):
        """Delete key from cache."""
        try:
            self.redis_client.delete(key)
        except Exception as e:
            print(f"Cache delete error: {e}")
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return self.redis_client.exists(key) > 0
        except Exception:
            return False


# Global cache instance
_cache_service: CacheService = None


def get_cache() -> CacheService:
    """Get or create global cache service."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service


# ============================================================================
# EMBEDDING CACHE
# ============================================================================

def get_cached_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    """
    Get cached embeddings for texts.
    Returns None if any text is not cached (to maintain consistency).
    """
    cache = get_cache()
    embeddings = []
    
    for text in texts:
        key = cache._make_key("emb", text)
        cached = cache.get_json(key)
        
        if cached is None:
            return None  # Cache miss - need to recompute all
        
        embeddings.append(cached)
    
    return np.array(embeddings)


def cache_embeddings(texts: List[str], embeddings: np.ndarray):
    """Cache embeddings for texts."""
    cache = get_cache()
    ttl = settings.CACHE_TTL_EMBEDDING
    
    for text, embedding in zip(texts, embeddings):
        key = cache._make_key("emb", text)
        # Convert numpy array to list for JSON serialization
        embedding_list = embedding.tolist()
        cache.set_json(key, embedding_list, ttl)


# ============================================================================
# RERANK CACHE
# ============================================================================

def get_cached_rerank(
    query: str, 
    candidates: List[Dict[str, Any]], 
    text_key: str
) -> Optional[List[Tuple[Dict[str, Any], float]]]:
    """
    Get cached rerank results.
    Cache key based on query + all candidate texts.
    """
    cache = get_cache()
    
    # Build stable cache key from query + candidate texts
    texts = [c.get(text_key, "") for c in candidates]
    cache_data = f"{query}||{'||'.join(texts)}"
    key = cache._make_key("rerank", cache_data)
    
    cached = cache.get_json(key)
    if cached:
        # Reconstruct results
        results = []
        for i, score in enumerate(cached):
            results.append((candidates[i], score))
        return results
    
    return None


def cache_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    text_key: str,
    results: List[Tuple[Dict[str, Any], float]]
):
    """Cache rerank results."""
    cache = get_cache()
    ttl = settings.CACHE_TTL_RERANK
    
    # Build cache key
    texts = [c.get(text_key, "") for c in candidates]
    cache_data = f"{query}||{'||'.join(texts)}"
    key = cache._make_key("rerank", cache_data)
    
    # Store only scores (candidates are passed in)
    scores = [score for _, score in results]
    cache.set_json(key, scores, ttl)


# ============================================================================
# ANSWER CACHE
# ============================================================================

def get_cached_answer(
    tenant_id: str,
    user_id: str,
    query: str
) -> Optional[Dict[str, Any]]:
    """
    Get cached answer for a query.
    Cache key includes tenant and user for isolation.
    """
    cache = get_cache()
    
    cache_data = f"{tenant_id}::{user_id}::{query}"
    key = cache._make_key("qa", cache_data)
    
    return cache.get_json(key)


def cache_answer(
    tenant_id: str,
    user_id: str,
    query: str,
    answer_data: Dict[str, Any]
):
    """
    Cache complete answer with citations.
    
    Args:
        tenant_id: Tenant ID
        user_id: User ID
        query: Query text
        answer_data: Full response dict {query, answer, citations, ...}
    """
    cache = get_cache()
    ttl = settings.CACHE_TTL_ANSWER
    
    cache_data = f"{tenant_id}::{user_id}::{query}"
    key = cache._make_key("qa", cache_data)
    
    cache.set_json(key, answer_data, ttl)


# ============================================================================
# CHUNK TEXT CACHE (for backwards compatibility, but now stored in PostgreSQL)
# ============================================================================

def get_cached_chunk_text(chunk_id: str) -> Optional[str]:
    """Get cached chunk text."""
    cache = get_cache()
    key = f"chunk_text:{chunk_id}"
    
    cached = cache.get(key)
    if cached:
        return cached.decode('utf-8')
    return None


def cache_chunk_text(chunk_id: str, text: str):
    """Cache chunk text."""
    cache = get_cache()
    key = f"chunk_text:{chunk_id}"
    ttl = settings.CACHE_TTL_RERANK  # Same TTL as rerank
    
    cache.set(key, text.encode('utf-8'), ttl)


# ============================================================================
# RATE LIMITING HELPERS
# ============================================================================

def check_rate_limit(tenant_id: str, limit: int, window_seconds: int) -> bool:
    """
    Check if tenant is within rate limit.
    
    Args:
        tenant_id: Tenant ID
        limit: Maximum requests allowed
        window_seconds: Time window in seconds
    
    Returns:
        True if within limit, False if exceeded
    """
    cache = get_cache()
    key = f"rate:{tenant_id}:{window_seconds}"
    
    try:
        # Increment counter
        count = cache.redis_client.incr(key)
        
        # Set expiry on first request
        if count == 1:
            cache.redis_client.expire(key, window_seconds)
        
        return count <= limit
    
    except Exception as e:
        print(f"Rate limit check error: {e}")
        return True  # Fail open


def get_rate_limit_remaining(tenant_id: str, limit: int, window_seconds: int) -> int:
    """Get remaining requests in current window."""
    cache = get_cache()
    key = f"rate:{tenant_id}:{window_seconds}"
    
    try:
        current = cache.get(key)
        if current is None:
            return limit
        
        count = int(current.decode('utf-8'))
        return max(0, limit - count)
    
    except Exception:
        return limit

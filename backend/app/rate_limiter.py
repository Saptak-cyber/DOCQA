"""
Rate limiting middleware using Redis.
Protects API endpoints from abuse.
"""

from fastapi import Request, HTTPException, status
from app.services.cache_service import check_rate_limit, get_rate_limit_remaining
from app.config import get_settings
from app.auth import AuthContext

settings = get_settings()


class RateLimiter:
    """Rate limiter for API endpoints."""
    
    def __init__(
        self, 
        requests_per_minute: int = None,
        requests_per_hour: int = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Max requests per minute (default from settings)
            requests_per_hour: Max requests per hour (default from settings)
        """
        self.requests_per_minute = requests_per_minute or settings.RATE_LIMIT_PER_MINUTE
        self.requests_per_hour = requests_per_hour or settings.RATE_LIMIT_PER_HOUR
    
    async def __call__(self, request: Request, auth_context: AuthContext):
        """
        Check rate limits for the request.
        
        Args:
            request: FastAPI request
            auth_context: Authentication context with tenant_id
        
        Raises:
            HTTPException: If rate limit exceeded
        """
        tenant_id = auth_context.tenant_id
        
        # Check per-minute limit
        if not check_rate_limit(tenant_id, self.requests_per_minute, 60):
            remaining = get_rate_limit_remaining(tenant_id, self.requests_per_minute, 60)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.requests_per_minute} requests per minute. Try again later.",
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": "60",
                    "Retry-After": "60"
                }
            )
        
        # Check per-hour limit
        if not check_rate_limit(tenant_id, self.requests_per_hour, 3600):
            remaining = get_rate_limit_remaining(tenant_id, self.requests_per_hour, 3600)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.requests_per_hour} requests per hour. Try again later.",
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_hour),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": "3600",
                    "Retry-After": "3600"
                }
            )


def get_rate_limiter(
    requests_per_minute: int = None,
    requests_per_hour: int = None
) -> RateLimiter:
    """
    Create a rate limiter dependency.
    
    Usage:
        @app.post("/api/v1/query", dependencies=[Depends(get_rate_limiter())])
        async def query_endpoint(...):
            ...
    
    Args:
        requests_per_minute: Override default per-minute limit
        requests_per_hour: Override default per-hour limit
    
    Returns:
        RateLimiter instance
    """
    return RateLimiter(requests_per_minute, requests_per_hour)


# Pre-configured rate limiters for different endpoint types

def standard_rate_limit():
    """Standard rate limit for most endpoints."""
    return get_rate_limiter()


def heavy_rate_limit():
    """Heavy rate limit for expensive operations (uploads, ingestion)."""
    return get_rate_limiter(requests_per_minute=5, requests_per_hour=20)


def query_rate_limit():
    """Rate limit specifically for query endpoints."""
    return get_rate_limiter(requests_per_minute=10, requests_per_hour=100)

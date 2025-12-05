"""
Authentication and authorization middleware.
Supports API key-based authentication with multi-tenancy.
"""

from fastapi import Security, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime
from typing import Optional
from app.database import get_db
from app.models import APIKey

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer security scheme
security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage."""
    return pwd_context.hash(api_key)


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash."""
    return pwd_context.verify(plain_key, hashed_key)


class AuthContext:
    """Authentication context containing tenant and user info."""
    
    def __init__(self, tenant_id: str, user_id: str = None, api_key_id: str = None):
        self.tenant_id = tenant_id
        self.user_id = user_id or tenant_id  # Default user_id to tenant_id
        self.api_key_id = api_key_id


async def get_auth_context(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db)
) -> AuthContext:
    """
    Verify API key and extract authentication context.
    
    Args:
        credentials: HTTP Bearer token from request
        db: Database session
    
    Returns:
        AuthContext with tenant_id and user_id
    
    Raises:
        HTTPException: If authentication fails
    """
    api_key = credentials.credentials
    
    # Query all active API keys (in production, add indexing on key_hash)
    api_keys = db.query(APIKey).filter(
        APIKey.is_active == True
    ).all()
    
    # Check each key
    for key_record in api_keys:
        if verify_api_key(api_key, key_record.key_hash):
            # Check expiry
            if key_record.expires_at and key_record.expires_at < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key has expired"
                )
            
            # Update last used timestamp
            key_record.last_used_at = datetime.utcnow()
            db.commit()
            
            return AuthContext(
                tenant_id=key_record.tenant_id,
                api_key_id=key_record.key_id
            )
    
    # No matching key found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "Bearer"}
    )


async def get_optional_auth_context(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_optional),
    db: Session = Depends(get_db)
) -> Optional[AuthContext]:
    """
    Optional authentication - returns None if no credentials provided.
    Useful for endpoints that work with or without authentication.
    """
    if credentials is None:
        return None
    
    try:
        return await get_auth_context(credentials, db)
    except HTTPException:
        return None


def create_api_key_for_tenant(
    db: Session,
    tenant_id: str,
    name: str = None,
    rate_limit_minute: int = 10,
    rate_limit_hour: int = 100,
    expires_at: datetime = None
) -> tuple[str, APIKey]:
    """
    Create a new API key for a tenant.
    
    Args:
        db: Database session
        tenant_id: Tenant ID
        name: Friendly name for the key
        rate_limit_minute: Requests per minute
        rate_limit_hour: Requests per hour
        expires_at: Optional expiry date
    
    Returns:
        Tuple of (plain_api_key, APIKey record)
        IMPORTANT: plain_api_key is only returned once!
    """
    import secrets
    
    # Generate random API key (32 bytes = 256 bits)
    plain_key = secrets.token_urlsafe(32)
    
    # Hash for storage
    key_hash = hash_api_key(plain_key)
    
    # Create record
    api_key_record = APIKey(
        tenant_id=tenant_id,
        key_hash=key_hash,
        name=name,
        is_active=True,
        rate_limit_minute=rate_limit_minute,
        rate_limit_hour=rate_limit_hour,
        expires_at=expires_at
    )
    
    db.add(api_key_record)
    db.commit()
    db.refresh(api_key_record)
    
    return plain_key, api_key_record


def deactivate_api_key(db: Session, key_id: str):
    """Deactivate an API key."""
    key = db.query(APIKey).filter(APIKey.key_id == key_id).first()
    if key:
        key.is_active = False
        db.commit()


def list_tenant_api_keys(db: Session, tenant_id: str) -> list[APIKey]:
    """List all API keys for a tenant."""
    return db.query(APIKey).filter(APIKey.tenant_id == tenant_id).all()

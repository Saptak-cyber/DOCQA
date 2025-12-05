"""
Database models for Document Q&A System.
Improvements:
- Multi-tenancy support with tenant_id
- Full text storage in PostgreSQL (not Cloudinary)
- Soft delete support
- Enhanced metadata tracking
"""

from sqlalchemy import Column, String, Integer, Text, Boolean, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import uuid

Base = declarative_base()


class Document(Base):
    """Document model with multi-tenancy and soft delete support."""
    
    __tablename__ = "documents"
    
    # Primary fields
    doc_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    tenant_id = Column(String, nullable=False, index=True)  # Multi-tenancy support
    user_id = Column(String, nullable=False, index=True)
    
    # Document metadata
    filename = Column(String, nullable=False)
    cloudinary_url = Column(String, nullable=False)  # PDF storage only
    pages = Column(Integer, default=0)
    
    # Status tracking
    status = Column(String, default="queued", index=True)  # queued, processing, ready, failed
    
    # Soft delete
    is_deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_tenant_user', 'tenant_id', 'user_id'),
        Index('idx_tenant_status', 'tenant_id', 'status'),
        Index('idx_tenant_deleted', 'tenant_id', 'is_deleted'),
    )
    
    def soft_delete(self):
        """Mark document as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()


class Chunk(Base):
    """
    Chunk model with full text storage in PostgreSQL.
    No more Cloudinary for chunk text - simpler and faster!
    """
    
    __tablename__ = "chunks"
    
    # Primary fields
    chunk_id = Column(String, primary_key=True, index=True)  # Format: {doc_id}_{idx}
    doc_id = Column(String, ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Chunk metadata
    page = Column(Integer, nullable=True)
    idx = Column(Integer, nullable=False)  # Chunk index within document
    
    # Text storage (moved from Cloudinary to PostgreSQL)
    text = Column(Text, nullable=False)  # Full chunk text
    preview = Column(String(500))  # Short preview for display
    
    # Multi-tenancy (denormalized for faster queries)
    tenant_id = Column(String, nullable=False, index=True)
    
    # Soft delete
    is_deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    # Indexes
    __table_args__ = (
        Index('idx_doc_page', 'doc_id', 'page'),
        Index('idx_tenant_doc', 'tenant_id', 'doc_id'),
        Index('idx_chunk_deleted', 'is_deleted'),
    )
    
    def soft_delete(self):
        """Mark chunk as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()


class QueryHistory(Base):
    """Store query history for analytics and caching."""
    
    __tablename__ = "query_history"
    
    query_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    tenant_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    
    # Query details
    query_text = Column(Text, nullable=False)
    answer = Column(Text)
    
    # Performance metrics
    latency_ms = Column(Integer)  # Total query time
    doc_count = Column(Integer)   # Documents searched
    chunk_count = Column(Integer) # Chunks retrieved
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_tenant_user_created', 'tenant_id', 'user_id', 'created_at'),
    )


class APIKey(Base):
    """API key management for multi-tenant authentication."""
    
    __tablename__ = "api_keys"
    
    key_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, nullable=False, unique=True, index=True)
    
    # Key details
    key_hash = Column(String, nullable=False)  # Hashed API key
    name = Column(String)  # Friendly name
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    
    # Rate limiting metadata (stored for tracking)
    rate_limit_minute = Column(Integer, default=10)
    rate_limit_hour = Column(Integer, default=100)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)

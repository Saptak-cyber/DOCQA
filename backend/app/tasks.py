"""
Celery tasks for document ingestion and processing.
Organized into separate modules/queues for better management.
"""

from celery import Task
import pdfplumber
import tempfile
import os
from typing import List, Dict, Any
import numpy as np
from datetime import datetime
import uuid

from app.celery_app import celery_app
from app.database import get_db_context
from app.models import Document, Chunk
from app.services.embedding_service import get_embedding_service
from app.services.vector_db_service import get_vector_db
from app.services.cache_service import cache_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cloudinary.uploader
from app.config import get_settings

settings = get_settings()


class CallbackTask(Task):
    """Base task with callbacks for progress tracking."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        print(f"Task {task_id} failed: {exc}")
        
        # Update document status to failed
        if args and len(args) > 0:
            doc_id = args[0]
            with get_db_context() as db:
                doc = db.query(Document).filter(Document.doc_id == doc_id).first()
                if doc:
                    doc.status = "failed"
                    db.commit()


@celery_app.task(
    name="app.tasks.ingestion.ingest_document",
    base=CallbackTask,
    queue="ingestion",
    bind=True
)
def ingest_document(self, doc_id: str, pdf_url: str, tenant_id: str, user_id: str):
    """
    Ingest a document: extract text, chunk, embed, and index.
    
    This is the IMPROVED VERSION:
    - No sentence-level indexing (truly lazy - can be added on-demand later)
    - Chunk text stored in PostgreSQL, not Cloudinary
    - Multi-tenancy support
    
    Args:
        doc_id: Document ID
        pdf_url: Cloudinary URL of PDF
        tenant_id: Tenant ID
        user_id: User ID
    """
    print(f"[{doc_id}] Starting ingestion...")
    
    with get_db_context() as db:
        # Update status
        doc = db.query(Document).filter(Document.doc_id == doc_id).first()
        if not doc:
            raise ValueError(f"Document {doc_id} not found")
        
        doc.status = "processing"
        db.commit()
        
        try:
            # Step 1: Download PDF
            print(f"[{doc_id}] Downloading PDF...")
            temp_file = download_pdf(pdf_url)
            
            # Step 2: Extract text from pages
            print(f"[{doc_id}] Extracting text...")
            pages_text = extract_pdf_pages(temp_file)
            doc.pages = len(pages_text)
            db.commit()
            
            # Step 3: Generate document-level embedding
            print(f"[{doc_id}] Generating document embedding...")
            doc_summary = " ".join(pages_text)[:4000]  # First 4000 chars
            embed_service = get_embedding_service()
            doc_embedding = embed_service.embed_texts([doc_summary])[0]
            
            # Step 4: Upsert document vector to Qdrant
            print(f"[{doc_id}] Upserting document vector...")
            vector_db = get_vector_db()
            vector_db.upsert_document(
                doc_id=doc_id,
                tenant_id=tenant_id,
                vector=doc_embedding,
                metadata={
                    "filename": doc.filename,
                    "pages": doc.pages,
                    "user_id": user_id
                }
            )
            
            # Step 5: Chunk the document
            print(f"[{doc_id}] Chunking text...")
            chunks_data = chunk_document(pages_text, doc_id)
            
            # Step 6: Extract chunk texts and generate embeddings
            print(f"[{doc_id}] Generating chunk embeddings...")
            chunk_texts = [chunk["text"] for chunk in chunks_data]
            chunk_embeddings = embed_service.embed_texts(chunk_texts)
            
            # Cache embeddings (optional but recommended)
            cache_embeddings(chunk_texts, chunk_embeddings)
            
            # Step 7: Store chunks in PostgreSQL
            print(f"[{doc_id}] Storing chunks in database...")
            chunk_records = []
            for chunk_data in chunks_data:
                chunk_record = Chunk(
                    chunk_id=chunk_data["chunk_id"],
                    doc_id=doc_id,
                    tenant_id=tenant_id,
                    page=chunk_data["page"],
                    idx=chunk_data["idx"],
                    text=chunk_data["text"],  # Store full text in PostgreSQL!
                    preview=chunk_data["text"][:500],  # Short preview
                    is_deleted=False
                )
                chunk_records.append(chunk_record)
            
            db.add_all(chunk_records)
            db.commit()
            
            # Step 8: Upsert chunk vectors to Qdrant
            print(f"[{doc_id}] Upserting chunk vectors...")
            qdrant_chunks = []
            for chunk_data in chunks_data:
                qdrant_chunks.append({
                    "chunk_id": chunk_data["chunk_id"],
                    "doc_id": doc_id,
                    "tenant_id": tenant_id,
                    "page": chunk_data["page"],
                    "idx": chunk_data["idx"],
                    "preview": chunk_data["text"][:400],  # For display
                    "user_id": user_id
                })
            
            vector_db.upsert_chunks_batch(qdrant_chunks, chunk_embeddings)
            
            # Step 9: Mark document as ready
            doc.status = "ready"
            db.commit()
            
            print(f"[{doc_id}] ✓ Ingestion complete! {len(chunks_data)} chunks indexed.")
            
            # Cleanup
            os.unlink(temp_file)
            
            return {
                "doc_id": doc_id,
                "status": "ready",
                "pages": doc.pages,
                "chunks": len(chunks_data)
            }
        
        except Exception as e:
            print(f"[{doc_id}] ✗ Ingestion failed: {e}")
            doc.status = "failed"
            db.commit()
            raise


def download_pdf(pdf_url: str) -> str:
    """
    Download PDF from Cloudinary to temporary file.
    
    Args:
        pdf_url: Cloudinary URL
    
    Returns:
        Path to temporary file
    """
    import requests
    
    # Add attachment transformation if needed
    if "fl_attachment" not in pdf_url:
        if "?" in pdf_url:
            pdf_url = pdf_url.replace("/upload/", "/upload/fl_attachment/")
        else:
            pdf_url = pdf_url.replace("/upload/", "/upload/fl_attachment/")
    
    response = requests.get(pdf_url, stream=True)
    response.raise_for_status()
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    for chunk in response.iter_content(chunk_size=8192):
        temp_file.write(chunk)
    temp_file.close()
    
    # Verify PDF
    with open(temp_file.name, "rb") as f:
        header = f.read(4)
        if header != b"%PDF":
            raise ValueError("Downloaded file is not a valid PDF")
    
    return temp_file.name


def extract_pdf_pages(pdf_path: str) -> List[str]:
    """
    Extract text from all pages of a PDF.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        List of page texts
    """
    pages_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text.strip())
    
    return pages_text


def chunk_document(pages_text: List[str], doc_id: str) -> List[Dict[str, Any]]:
    """Chunk document text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for page_num, page_text in enumerate(pages_text, start=1):
        page_chunks = splitter.split_text(page_text)
        
        for chunk_text in page_chunks:
            chunk_id = str(uuid.uuid4())
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "idx": len(chunks),  # Changed from chunk_index to idx
                "page": page_num,
                "text": chunk_text,
                "char_count": len(chunk_text),
            })
    
    return chunks


# ============================================================================
# MAINTENANCE TASKS
# ============================================================================

@celery_app.task(name="app.tasks.maintenance.cleanup_deleted", queue="maintenance")
def cleanup_deleted_documents():
    """
    Periodic task to clean up soft-deleted documents.
    Removes vectors from Qdrant and old database records.
    """
    print("Running cleanup of deleted documents...")
    
    from datetime import timedelta
    cutoff_date = datetime.utcnow() - timedelta(days=30)
    
    with get_db_context() as db:
        # Find old deleted documents
        deleted_docs = db.query(Document).filter(
            Document.is_deleted == True,
            Document.deleted_at < cutoff_date
        ).all()
        
        vector_db = get_vector_db()
        
        for doc in deleted_docs:
            print(f"  Cleaning up {doc.doc_id}...")
            
            # Delete from Qdrant
            try:
                vector_db.delete_document(doc.doc_id)
                vector_db.delete_chunks_by_doc(doc.doc_id)
            except Exception as e:
                print(f"  Warning: Qdrant deletion failed - {e}")
            
            # Delete from PostgreSQL (cascade will delete chunks)
            db.delete(doc)
        
        db.commit()
        
        print(f"✓ Cleaned up {len(deleted_docs)} documents")
        return len(deleted_docs)


@celery_app.task(name="app.tasks.maintenance.clear_old_cache", queue="maintenance")
def clear_old_cache_keys():
    """Clear old cache keys to prevent memory bloat."""
    from app.services.cache_service import get_cache
    
    print("Clearing old cache keys...")
    # Redis TTL handles most cleanup automatically
    # This task can be extended for custom cache management
    
    cache = get_cache()
    # Add custom cleanup logic here if needed
    
    print("✓ Cache cleanup complete")
    return True

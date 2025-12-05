"""
Main FastAPI application with improved architecture:
- Streaming responses for queries
- Multi-tenancy with authentication
- Rate limiting
- Enhanced caching
- 2-level retrieval (doc + chunk only, no sentence)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import cloudinary
import cloudinary.uploader
import time
import json

from app.config import get_settings
from app.database import get_db, init_db
from app.models import Document, Chunk, QueryHistory
from app.auth import get_auth_context, get_optional_auth_context, AuthContext
from app.rate_limiter import query_rate_limit, heavy_rate_limit
from app.tasks import ingest_document
from app.services.embedding_service import embed_texts
from app.services.rerank_service import rerank_candidates
from app.services.llm_service import generate_answer, build_rag_prompt
from app.services.vector_db_service import get_vector_db
from app.services.cache_service import get_cached_answer, cache_answer

settings = get_settings()

# Configure Cloudinary
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True
)

# Create FastAPI app
app = FastAPI(
    title="Document Q&A System",
    description="RAG-powered document Q&A with multi-tenancy and streaming",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class UploadResponse(BaseModel):
    success: bool
    doc_id: str
    message: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_docs: int = Field(default=8, ge=1, le=20)
    top_chunks_per_doc: int = Field(default=4, ge=1, le=10)


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    page: Optional[int]
    score: float
    preview: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    latency_ms: int


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    status: str
    created_at: datetime


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("🚀 Starting Document Q&A System...")
    init_db()
    print("✓ Database initialized")
    
    # Pre-load models
    from app.services.embedding_service import get_embedding_service
    from app.services.rerank_service import get_rerank_service
    from app.services.llm_service import get_llm_service
    from app.services.vector_db_service import get_vector_db
    
    get_embedding_service()
    get_rerank_service()
    get_llm_service()
    get_vector_db()
    
    print("✓ All services loaded")
    print(f"✓ API ready at {settings.API_V1_PREFIX}")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Document Q&A System",
        "version": "2.0.0",
        "status": "healthy"
    }


@app.post(
    f"{settings.API_V1_PREFIX}/upload",
    response_model=UploadResponse,
    dependencies=[Depends(heavy_rate_limit)]
)
async def upload_document(
    file: UploadFile = File(...),
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """
    Upload a PDF document for ingestion.
    
    - Uploads PDF to Cloudinary
    - Queues async ingestion task
    - Returns immediately with doc_id
    """
    # Validate PDF
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    try:
        # Upload to Cloudinary
        contents = await file.read()
        upload_result = cloudinary.uploader.upload(
            contents,
            resource_type="auto",
            folder="pdfs"
        )
        cloudinary_url = upload_result["secure_url"]
        
        # Create document record
        doc_id = str(uuid.uuid4())
        document = Document(
            doc_id=doc_id,
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            filename=file.filename,
            cloudinary_url=cloudinary_url,
            status="queued"
        )
        
        db.add(document)
        db.commit()
        
        # Trigger async ingestion with correct task name
        from app.celery_app import celery_app
        celery_app.send_task(
            "app.tasks.ingestion.ingest_document",
            args=[doc_id, cloudinary_url, auth.tenant_id, auth.user_id],
            queue="ingestion"
        )
        
        return UploadResponse(
            success=True,
            doc_id=doc_id,
            message=f"Document queued for ingestion: {file.filename}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@app.post(
    f"{settings.API_V1_PREFIX}/query",
    response_model=QueryResponse,
    dependencies=[Depends(query_rate_limit)]
)
async def query_documents(
    request: QueryRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """
    Query documents with RAG.
    
    Improved pipeline:
    1. Check answer cache
    2. Embed query
    3. Doc-level search (coarse)
    4. Chunk-level search (fine)
    5. Rerank chunks
    6. Build context
    7. Generate answer
    8. Cache answer
    """
    start_time = time.time()
    
    # Check cache first
    cached = get_cached_answer(auth.tenant_id, auth.user_id, request.query)
    if cached:
        print(f"✓ Cache hit for query: {request.query[:50]}...")
        return QueryResponse(**cached)
    
    try:
        # Step 1: Embed query
        query_embedding = embed_texts([request.query], use_cache=True)[0]
        
        # Step 2: Document-level search
        vector_db = get_vector_db()
        doc_results = vector_db.search_documents(
            query_vector=query_embedding,
            tenant_id=auth.tenant_id,
            top_k=request.top_docs
        )
        
        if not doc_results:
            return QueryResponse(
                query=request.query,
                answer="No relevant documents found. Please upload documents first.",
                citations=[],
                latency_ms=int((time.time() - start_time) * 1000)
            )
        
        # Get doc_ids
        doc_ids = [doc["doc_id"] for doc in doc_results]
        
        # Step 3: Chunk-level search (within found documents)
        chunk_results = vector_db.search_chunks(
            query_vector=query_embedding,
            tenant_id=auth.tenant_id,
            doc_ids=doc_ids,
            top_k=request.top_docs * request.top_chunks_per_doc
        )
        
        if not chunk_results:
            return QueryResponse(
                query=request.query,
                answer="No relevant content found in documents.",
                citations=[],
                latency_ms=int((time.time() - start_time) * 1000)
            )
        
        # Step 4: Fetch full chunk texts from PostgreSQL
        chunk_ids = [chunk["chunk_id"] for chunk in chunk_results]
        chunks = db.query(Chunk).filter(
            Chunk.chunk_id.in_(chunk_ids),
            Chunk.is_deleted == False
        ).all()
        
        # Build candidates for reranking
        candidates = []
        for chunk in chunks:
            # Find matching result for score
            result = next((r for r in chunk_results if r["chunk_id"] == chunk.chunk_id), None)
            if result:
                candidates.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "page": chunk.page,
                    "text": chunk.text,  # Full text from PostgreSQL
                    "preview": chunk.preview,
                    "vector_score": result["score"]
                })
        
        # Step 5: Rerank with cross-encoder
        reranked = rerank_candidates(
            query=request.query,
            candidates=candidates,
            text_key="text",
            top_k=8,
            use_cache=True
        )
        
        # Step 6: Build context for LLM
        context_chunks = []
        for candidate, rerank_score in reranked:
            context_chunks.append({
                "doc_id": candidate["doc_id"],
                "page": candidate["page"],
                "text": candidate["text"]
            })
        
        # Step 7: Generate answer
        prompt = build_rag_prompt(request.query, context_chunks)
        answer = generate_answer(prompt, max_tokens=1024, temperature=0.7, stream=False)
        
        # Step 8: Build citations
        citations = []
        for candidate, rerank_score in reranked:
            citations.append(Citation(
                doc_id=candidate["doc_id"],
                chunk_id=candidate["chunk_id"],
                page=candidate["page"],
                score=rerank_score,
                preview=candidate["preview"]
            ))
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Build response
        response = QueryResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            latency_ms=latency_ms
        )
        
        # Cache answer
        cache_answer(
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            query=request.query,
            answer_data=response.model_dump()
        )
        
        # Store in query history
        history = QueryHistory(
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            query_text=request.query,
            answer=answer,
            latency_ms=latency_ms,
            doc_count=len(doc_ids),
            chunk_count=len(chunk_results)
        )
        db.add(history)
        db.commit()
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@app.post(f"{settings.API_V1_PREFIX}/query/stream")
async def query_documents_stream(
    request: QueryRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """
    Query documents with streaming response.
    Returns Server-Sent Events (SSE) stream.
    """
    async def generate_stream():
        """Generator for SSE stream."""
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching documents...'})}\n\n"
            
            # Step 1: Embed query
            query_embedding = embed_texts([request.query], use_cache=True)[0]
            
            # Step 2: Document search
            vector_db = get_vector_db()
            doc_results = vector_db.search_documents(
                query_vector=query_embedding,
                tenant_id=auth.tenant_id,
                top_k=request.top_docs
            )
            
            yield f"data: {json.dumps({'type': 'status', 'message': f'Found {len(doc_results)} documents'})}\n\n"
            
            if not doc_results:
                yield f"data: {json.dumps({'type': 'answer', 'content': 'No relevant documents found.'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Step 3: Chunk search
            doc_ids = [doc["doc_id"] for doc in doc_results]
            chunk_results = vector_db.search_chunks(
                query_vector=query_embedding,
                tenant_id=auth.tenant_id,
                doc_ids=doc_ids,
                top_k=request.top_docs * request.top_chunks_per_doc
            )
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Refining results...'})}\n\n"
            
            # Step 4: Fetch and rerank
            chunk_ids = [chunk["chunk_id"] for chunk in chunk_results]
            chunks = db.query(Chunk).filter(Chunk.chunk_id.in_(chunk_ids), Chunk.is_deleted == False).all()
            
            candidates = []
            for chunk in chunks:
                result = next((r for r in chunk_results if r["chunk_id"] == chunk.chunk_id), None)
                if result:
                    candidates.append({
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "page": chunk.page,
                        "text": chunk.text,
                        "preview": chunk.preview,
                        "vector_score": result["score"]
                    })
            
            reranked = rerank_candidates(request.query, candidates, "text", top_k=8, use_cache=True)
            
            # Step 5: Build context
            context_chunks = [{"doc_id": c["doc_id"], "page": c["page"], "text": c["text"]} 
                            for c, _ in reranked]
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer...'})}\n\n"
            
            # Step 6: Stream answer
            prompt = build_rag_prompt(request.query, context_chunks)
            answer_stream = generate_answer(prompt, stream=True)
            
            for token in answer_stream:
                yield f"data: {json.dumps({'type': 'answer', 'content': token})}\n\n"
            
            # Get document filenames for citations
            doc_ids_set = list(set([c["doc_id"] for c, _ in reranked]))
            docs = db.query(Document).filter(Document.doc_id.in_(doc_ids_set)).all()
            doc_filenames = {doc.doc_id: doc.filename for doc in docs}
            
            # Send citations with filenames
            citations = [
                {
                    "doc_id": c["doc_id"],
                    "filename": doc_filenames.get(c["doc_id"], "Unknown"),
                    "chunk_id": c["chunk_id"],
                    "page": c["page"],
                    "score": float(score),
                    "preview": c["preview"]
                }
                for c, score in reranked
            ]
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
            
            # Done
            yield "data: [DONE]\n\n"
        
        except Exception as e:
            print(f"❌ Stream error: {str(e)}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get(f"{settings.API_V1_PREFIX}/documents", response_model=List[DocumentInfo])
async def list_documents(
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 50
):
    """List all documents for the authenticated tenant."""
    documents = db.query(Document).filter(
        Document.tenant_id == auth.tenant_id,
        Document.is_deleted == False
    ).order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
    
    return [
        DocumentInfo(
            doc_id=doc.doc_id,
            filename=doc.filename,
            pages=doc.pages,
            status=doc.status,
            created_at=doc.created_at
        )
        for doc in documents
    ]


@app.delete(f"{settings.API_V1_PREFIX}/documents/all")
async def delete_all_documents(
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """
    Delete all documents for the authenticated tenant.
    Removes: DB records, Qdrant vectors, and Redis cache.
    """
    tenant_id = auth.tenant_id
    
    try:
        # 1. Get all documents for this tenant
        documents = db.query(Document).filter(
            Document.tenant_id == tenant_id,
            Document.is_deleted == False
        ).all()
        
        doc_ids = [doc.doc_id for doc in documents]
        
        if not doc_ids:
            return {"message": "No documents to delete", "deleted_count": 0}
        
        # 2. Delete from Qdrant
        vector_db = get_vector_db()
        for doc_id in doc_ids:
            try:
                vector_db.delete_document(doc_id)
                vector_db.delete_chunks_by_doc(doc_id)
            except Exception as e:
                print(f"Error deleting vectors for {doc_id}: {e}")
        
        # 3. Delete chunks from PostgreSQL
        db.query(Chunk).filter(
            Chunk.doc_id.in_(doc_ids)
        ).delete(synchronize_session=False)
        
        # 4. Delete documents from PostgreSQL
        db.query(Document).filter(
            Document.doc_id.in_(doc_ids)
        ).delete(synchronize_session=False)
        
        # 5. Clear cache for this tenant
        from app.services.cache_service import get_cache
        cache = get_cache()
        if cache.redis_client:
            # Clear all keys matching tenant pattern
            for key in cache.redis_client.scan_iter(f"*{tenant_id}*"):
                cache.redis_client.delete(key)
        
        db.commit()
        
        return {
            "message": f"Successfully deleted all documents",
            "deleted_count": len(doc_ids),
            "doc_ids": doc_ids
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting documents: {str(e)}"
        )


@app.delete(f"{settings.API_V1_PREFIX}/documents/{{doc_id}}")
async def delete_document(
    doc_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Soft delete a document."""
    document = db.query(Document).filter(
        Document.doc_id == doc_id,
        Document.tenant_id == auth.tenant_id
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Soft delete
    document.soft_delete()
    
    # Soft delete all chunks
    chunks = db.query(Chunk).filter(Chunk.doc_id == doc_id).all()
    for chunk in chunks:
        chunk.soft_delete()
    
    db.commit()
    
    return {"success": True, "message": f"Document {doc_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

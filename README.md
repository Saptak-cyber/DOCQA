# Document Q&A System

An advanced RAG (Retrieval-Augmented Generation) system that delivers accurate, context-aware answers from PDF documents using a sophisticated multi-stage pipeline.

## How It Works

**Ingestion Pipeline:**
1. PDFs are uploaded and stored in Cloudinary
2. Text is extracted page-by-page using pdfplumber
3. Content is chunked with overlap for context preservation
4. Each chunk is embedded using all-mpnet-base-v2 (768-dimensional vectors)
5. Two-level indexing: document-level vectors for coarse search, chunk-level vectors for precise retrieval
6. All data stored in PostgreSQL (text) and Qdrant (vectors) with multi-tenant isolation

**Query Pipeline:**
1. User query is embedded using the same model
2. **Document-level search**: Finds relevant PDFs from entire collection
3. **Chunk-level search**: Retrieves top passages within those documents
4. **Cross-encoder reranking**: Re-scores chunks using ms-marco-MiniLM for semantic precision
5. **LLM generation**: Ollama (phi3.5) generates natural answers from top-ranked context
6. **Streaming response**: Answer streams token-by-token with source citations

## What Makes It Different

Unlike simple keyword search or basic RAG systems, this implements:
- **2-level retrieval** (doc→chunk) instead of flat search, reducing noise
- **Cross-encoder reranking** for superior semantic matching vs. vector search alone
- **Multi-tenancy** with API key authentication and rate limiting
- **Redis caching** at multiple layers (embeddings, reranking, answers)
- **Streaming responses** for better UX vs. blocking generation
- **Full-text storage in PostgreSQL** alongside vectors, enabling hybrid approaches

This architecture balances accuracy, speed, and scalability for production-grade document Q&A.

---

## Quick Start

## Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Ollama (for local LLM)

## Setup

### 1. Start Infrastructure
```bash
docker-compose up -d
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your Cloudinary credentials
```

### 3. Initialize Database
```bash
python -c "from app.database import init_db; init_db()"
```

### 4. Create API Key
```bash
python create_api_key.py --tenant demo-tenant --name "Dev Key"
# Save the generated API key!
```

### 5. Start Ollama
```bash
ollama serve
ollama pull phi3.5
```

### 6. Run Backend
```bash
# Terminal 1: API Server
uvicorn app.main:app --reload --port 8000

# Terminal 2: Celery Worker (ingestion queue)
celery -A app.celery_app worker -Q ingestion --loglevel=info --pool=solo

# Terminal 3: Celery Worker (maintenance queue) - optional
celery -A app.celery_app worker -Q maintenance --loglevel=info --pool=solo
```

### 7. Frontend Setup
```bash
cd ../frontend
npm install
npm run dev
```

## Usage

### API Endpoints
- Upload: `POST /api/v1/upload` (multipart/form-data, requires Bearer token)
- Query: `POST /api/v1/query` (JSON, requires Bearer token)
- Stream: `POST /api/v1/query/stream` (SSE, requires Bearer token)
- List Docs: `GET /api/v1/documents` (requires Bearer token)
- Delete: `DELETE /api/v1/documents/{doc_id}` (requires Bearer token)

### Authentication
Include API key in Authorization header:
```
Authorization: Bearer your-api-key-here
```

## Services
- API: http://localhost:8000
- Frontend: http://localhost:3000
- Qdrant: http://localhost:6333
- PostgreSQL: localhost:5432
- Redis: localhost:6379

## Architecture Improvements
✅ 2-level retrieval (doc + chunk, no sentence overhead)
✅ Chunk text in PostgreSQL (not Cloudinary)
✅ Multi-tenancy with API key auth
✅ Enhanced caching (embeddings, rerank, answers)
✅ Rate limiting per tenant
✅ Streaming responses
✅ Soft deletes
✅ Multi-queue Celery (ingestion/maintenance)

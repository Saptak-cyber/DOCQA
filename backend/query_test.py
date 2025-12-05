"""
Test script to query documents using the RAG pipeline.
This simulates what the API endpoint does.
"""

from app.database import SessionLocal
from app.models import Document, Chunk
from app.services.embedding_service import get_embedding_service
from app.services.vector_db_service import get_vector_db
from app.services.rerank_service import rerank_candidates
from app.services.llm_service import generate_answer, build_rag_prompt

def query_documents(query: str, tenant_id: str, top_docs: int = 5, top_chunks: int = 8):
    """
    Query documents and get an answer.
    
    Args:
        query: The question to ask
        tenant_id: Your tenant ID (from API key)
        top_docs: Number of documents to search
        top_chunks: Number of chunks to use for context
    """
    print(f"\n🔍 Query: {query}\n")
    
    # Step 1: Embed the query
    print("1️⃣ Generating query embedding...")
    embed_service = get_embedding_service()
    query_embedding = embed_service.embed_texts([query])[0]
    print(f"   ✓ Embedding dimension: {len(query_embedding)}")
    
    # Step 2: Search documents in Qdrant
    print("\n2️⃣ Searching documents...")
    vector_db = get_vector_db()
    doc_results = vector_db.search_documents(
        query_vector=query_embedding,
        tenant_id=tenant_id,
        top_k=top_docs
    )
    print(f"   ✓ Found {len(doc_results)} relevant documents")
    
    if not doc_results:
        print("\n❌ No documents found!")
        return
    
    # Step 3: Search chunks within those documents
    print("\n3️⃣ Searching chunks...")
    doc_ids = [doc["doc_id"] for doc in doc_results]
    chunk_results = vector_db.search_chunks(
        query_vector=query_embedding,
        tenant_id=tenant_id,
        doc_ids=doc_ids,
        top_k=top_docs * 4
    )
    print(f"   ✓ Found {len(chunk_results)} relevant chunks")
    
    # Step 4: Fetch full chunk text from PostgreSQL
    print("\n4️⃣ Fetching chunk text from database...")
    db = SessionLocal()
    chunk_ids = [chunk["chunk_id"] for chunk in chunk_results]
    
    # Debug: Check types and values
    print(f"   DEBUG: Qdrant chunk_ids sample: {chunk_ids[:2]}")
    print(f"   DEBUG: Qdrant chunk_id types: {[type(cid).__name__ for cid in chunk_ids[:2]]}")
    
    # Get all chunks from DB to compare
    all_chunks = db.query(Chunk).filter(Chunk.is_deleted == False).all()
    print(f"   DEBUG: PostgreSQL has {len(all_chunks)} chunks")
    if all_chunks:
        print(f"   DEBUG: PostgreSQL chunk_id sample: {all_chunks[0].chunk_id}")
        print(f"   DEBUG: PostgreSQL chunk_id type: {type(all_chunks[0].chunk_id).__name__}")
    
    chunks = db.query(Chunk).filter(
        Chunk.chunk_id.in_(chunk_ids),
        Chunk.is_deleted == False
    ).all()
    
    print(f"   DEBUG: Query matched {len(chunks)} chunks")
    
    # Build candidates for reranking
    candidates = []
    for chunk in chunks:
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
    
    print(f"   ✓ Retrieved {len(candidates)} chunks with text")
    
    # Step 5: Rerank with cross-encoder
    print("\n5️⃣ Reranking chunks...")
    reranked = rerank_candidates(
        query=query,
        candidates=candidates,
        text_key="text",
        top_k=top_chunks,
        use_cache=True
    )
    print(f"   ✓ Top {len(reranked)} chunks after reranking")
    
    # Step 6: Build context for LLM
    print("\n6️⃣ Building context...")
    context_chunks = []
    for candidate, rerank_score in reranked:
        context_chunks.append({
            "doc_id": candidate["doc_id"],
            "page": candidate["page"],
            "text": candidate["text"]
        })
        print(f"   • Page {candidate['page']}: {candidate['text'][:80]}... (score: {rerank_score:.3f})")
    
    # Step 7: Generate answer
    print("\n7️⃣ Generating answer from LLM...")
    prompt = build_rag_prompt(query, context_chunks)
    answer = generate_answer(prompt, max_tokens=1024, temperature=0.7, stream=False)
    
    # Print results
    print("\n" + "="*80)
    print("📝 ANSWER:")
    print("="*80)
    print(answer)
    print("="*80)
    
    # Print citations
    print("\n📚 SOURCES:")
    for i, (candidate, score) in enumerate(reranked, 1):
        print(f"\n[{i}] Doc: {candidate['doc_id'][:12]}... | Page: {candidate['page']} | Score: {score:.3f}")
        print(f"    Preview: {candidate['preview'][:150]}...")
    
    db.close()
    return answer


if __name__ == "__main__":
    # Your tenant_id - get this from your API key record in the database
    # Or use the one from your .env file
    TENANT_ID = "demo-tenant"  # Updated with your actual tenant_id
    
    # Example queries
        # "What is this document about?",
        # "What are the main technical requirements?",
        # "Explain the key features mentioned in the document",
    queries = [
        "What are the requirements in this document?"
    ]
    
    # Ask a question
    query = queries[0]  # Change index to try different queries
    # Or use a custom query:
    # query = "Your custom question here"
    
    answer = query_documents(
        query=query,
        tenant_id=TENANT_ID,
        top_docs=5,
        top_chunks=8
    )

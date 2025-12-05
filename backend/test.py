from app.database import SessionLocal
from app.models import Chunk

db = SessionLocal()
chunks = db.query(Chunk).filter(Chunk.is_deleted == False).all()

for chunk in chunks:
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Page: {chunk.page}")
    print(f"Text: {chunk.text}")  # Full text here
    print("-" * 50)
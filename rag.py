# rag.py
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

# Local, fast sentence transformer embedding function
EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks respecting sentence boundaries."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Don't cut a word in half if we are not at the very end
        if end < text_length:
            last_space = chunk.rfind(" ")
            if last_space != -1:
                chunk = chunk[:last_space]
                end = start + last_space
                
        cleaned = chunk.strip()
        if len(cleaned) > 40:  # Skip tiny fragments
            chunks.append(cleaned)
            
        start = end - overlap
        
    return chunks

def build_vector_store(paper_id: str, text: str, abstract: str = "") -> Any:
    """
    Creates an in-memory ephemeral Chroma collection and indexes paper chunks.
    """
    # In-memory client ensures clean isolation per run with zero disk overhead
    client = chromadb.EphemeralClient()
    collection_name = f"paper_{paper_id.replace('.', '_').replace('/', '_')}"
    
    # Reset if re-testing
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
        
    collection = client.create_collection(
        name=collection_name,
        embedding_function=EMBEDDING_FUNCTION,
        metadata={"hnsw:space": "cosine"}
    )
    
    chunks = chunk_text(text)
    
    # If fallback mode was used or chunking returned very little, include the abstract
    if abstract and (not chunks or len(chunks) < 2):
        chunks.append(f"Abstract: {abstract}")
        
    if not chunks:
        chunks = [abstract if abstract else "No content available."]
        
    documents = chunks
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_id": i, "paper_id": paper_id} for i in range(len(chunks))]
    
    # Ingest into Chroma in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i + batch_size],
            ids=ids[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
        
    return collection

def retrieve_relevant_chunks(collection: Any, query: str, top_k: int = 4) -> List[str]:
    """Retrieve top-k relevant text chunks for a question."""
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count())
    )
    if results and "documents" in results and results["documents"]:
        return results["documents"][0]
    return []
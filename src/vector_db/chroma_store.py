import chromadb
from chromadb.config import Settings
from typing import List
import os

CHROMA_PATH = "vector_db/chroma_db"



def get_chroma_collection(collection_name: str = "documents"):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def reset_chroma_collection(collection_name: str = "documents"):
    """
    Wipe the existing collection and create a fresh one.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    try:
        client.delete_collection(collection_name)
        print(f"  ChromaDB: deleted old collection '{collection_name}'")
    except ValueError:
        # ده الخطأ الوحيد المسموح نتجاهله (معناه إنها لسه متكريتتش فعلاً)
        print(f"  ChromaDB: collection '{collection_name}' did not exist, skipping delete.")
    
    # نستخدم get_or_create كنوع من الأمان الإضافي (Robustness) في الـ Production
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"  ChromaDB: active collection '{collection_name}' is ready")
    return collection


def add_to_chroma(collection, chunks: List[str],
                  embeddings, metadata: List[dict]):
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadata
    )
    print(f"  ChromaDB: added {len(chunks)} chunks")


def search_chroma(collection, query_vector,
                  top_k: int = 3) -> List[dict]:
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=top_k
    )
    output = []
    for i in range(len(results["documents"][0])):
        output.append({
            "text":   results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "score":  1 - results["distances"][0][i]  # convert distance→similarity
        })
    return output
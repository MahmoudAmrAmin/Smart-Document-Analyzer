import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

from src.config.settings import TOP_K, VECTOR_STORE, MIN_SCORE
from src.embeddings.embedder import embed_query
from src.vector_db.faiss_store import load_faiss_index, search_faiss
from src.vector_db.chroma_store import get_chroma_collection, search_chroma
from typing import List

# Separate caches per store — both can be loaded simultaneously
_cache = {
    "faiss_index": None,
    "faiss_meta":  None,
    "chroma_col":  None
}


def reset_cache():
    """Force reload of all vector stores on next query."""
    _cache["faiss_index"] = None
    _cache["faiss_meta"]  = None
    _cache["chroma_col"]  = None
    print("  Retriever cache cleared.")


def _get_faiss():
    if _cache["faiss_index"] is None:
        print("  Loading FAISS index...")
        _cache["faiss_index"], _cache["faiss_meta"] = load_faiss_index()
    return _cache["faiss_index"], _cache["faiss_meta"]


def _get_chroma():
    if _cache["chroma_col"] is None:
        print("  Loading ChromaDB collection...")
        _cache["chroma_col"] = get_chroma_collection()
    return _cache["chroma_col"]


def _keyword_fallback(query: str,
                      metadata: List[dict],
                      top_k: int) -> List[dict]:
    """Keyword overlap fallback when semantic scores are too low."""
    query_words = set(query.lower().split())
    scored      = []

    for item in metadata:
        text = item.get("text", "").lower()
        hits = sum(1 for w in query_words if w in text)
        if hits > 0:
            scored.append({
                "text":   item["text"],
                "source": item["source"],
                "score":  round(hits / len(query_words), 4)
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def retrieve(query: str,
             top_k: int = TOP_K,
             store: str = VECTOR_STORE,
             min_score: float = MIN_SCORE) -> List[dict]:
    """
    Retrieve top-k relevant chunks for a query.
    store parameter is respected — faiss and chroma are
    independently cached and both work correctly.
    """
    query_vec = embed_query(query)
    print(f"  Retrieving from: {store.upper()} (top_k={top_k})")

    # ── Semantic search ────────────────────────────────
    if store == "faiss":
        index, metadata = _get_faiss()
        results = search_faiss(index, metadata, query_vec, top_k * 2)

    elif store == "chroma":
        collection = _get_chroma()
        results    = search_chroma(collection, query_vec, top_k * 2)

    else:
        raise ValueError(f"Unknown store: '{store}'. Use 'faiss' or 'chroma'.")

    # ── Filter low-score chunks ────────────────────────
    filtered = [r for r in results if r["score"] >= min_score]

    # ── Keyword fallback ───────────────────────────────
    if not filtered:
        print(f"  ⚠️  Low scores — using keyword fallback")
        if store == "faiss":
            _, metadata = _get_faiss()
            fallback_meta = metadata
        else:
            col      = _get_chroma()
            all_docs = col.get(include=["documents", "metadatas"])
            fallback_meta = [
                {"text": doc, "source": meta.get("source", "")}
                for doc, meta in zip(
                    all_docs["documents"],
                    all_docs["metadatas"]
                )
            ]
        filtered = _keyword_fallback(query, fallback_meta, top_k)

    return filtered[:top_k]
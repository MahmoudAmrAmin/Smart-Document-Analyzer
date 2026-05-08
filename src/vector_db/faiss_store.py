import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple

FAISS_INDEX_PATH = "vector_db/faiss_index/index.faiss"
FAISS_META_PATH  = "vector_db/faiss_index/metadata.pkl"


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index using Inner Product (= cosine sim when normalized).
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built — {index.ntotal} vectors, dim={dim}")
    return index


def save_faiss_index(index: faiss.IndexFlatIP, metadata: List[dict]):
    """
    Save FAISS index + metadata (chunk text + source filename) to disk.
    """
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved FAISS index → {FAISS_INDEX_PATH}")


def load_faiss_index() -> Tuple[faiss.IndexFlatIP, List[dict]]:
    """
    Load FAISS index + metadata from disk.
    """
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "rb") as f:
        metadata = pickle.load(f)
    print(f"Loaded FAISS index — {index.ntotal} vectors")
    return index, metadata


def search_faiss(index: faiss.IndexFlatIP,
                 metadata: List[dict],
                 query_vector: np.ndarray,
                 top_k: int = 3) -> List[dict]:
    """
    Retrieve top-k most similar chunks for a query vector.

    Returns:
        List of dicts: {text, source, score}
    """
    scores, indices = index.search(query_vector, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text":   metadata[idx]["text"],
            "source": metadata[idx]["source"],
            "score":  float(score)
        })
    return results
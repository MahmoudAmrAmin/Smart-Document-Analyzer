from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Load once at module level — avoids reloading on every call
MODEL_NAME = "all-MiniLM-L6-v2"   # fast, good quality, 384-dim vectors
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_chunks(chunks: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Convert a list of text chunks into embedding vectors.

    Args:
        chunks:     list of text strings
        batch_size: how many chunks to encode at once

    Returns:
        numpy array of shape (len(chunks), 384)
    """
    model = get_model()
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True   # cosine similarity works directly
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user query for retrieval.

    Returns:
        numpy array of shape (1, 384)
    """
    model = get_model()
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return vector
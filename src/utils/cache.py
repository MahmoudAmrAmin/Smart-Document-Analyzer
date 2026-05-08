"""
Response cache for RAG queries.
Stores previous answers and returns them instantly
for the same or very similar questions.
Uses semantic similarity to detect near-duplicate questions.
"""

import os
import sys
import json
import hashlib
import time
from typing import Optional

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

CACHE_PATH = os.path.join(ROOT_DIR, "vector_db", "cache.json")
SIMILARITY_THRESHOLD = 0.92   # how similar questions must be to use cache


class ResponseCache:
    """
    Two-level cache:
    1. Exact match  → instant (hash lookup)
    2. Semantic match → near-duplicate detection
    """

    def __init__(self):
        self._cache   = {}   # {hash: {question, answer, metadata, ts}}
        self._vectors = {}   # {hash: embedding vector}
        self._load()

    def _load(self):
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH) as f:
                    self._cache = json.load(f)
                print(f"  Cache loaded: {len(self._cache)} entries")
            except Exception:
                self._cache = {}

    def _save(self):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(self._cache, f, indent=2)

    def _hash(self, question: str) -> str:
        return hashlib.md5(question.strip().lower().encode()).hexdigest()

    def _embed(self, text: str):
        from src.embeddings.embedder import embed_query
        return embed_query(text)[0]

    def _cosine_sim(self, a, b) -> float:
        import numpy as np
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def get(self, question: str) -> Optional[dict]:
        """
        Look up a question in cache.
        Returns cached result if exact or semantic match found.
        """
        key = self._hash(question)

        # Level 1 — exact match
        if key in self._cache:
            entry = self._cache[key]
            print(f"  ✅ Cache HIT (exact): {question[:50]}")
            entry["from_cache"] = True
            entry["latency_s"]  = 0.0
            return entry

        # Level 2 — semantic match
        try:
            q_vec = self._embed(question)
            for k, entry in self._cache.items():
                if k not in self._vectors:
                    continue
                sim = self._cosine_sim(q_vec, self._vectors[k])
                if sim >= SIMILARITY_THRESHOLD:
                    print(f"  ✅ Cache HIT (semantic, sim={sim:.3f}): "
                          f"{question[:50]}")
                    result              = entry.copy()
                    result["from_cache"] = True
                    result["latency_s"]  = 0.0
                    result["cache_sim"]  = round(sim, 3)
                    return result
        except Exception:
            pass  # embedding failed — skip semantic check

        return None

    def set(self, question: str, result: dict):
        """Store a new result in cache."""
        key = self._hash(question)
        entry = {
            "question": question,
            "answer":   result.get("answer", ""),
            "doc_type": result.get("doc_type"),
            "sources":  result.get("sources", []),
            "latency_s": result.get("latency_s", 0),
            "timestamp": time.time()
        }
        self._cache[key] = entry

        # Store embedding for semantic matching
        try:
            self._vectors[key] = self._embed(question).tolist()
        except Exception:
            pass

        self._save()
        print(f"  💾 Cached: {question[:50]}")

    def clear(self):
        """Wipe entire cache — call after re-indexing."""
        self._cache   = {}
        self._vectors = {}
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
        print("  Cache cleared.")

    @property
    def size(self) -> int:
        return len(self._cache)


# Module-level singleton
_cache_instance = None

def get_cache() -> ResponseCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ResponseCache()
    return _cache_instance

def clear_cache():
    global _cache_instance
    if _cache_instance:
        _cache_instance.clear()
    _cache_instance = None
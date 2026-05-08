"""
Answer quality evaluator.
Computes metrics shown in the UI after every answer:
- Retrieval relevance: how well chunks match the question
- Answer coverage: how much of the answer is grounded in chunks
- Confidence score: combined quality signal
"""

import os
import sys
import re

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

from typing import List
import numpy as np


def _cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def compute_retrieval_relevance(question: str,
                                chunks: List[dict]) -> float:
    """
    Average semantic similarity between question and retrieved chunks.
    Score 0-1: higher = chunks are more relevant to the question.
    """
    if not chunks:
        return 0.0

    try:
        from src.embeddings.embedder import embed_query, embed_chunks
        q_vec      = embed_query(question)[0]
        chunk_texts = [c["text"] for c in chunks]
        c_vecs     = embed_chunks(chunk_texts)
        sims       = [_cosine_sim(q_vec, cv) for cv in c_vecs]
        return round(float(np.mean(sims)), 4)
    except Exception:
        # Fallback: use retrieval scores already in chunks
        scores = [c.get("score", 0) for c in chunks]
        return round(float(np.mean(scores)), 4) if scores else 0.0


def compute_answer_grounding(answer: str,
                              chunks: List[dict]) -> float:
    """
    Estimates how much of the answer is grounded in the chunks.
    Uses keyword overlap as a proxy for grounding.
    Score 0-1: higher = answer is more grounded in retrieved text.
    """
    if not answer or not chunks:
        return 0.0

    # Not-found response
    if "could not find" in answer.lower():
        return 0.0

    # Combine all chunk text
    context = " ".join(c["text"] for c in chunks).lower()

    # Extract meaningful words from answer (ignore short words)
    answer_words = set(
        w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', answer)
    )

    if not answer_words:
        return 0.0

    grounded = sum(1 for w in answer_words if w in context)
    score    = grounded / len(answer_words)
    return round(min(score, 1.0), 4)


def compute_confidence(retrieval_relevance: float,
                       answer_grounding: float,
                       top_chunk_score: float) -> float:
    """
    Combined confidence score (0-1).
    Weighted combination of all three signals.
    """
    confidence = (
        0.40 * retrieval_relevance +
        0.35 * answer_grounding    +
        0.25 * top_chunk_score
    )
    return round(min(confidence, 1.0), 4)


def evaluate_answer(question: str,
                    answer: str,
                    chunks: List[dict]) -> dict:
    """
    Full evaluation of a RAG answer.

    Returns:
        {
            "retrieval_relevance": float,  # 0-1
            "answer_grounding":    float,  # 0-1
            "top_chunk_score":     float,  # 0-1
            "confidence":          float,  # 0-1
            "quality_label":       str     # Excellent/Good/Fair/Poor
        }
    """
    top_score  = max((c.get("score", 0) for c in chunks), default=0.0)
    relevance  = compute_retrieval_relevance(question, chunks)
    grounding  = compute_answer_grounding(answer, chunks)
    confidence = compute_confidence(relevance, grounding, top_score)

    # Quality label
    if confidence >= 0.75:
        label = "🟢 Excellent"
    elif confidence >= 0.55:
        label = "🟡 Good"
    elif confidence >= 0.35:
        label = "🟠 Fair"
    else:
        label = "🔴 Poor"

    return {
        "retrieval_relevance": relevance,
        "answer_grounding":    grounding,
        "top_chunk_score":     round(top_score, 4),
        "confidence":          confidence,
        "quality_label":       label
    }
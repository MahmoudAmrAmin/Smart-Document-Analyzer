import os
import sys
import json
import time

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

from src.rag.retriever import retrieve, reset_cache
from src.rag.prompt_builder import build_prompt
from src.rag.generator import generate_answer, stream_answer
from src.rag.query_expander import expand_query
from src.utils.cache import get_cache
from src.utils.evaluator import evaluate_answer
from src.config.settings import TOP_K, VECTOR_STORE
from collections import defaultdict

DOC_TYPES_PATH = os.path.join(
    ROOT_DIR, "vector_db", "faiss_index", "doc_types.json"
)


def reset_retriever_cache():
    reset_cache()


def load_doc_types() -> dict:
    if os.path.exists(DOC_TYPES_PATH):
        with open(DOC_TYPES_PATH) as f:
            return json.load(f)
    return {}


def get_doc_type_for_query(chunks: list, doc_types: dict) -> tuple:
    if not chunks:
        return None, 0.0

    weighted_votes   = defaultdict(float)
    label_confidence = {}

    for chunk in chunks:
        source     = chunk.get("source", "")
        ret_score  = chunk.get("score", 1.0)
        dtype_info = doc_types.get(source, {})
        label      = dtype_info.get("label", "unknown")
        conf       = dtype_info.get("confidence", 0.0)

        if label == "unknown":
            continue

        weighted_votes[label]   += ret_score * conf
        label_confidence[label]  = conf

    if not weighted_votes:
        return "unknown", 0.0

    best_label = max(weighted_votes, key=weighted_votes.get)
    return best_label, label_confidence.get(best_label, 0.0)


def ask(question: str,
        doc_type: str = None,
        top_k: int = TOP_K,
        store: str = VECTOR_STORE,
        use_cache: bool = True,
        use_expansion: bool = True) -> dict:
    """
    Full RAG pipeline with caching, query expansion, and evaluation.

    Returns:
        {
            "question":            str,
            "answer":              str,
            "doc_type":            str,
            "confidence":          float,
            "sources":             list,
            "chunks":              list,
            "latency_s":           float,
            "from_cache":          bool,
            "eval":                dict,
            "timing":              dict,
            "expanded_queries":    list
        }
    """
    start  = time.time()
    timing = {}

    # ── 1. Cache check ─────────────────────────────────
    if use_cache:
        cache  = get_cache()
        cached = cache.get(question)
        if cached:
            cached["from_cache"] = True
            cached["latency_s"]  = 0.0
            cached["timing"]     = {"cache_hit": True}
            cached["eval"]       = cached.get("eval", {})
            return cached

    # ── 2. Query expansion ─────────────────────────────
    t0 = time.time()
    if use_expansion:
        expanded = expand_query(question, use_llm=True)
    else:
        expanded = [question]
    timing["expansion_s"] = round(time.time() - t0, 3)

    # ── 3. Retrieve — merge results from all query variants
    t0 = time.time()
    seen_texts = set()
    all_chunks = []

    for variant in expanded:
        results = retrieve(variant, top_k=top_k, store=store)
        for r in results:
            key = r["text"][:100]
            if key not in seen_texts:
                seen_texts.add(key)
                all_chunks.append(r)

    # Re-sort by score and keep top_k
    all_chunks = sorted(
        all_chunks, key=lambda x: x.get("score", 0), reverse=True
    )[:top_k]
    timing["retrieval_s"] = round(time.time() - t0, 3)

    if not all_chunks:
        return {
            "question":         question,
            "answer":           "No relevant content found in the documents.",
            "doc_type":         None,
            "confidence":       0.0,
            "sources":          [],
            "chunks":           [],
            "latency_s":        round(time.time() - start, 2),
            "from_cache":       False,
            "eval":             {},
            "timing":           timing,
            "expanded_queries": expanded
        }

    # ── 4. Resolve doc_type ────────────────────────────
    stored_doc_types      = load_doc_types()
    doc_type, ml_conf     = get_doc_type_for_query(
        all_chunks, stored_doc_types
    )

    # ── 5. Build prompt ────────────────────────────────
    prompt = build_prompt(question, all_chunks, doc_type=None)

    # ── 6. Generate answer ─────────────────────────────
    t0     = time.time()
    answer = generate_answer(prompt)
    timing["generation_s"] = round(time.time() - t0, 3)

    # ── 7. Evaluate answer quality ─────────────────────
    t0   = time.time()
    eval_metrics = evaluate_answer(question, answer, all_chunks)
    timing["eval_s"] = round(time.time() - t0, 3)

    timing["total_s"] = round(time.time() - start, 2)

    result = {
        "question":         question,
        "answer":           answer,
        "doc_type":         doc_type,
        "confidence":       ml_conf,
        "sources":          [{"source": c["source"],
                              "score":  c["score"]}
                             for c in all_chunks],
        "chunks":           all_chunks,
        "latency_s":        timing["total_s"],
        "from_cache":       False,
        "eval":             eval_metrics,
        "timing":           timing,
        "expanded_queries": expanded
    }

    # ── 8. Store in cache ──────────────────────────────
    if use_cache:
        cache.set(question, result)

    return result


def ask_stream(question: str,
               doc_type: str = None,
               top_k: int = TOP_K,
               store: str = VECTOR_STORE,
               use_cache: bool = True,
               use_expansion: bool = True):
    """
    Streaming version of ask().
    Returns (metadata_dict, token_generator) tuple.
    metadata contains everything except the answer.
    Use token_generator with st.write_stream().
    """
    start  = time.time()
    timing = {}

    # Cache check
    if use_cache:
        cache  = get_cache()
        cached = cache.get(question)
        if cached:
            # Return cached answer as a generator
            def _cached_gen():
                yield cached["answer"]
            cached["from_cache"] = True
            cached["latency_s"]  = 0.0
            return cached, _cached_gen()

    # Query expansion
    t0       = time.time()
    expanded = expand_query(question, use_llm=False) \
               if not use_expansion else \
               expand_query(question, use_llm=True)
    timing["expansion_s"] = round(time.time() - t0, 3)

    # Retrieve
    t0         = time.time()
    seen_texts = set()
    all_chunks = []

    for variant in expanded:
        results = retrieve(variant, top_k=top_k, store=store)
        for r in results:
            key = r["text"][:100]
            if key not in seen_texts:
                seen_texts.add(key)
                all_chunks.append(r)

    all_chunks = sorted(
        all_chunks, key=lambda x: x.get("score", 0), reverse=True
    )[:top_k]
    timing["retrieval_s"] = round(time.time() - t0, 3)

    if not all_chunks:
        def _empty_gen():
            yield "No relevant content found in the documents."
        return {"question": question, "chunks": [], "sources": [],
                "doc_type": None, "eval": {}, "from_cache": False,
                "timing": timing, "expanded_queries": expanded}, _empty_gen()

    # Resolve doc_type
    stored_doc_types  = load_doc_types()
    doc_type, ml_conf = get_doc_type_for_query(all_chunks, stored_doc_types)

    # Build prompt
    prompt = build_prompt(question, all_chunks, doc_type=None)

    # Metadata returned immediately (before generation starts)
    metadata = {
        "question":         question,
        "doc_type":         doc_type,
        "confidence":       ml_conf,
        "sources":          [{"source": c["source"],
                              "score":  c["score"]}
                             for c in all_chunks],
        "chunks":           all_chunks,
        "from_cache":       False,
        "timing":           timing,
        "expanded_queries": expanded,
        "eval":             {}   # filled after streaming completes
    }

    # Return metadata + streaming generator
    return metadata, stream_answer(prompt)
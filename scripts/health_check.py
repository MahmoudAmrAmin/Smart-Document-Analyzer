"""
Smart Document Analyzer — System Health Check
Run this before demo or deployment to verify everything is working.

Usage: python scripts/health_check.py
"""

import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

PASS  = "✅"
FAIL  = "❌"
WARN  = "⚠️ "
SEP   = "─" * 55


def check(label: str, fn) -> bool:
    try:
        result = fn()
        status = PASS if result else WARN
        print(f"  {status}  {label}")
        return bool(result)
    except Exception as e:
        print(f"  {FAIL}  {label}  →  {e}")
        return False


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def main():
    print("\n" + "=" * 55)
    print("  🧠 Smart Document Analyzer — Health Check")
    print("=" * 55)

    results = []

    # ── Environment ────────────────────────────────────
    section("1. Environment")

    results.append(check(
        "Python 3.10+",
        lambda: sys.version_info >= (3, 10)
    ))
    results.append(check(
        "GROQ_API_KEY set",
        lambda: bool(os.getenv("GROQ_API_KEY", ""))
    ))
    results.append(check(
        ".env file exists",
        lambda: os.path.exists(os.path.join(ROOT_DIR, ".env"))
    ))

    # ── Dependencies ───────────────────────────────────
    section("2. Core Dependencies")

    deps = [
        ("streamlit",            "streamlit"),
        ("faiss-cpu",            "faiss"),
        ("chromadb",             "chromadb"),
        ("sentence-transformers","sentence_transformers"),
        ("pymupdf",              "fitz"),
        ("scikit-learn",         "sklearn"),
        ("groq",                 "groq"),
        ("pandas",               "pandas"),
        ("numpy",                "numpy"),
        ("python-dotenv",        "dotenv"),
    ]

    for display, mod in deps:
        results.append(check(
            display,
            lambda m=mod: __import__(m) is not None
        ))

    # ── Project modules ────────────────────────────────
    section("3. Project Modules")

    modules = [
        "src.config.settings",
        "src.ingestion.pdf_loader",
        "src.preprocessing.text_cleaner",
        "src.chunking.text_chunker",
        "src.embeddings.embedder",
        "src.dl.layout_detector",
        "src.dl.layout_chunker",
        "src.ml.predictor",
        "src.rag.rag_pipeline",
        "src.rag.query_expander",
        "src.utils.cache",
        "src.utils.evaluator",
        "src.vector_db.faiss_store",
        "src.vector_db.chroma_store",
        "src.pipeline.full_pipeline",
    ]

    for mod in modules:
        results.append(check(
            mod,
            lambda m=mod: __import__(m) is not None
        ))

    # ── Files + folders ────────────────────────────────
    section("4. Files & Folders")

    from src.config.settings import (
        ROOT_DIR, DOCUMENTS_FOLDER, FAISS_INDEX_PATH,
        ML_MODEL_PATH, CHROMA_PATH
    )

    path_checks = [
        ("data/raw/documents/ folder",    DOCUMENTS_FOLDER),
        ("vector_db/faiss_index/ folder", os.path.dirname(FAISS_INDEX_PATH)),
        ("ML model (doc_classifier.pkl)", ML_MODEL_PATH),
        ("FAISS index",                   FAISS_INDEX_PATH),
    ]

    for label, path in path_checks:
        results.append(check(label, lambda p=path: os.path.exists(p)))

    # ── Embedding model ────────────────────────────────
    section("5. Embedding Model")

    def test_embedding():
        from src.embeddings.embedder import embed_query
        vec = embed_query("test")
        return vec.shape == (1, 384)

    results.append(check("Load + embed a test query", test_embedding))

    # ── ML classifier ──────────────────────────────────
    section("6. ML Classifier")

    def test_classifier():
        from src.ml.predictor import predict_document_type
        result = predict_document_type(
            "This agreement is entered into between the parties."
        )
        return result["label"] in ["contract", "invoice", "research"]

    results.append(check("Predict document type", test_classifier))

    # ── Vector retrieval ───────────────────────────────
    section("7. Vector Retrieval")

    def test_faiss():
        from src.rag.retriever import retrieve
        results = retrieve("test query", top_k=3, store="faiss")
        return isinstance(results, list)

    results.append(check("FAISS retrieval", test_faiss))

    def test_chroma():
        from src.rag.retriever import retrieve
        results = retrieve("test query", top_k=3, store="chroma")
        return isinstance(results, list)

    results.append(check("ChromaDB retrieval", test_chroma))

    # ── RAG pipeline ───────────────────────────────────
    section("8. RAG Pipeline")

    def test_rag():
        from src.rag.rag_pipeline import ask
        result = ask(
            "What is this document about?",
            top_k=3,
            use_cache=False,
            use_expansion=False
        )
        return bool(result.get("answer"))

    results.append(check("Full ask() call", test_rag))

    def test_cache():
        from src.utils.cache import get_cache
        cache = get_cache()
        return cache is not None

    results.append(check("Response cache", test_cache))

    # ── Summary ────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    pct    = passed / total * 100

    print(f"\n{'=' * 55}")
    print(f"  Results: {passed}/{total} checks passed  ({pct:.0f}%)")

    if passed == total:
        print(f"  {PASS} System is fully operational!")
        print(f"\n  Run the app:")
        print(f"    streamlit run \"front end/app.py\"")
    elif passed >= total * 0.8:
        print(f"  {WARN}  Most checks passed — minor issues above.")
    else:
        print(f"  {FAIL}  Several checks failed — review errors above.")

    print("=" * 55 + "\n")
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

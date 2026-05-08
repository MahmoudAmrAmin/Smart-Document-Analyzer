import os
import sys
import json

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

from src.ingestion.pdf_loader import load_documents_from_folder
from src.preprocessing.text_cleaner import clean_documents
from src.embeddings.embedder import embed_chunks
from src.vector_db.faiss_store import build_faiss_index, save_faiss_index
from src.vector_db.chroma_store import reset_chroma_collection, add_to_chroma
from src.dl.layout_detector import detect_layouts_batch
from src.dl.layout_chunker import chunk_documents_by_layout
from src.config.settings import (DOCUMENTS_FOLDER, ROOT_DIR,
                                  FAISS_INDEX_PATH, FAISS_META_PATH,
                                  CHUNK_SIZE)

DOC_TYPES_PATH    = os.path.join(ROOT_DIR, "vector_db",
                                  "faiss_index", "doc_types.json")
LAYOUTS_SAVE_PATH = os.path.join(ROOT_DIR, "vector_db",
                                  "faiss_index", "layouts.json")


def clear_old_index():
    for path in [FAISS_INDEX_PATH, FAISS_META_PATH,
                 DOC_TYPES_PATH, LAYOUTS_SAVE_PATH]:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Cleared: {os.path.basename(path)}")


def _try_classify(text: str) -> dict:
    try:
        from src.ml.predictor import predict_document_type
        return predict_document_type(text)
    except FileNotFoundError:
        return {"label": "unknown", "confidence": 0.0, "all_scores": {}}
    except Exception as e:
        print(f"  ⚠️  Classification error: {e}")
        return {"label": "unknown", "confidence": 0.0, "all_scores": {}}


def run_indexing_pipeline(
    documents_folder: str = DOCUMENTS_FOLDER
) -> tuple:

    # ── 0. Clear old index ─────────────────────────────
    print("\n[0/6] Clearing old index...")
    clear_old_index()
    chroma_col = reset_chroma_collection()

    # ── 1. Load PDFs ───────────────────────────────────
    print("\n[1/6] Loading PDFs...")
    raw_docs = load_documents_from_folder(documents_folder)
    if not raw_docs:
        raise ValueError(f"No PDFs found in: {documents_folder}")

    # ── 2. Clean ───────────────────────────────────────
    print("\n[2/6] Cleaning text...")
    clean_docs = clean_documents(raw_docs)

    # ── 3. Classify ────────────────────────────────────
    print("\n[3/6] Classifying documents...")
    doc_types = {}
    for filename, text in clean_docs.items():
        result            = _try_classify(text)
        doc_types[filename] = result
        print(f"  {filename} → {result['label']} "
              f"({result['confidence']:.1%})")

    os.makedirs(os.path.dirname(DOC_TYPES_PATH), exist_ok=True)
    with open(DOC_TYPES_PATH, "w") as f:
        json.dump(doc_types, f, indent=2)

    # ── 4. Layout detection + layout-aware chunking ────
    print("\n[4/6] Detecting layout + chunking...")
    layouts      = detect_layouts_batch(clean_docs, pdf_folder=documents_folder)
    chunked_docs = chunk_documents_by_layout(layouts, chunk_size=CHUNK_SIZE)

    # Save layout summaries for UI display
    layout_summaries = {
        fname: layout.summary()
        for fname, layout in layouts.items()
    }

    # Save detailed region info for UI
    layout_details = {}
    for fname, layout in layouts.items():
        layout_details[fname] = {
            "summary": layout.summary(),
            "regions": [
                {
                    "type":       r.region_type,
                    "text":       r.text[:200],
                    "page":       r.page,
                    "confidence": r.confidence
                }
                for r in layout.regions[:50]  # cap at 50 for storage
            ]
        }

    with open(LAYOUTS_SAVE_PATH, "w") as f:
        json.dump(layout_details, f, indent=2)
    print(f"  Layout data saved → {LAYOUTS_SAVE_PATH}")

    # ── 5. Flatten chunks + metadata ───────────────────
    all_chunks, all_metadata = [], []
    for filename, chunks in chunked_docs.items():
        doc_label = doc_types.get(filename, {}).get("label", "unknown")
        for chunk_dict in chunks:
            all_chunks.append(chunk_dict["text"])
            all_metadata.append({
                "source":      filename,
                "text":        chunk_dict["text"],
                "doc_type":    doc_label,
                "region_type": chunk_dict.get("region_type", "paragraph"),
                "page":        chunk_dict.get("page", 0)
            })

    print(f"\n  Total chunks: {len(all_chunks)}")

    # ── 6. Embed + store ───────────────────────────────
    print("\n[5/6] Generating embeddings...")
    embeddings = embed_chunks(all_chunks)

    print("\n[6/6] Storing vectors...")
    index = build_faiss_index(embeddings)
    save_faiss_index(index, all_metadata)
    add_to_chroma(chroma_col, all_chunks, embeddings, all_metadata)

    # Reset retriever cache
    from src.rag.rag_pipeline import reset_retriever_cache
    reset_retriever_cache()

    print("\n✅ Indexing complete!")
    print(f"   Documents : {len(raw_docs)}")
    print(f"   Chunks    : {len(all_chunks)}")
    print(f"   Methods   : "
          f"{set(l.method for l in layouts.values())}")

    return index, all_metadata, doc_types, layout_details
import sys
import os
import json
import warnings
import shutil
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

import streamlit as st
from components.uploader import render_uploader, clear_documents_folder
from components.sidebar import render_sidebar
from components.chat import render_chat_history, render_query_input
from components.result_card import render_result_card
from components.doc_type_badge import render_doc_type_badge
from components.layout_viewer import render_layout_viewer
from src.rag.rag_pipeline import ask, ask_stream, reset_retriever_cache

from src.pipeline.full_pipeline import run_full_pipeline
from src.rag.rag_pipeline import ask, reset_retriever_cache
from src.config.settings import (ROOT_DIR, DOCUMENTS_FOLDER,
                                  FAISS_INDEX_PATH, FAISS_META_PATH)

DOC_TYPES_PATH = os.path.join(ROOT_DIR, "vector_db",
                               "faiss_index", "doc_types.json")
LAYOUTS_PATH   = os.path.join(ROOT_DIR, "vector_db",
                               "faiss_index", "layouts.json")


def clear_all_session_data():
    """
    Wipe documents folder + vector index + session state.
    Called on fresh start or when user clicks Clear.
    """
    # Clear uploaded PDFs
    clear_documents_folder()

    # Clear FAISS index files
    for path in [FAISS_INDEX_PATH, FAISS_META_PATH,
                 DOC_TYPES_PATH, LAYOUTS_PATH]:
        if os.path.exists(path):
            os.remove(path)

    # Clear ChromaDB
    chroma_path = os.path.join(ROOT_DIR, "vector_db", "chroma_db")
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    # Reset retriever cache
    reset_retriever_cache()


# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="Smart Document Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state init — ALWAYS start clean ────────────
# Never auto-load from disk — user must upload + index each session
defaults = {
    "messages":        [],
    "indexed":         False,      # ← always False on startup
    "doc_types":       {},
    "pipeline_result": None,
    "layout_details":  {}
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Auto-clear documents folder on first load ──────────
# This ensures no leftover PDFs from previous sessions
if "session_started" not in st.session_state:
    clear_all_session_data()
    st.session_state["session_started"] = True


# ══════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════
st.title("🧠 Smart Document Analyzer")
st.caption("Upload a PDF → Index → Ask anything about it.")


# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
new_upload = render_uploader()
config     = render_sidebar()

if new_upload:
    st.info("📥 New files uploaded — click **Index / Re-index Documents**.")

# Clear session button
st.sidebar.divider()
if st.sidebar.button("🗑️ Clear & Start Over",
                     use_container_width=True,
                     help="Removes all uploaded files and resets the session"):
    clear_all_session_data()
    # Reset all session state keys
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ══════════════════════════════════════════════════════
#  INDEXING
# ══════════════════════════════════════════════════════
if config["reindex"]:
    if config["doc_count"] == 0:
        st.sidebar.warning("No PDFs found. Upload documents first.")
    else:
        with st.spinner("⚙️ Indexing + classifying + detecting layout..."):
            try:
                result = run_full_pipeline()
                reset_retriever_cache()
                from src.utils.cache import clear_cache
                clear_cache()
                st.session_state.indexed         = True
                st.session_state.doc_types       = result["doc_types"]
                st.session_state.layout_details  = result.get("layout_details", {})
                st.session_state.pipeline_result = result
                st.session_state.messages        = []

                st.sidebar.success(
                    f"✅ Done! {result['doc_count']} docs · "
                    f"{result['chunk_count']} chunks"
                )
                st.rerun()

            except Exception as e:
                st.sidebar.error(f"Indexing failed: {e}")
                st.exception(e)


# ══════════════════════════════════════════════════════
#  NOT INDEXED YET
# ══════════════════════════════════════════════════════
if not st.session_state.indexed:
    st.warning(
        "⚠️ No index found. Upload PDFs and click "
        "**Index / Re-index Documents** to get started."
    )
    st.stop()


# ══════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════
col_chat, col_info = st.columns([3, 1])


# ── RIGHT COLUMN ───────────────────────────────────────
with col_info:
    st.markdown("### 📄 Document Info")

    if st.session_state.doc_types:
        for filename, dtype in st.session_state.doc_types.items():
            st.caption(f"**{filename}**")
            render_doc_type_badge(
                doc_type=dtype.get("label", "unknown"),
                confidence=dtype.get("confidence", 0.0),
                all_scores=dtype.get("all_scores", {})
            )
    else:
        st.caption("No classification results yet.")

    if st.session_state.pipeline_result:
        st.divider()
        r = st.session_state.pipeline_result
        col_a, col_b = st.columns(2)
        col_a.metric("📄 Docs",   r["doc_count"])
        col_b.metric("🧩 Chunks", r["chunk_count"])

    st.divider()
    render_layout_viewer(st.session_state.layout_details or None)


# ── LEFT COLUMN — chat ─────────────────────────────────
with col_chat:
    st.markdown("### 💬 Ask your document")

    render_chat_history(st.session_state.messages)

    query = render_query_input()

    if query:
        st.session_state.messages.append({
            "role":    "user",
            "content": query
        })

        with st.chat_message("assistant"):
            try:
                # ── Get metadata + streaming generator ────
                metadata, token_stream = ask_stream(
                    question=query,
                    doc_type=None,
                    top_k=config["top_k"],
                    store=config["store"],
                    use_cache=True,
                    use_expansion=True
                )

                # ── Stream the answer tokens live ──────────
                answer_placeholder = st.empty()
                full_answer        = ""

                if metadata.get("from_cache"):
                    # Cache hit — show instantly
                    for token in token_stream:
                        full_answer += token
                    answer_placeholder.markdown(full_answer)
                    st.caption("⚡ Instant — served from cache")
                else:
                    # Stream tokens as they arrive
                    with answer_placeholder:
                        full_answer = st.write_stream(token_stream)

                # ── Run evaluation after answer is complete ─
                from src.utils.evaluator import evaluate_answer
                eval_metrics = evaluate_answer(
                    query, full_answer, metadata.get("chunks", [])
                )
                metadata["eval"]   = eval_metrics
                metadata["answer"] = full_answer

                # ── Render metrics + sources ───────────────
                render_result_card(
                    answer="",        # already rendered above via stream
                    sources=metadata["sources"],
                    chunks=metadata["chunks"],
                    latency=metadata.get("latency_s",
                            metadata["timing"].get("total_s", 0)),
                    doc_type=metadata.get("doc_type"),
                    eval_metrics=eval_metrics,
                    timing=metadata.get("timing"),
                    expanded_queries=metadata.get("expanded_queries"),
                    from_cache=metadata.get("from_cache", False)
                )

                # ── Save to history ────────────────────────
                st.session_state.messages.append({
                    "role":             "assistant",
                    "content":          full_answer,
                    "sources":          metadata["sources"],
                    "latency":          metadata.get("latency_s", 0),
                    "doc_type":         metadata.get("doc_type"),
                    "confidence":       metadata.get("confidence", 0.0),
                    "eval":             eval_metrics,
                    "from_cache":       metadata.get("from_cache", False)
                })

            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.exception(e)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": err,
                    "sources": [],
                    "latency": 0
                })
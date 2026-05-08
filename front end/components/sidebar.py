import streamlit as st
import os
from src.config.settings import DOCUMENTS_FOLDER, VECTOR_STORE


def render_sidebar() -> dict:
    """
    Renders the full sidebar controls.

    Returns a config dict the main app uses:
    {
        "store":        "faiss" or "chroma",
        "top_k":        int,
        "reindex":      bool,
        "doc_count":    int
    }
    """
    st.sidebar.markdown("## ⚙️ Settings")

    # Vector store selector
    store = st.sidebar.radio(
        "Vector Store",
        options=["faiss", "chroma"],
        index=0 if VECTOR_STORE == "faiss" else 1,
        horizontal=True
    )

    # Top-K slider
    top_k = st.sidebar.slider(
        "Chunks to retrieve (Top-K)",
        min_value=1,
        max_value=8,
        value=3,
        help="More chunks = more context, but slower"
    )

    st.sidebar.divider()

    # Document stats
    doc_files = [f for f in os.listdir(DOCUMENTS_FOLDER)
                 if f.endswith(".pdf")] if os.path.exists(DOCUMENTS_FOLDER) else []
    doc_count = len(doc_files)

    st.sidebar.markdown("### 📄 Documents")
    if doc_files:
        for f in doc_files:
            st.sidebar.caption(f"• {f}")
    else:
        st.sidebar.caption("No documents uploaded yet.")

    st.sidebar.divider()

    # Re-index button
    reindex = st.sidebar.button(
        "🔄 Index / Re-index Documents",
        help="Run this after uploading new PDFs",
        use_container_width=True
    )

    return {
        "store":     store,
        "top_k":     top_k,
        "reindex":   reindex,
        "doc_count": doc_count
    }
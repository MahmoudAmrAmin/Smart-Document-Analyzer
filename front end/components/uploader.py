import streamlit as st
import os
import shutil
from src.config.settings import DOCUMENTS_FOLDER


def clear_documents_folder():
    """Remove all PDFs from the documents folder."""
    if os.path.exists(DOCUMENTS_FOLDER):
        for f in os.listdir(DOCUMENTS_FOLDER):
            if f.endswith(".pdf"):
                os.remove(os.path.join(DOCUMENTS_FOLDER, f))
    print("  Documents folder cleared.")


def render_uploader() -> bool:
    """
    Renders the PDF upload widget in the sidebar.
    Returns True if new files were uploaded.
    """
    st.sidebar.markdown("### 📂 Upload Documents")

    uploaded_files = st.sidebar.file_uploader(
        label="Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze"
    )

    if not uploaded_files:
        return False

    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
    new_files = []

    for uploaded_file in uploaded_files:
        save_path = os.path.join(DOCUMENTS_FOLDER, uploaded_file.name)
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            new_files.append(uploaded_file.name)

    if new_files:
        st.sidebar.success(f"✅ Saved: {', '.join(new_files)}")
        return True

    return False
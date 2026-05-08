import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

from src.pipeline.indexing_pipeline import run_indexing_pipeline
from src.config.settings import DOCUMENTS_FOLDER


def run_full_pipeline(documents_folder: str = DOCUMENTS_FOLDER) -> dict:
    index, metadata, doc_types, layout_details = run_indexing_pipeline(
        documents_folder
    )
    return {
        "doc_count":      len(doc_types),
        "chunk_count":    len(metadata),
        "doc_types":      doc_types,
        "layout_details": layout_details
    }
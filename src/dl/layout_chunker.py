"""
Layout-aware chunker.
Uses detected layout regions (titles, paragraphs, tables)
to create semantically meaningful chunks instead of
arbitrary word-count splits.

Benefits:
- Titles are attached to following paragraphs (better context)
- Tables stay intact as single chunks
- Lists stay grouped together
- Section boundaries respected
"""

import os
import sys
from typing import List

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

from src.dl.layout_detector import DocumentLayout, LayoutRegion
from src.config.settings import CHUNK_SIZE


def chunk_by_layout(layout: DocumentLayout,
                    chunk_size: int = CHUNK_SIZE) -> List[dict]:
    """
    Create chunks that respect document structure.

    Rules:
    - title + its following paragraphs → one chunk
    - table → always its own chunk (never split)
    - list items → grouped into one chunk
    - long paragraphs → split by word count with overlap

    Returns:
        List of dicts: {text, region_type, page, source}
    """
    chunks  = []
    regions = layout.regions
    i       = 0

    while i < len(regions):
        region = regions[i]

        # ── Table: never split ─────────────────────────
        if region.region_type == "table":
            if region.text.strip():
                chunks.append({
                    "text":        f"[TABLE]\n{region.text}",
                    "region_type": "table",
                    "page":        region.page,
                    "source":      layout.filename
                })
            i += 1

        # ── Title: attach following paragraphs ─────────
        elif region.region_type == "title":
            title_text = region.text
            body_parts = []
            j          = i + 1

            # Collect paragraphs/lists until next title or table
            while j < len(regions):
                next_r = regions[j]
                if next_r.region_type == "title":
                    break
                if next_r.region_type == "table":
                    break
                body_parts.append(next_r.text)
                j += 1

            combined = title_text
            if body_parts:
                combined += "\n" + " ".join(body_parts)

            # Split if too long
            words = combined.split()
            if len(words) <= chunk_size:
                chunks.append({
                    "text":        combined,
                    "region_type": "section",
                    "page":        region.page,
                    "source":      layout.filename
                })
            else:
                # Split oversized section keeping title in first chunk
                sub_chunks = _split_long_text(combined, chunk_size)
                for sc in sub_chunks:
                    chunks.append({
                        "text":        sc,
                        "region_type": "section",
                        "page":        region.page,
                        "source":      layout.filename
                    })

            i = j   # skip consumed regions

        # ── List: group consecutive list items ─────────
        elif region.region_type == "list":
            list_items = [region.text]
            j          = i + 1
            while j < len(regions) and regions[j].region_type == "list":
                list_items.append(regions[j].text)
                j += 1

            list_text = "\n".join(list_items)
            chunks.append({
                "text":        f"[LIST]\n{list_text}",
                "region_type": "list",
                "page":        region.page,
                "source":      layout.filename
            })
            i = j

        # ── Paragraph: split if too long ───────────────
        else:
            words = region.text.split()
            if len(words) <= chunk_size:
                if region.text.strip():
                    chunks.append({
                        "text":        region.text,
                        "region_type": "paragraph",
                        "page":        region.page,
                        "source":      layout.filename
                    })
            else:
                for sc in _split_long_text(region.text, chunk_size):
                    chunks.append({
                        "text":        sc,
                        "region_type": "paragraph",
                        "page":        region.page,
                        "source":      layout.filename
                    })
            i += 1

    # Filter empty chunks
    chunks = [c for c in chunks if len(c["text"].split()) > 5]

    print(f"  Layout chunker: {len(chunks)} chunks "
          f"from {len(regions)} regions")
    return chunks


def _split_long_text(text: str,
                     chunk_size: int,
                     overlap: int = 50) -> List[str]:
    """Split oversized text with word overlap."""
    words  = text.split()
    result = []
    start  = 0
    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        result.append(chunk)
        start += chunk_size - overlap
        if start < len(words) and len(words) - start < overlap:
            break
    return result


def chunk_documents_by_layout(layouts: dict,
                               chunk_size: int = CHUNK_SIZE) -> dict:
    """
    Apply layout-aware chunking to all documents.

    Args:
        layouts: {filename: DocumentLayout}

    Returns:
        {filename: [chunk_dict, ...]}
        where each chunk_dict has keys: text, region_type, page, source
    """
    result = {}
    for filename, layout in layouts.items():
        print(f"\n  Chunking by layout: {filename}")
        result[filename] = chunk_by_layout(layout, chunk_size)
    return result
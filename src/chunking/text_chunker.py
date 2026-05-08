import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str,
               chunk_size: int = 200,
               overlap: int = 50) -> List[str]:
    """
    Sentence-aware word-based chunking.
    Respects sentence boundaries so answers are never cut mid-sentence.
    """
    sentences = split_into_sentences(text)
    chunks    = []
    current   = []
    count     = 0

    for sentence in sentences:
        words = sentence.split()
        # If adding this sentence exceeds chunk_size, flush current chunk
        if count + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            # Keep last `overlap` words for context continuity
            overlap_words = " ".join(current).split()[-overlap:]
            current       = overlap_words + words
            count         = len(current)
        else:
            current.extend(words)
            count += len(words)

    # Don't forget last chunk
    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if len(c.split()) > 10]


def chunk_documents(documents: dict,
                    chunk_size: int = 200,
                    overlap: int = 50) -> dict:
    chunked = {}
    for filename, text in documents.items():
        chunks = chunk_text(text, chunk_size, overlap)
        chunked[filename] = chunks
        print(f"  Chunked: {filename} → {len(chunks)} chunks")
    return chunked
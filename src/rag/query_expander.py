"""
Query expansion — improves retrieval by reformulating
the user's question into multiple search variants.

Example:
  Input:  "What is the salary?"
  Output: ["What is the salary?",
           "salary compensation payment amount",
           "how much does the position pay"]
"""

import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

from src.config.settings import GROQ_API_KEY, GROQ_MODEL
from typing import List
import re


def _expand_with_llm(question: str) -> List[str]:
    """Use Groq to generate query variants."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""Generate 2 alternative search queries for the following question.
These should help find relevant text chunks in a document.
Return ONLY the queries, one per line, no numbering, no explanations.

Original question: {question}

Alternative queries:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.3
    )

    raw      = response.choices[0].message.content.strip()
    variants = [
        line.strip() for line in raw.split("\n")
        if line.strip() and len(line.strip()) > 5
    ]
    return variants[:2]   # max 2 variants


def _expand_with_rules(question: str) -> List[str]:
    """
    Rule-based query expansion — no API call needed.
    Generates keyword-focused variant from the question.
    """
    # Remove question words to create keyword query
    stopwords   = {"what", "is", "the", "are", "how", "why",
                   "when", "who", "where", "does", "do", "a",
                   "an", "of", "in", "for", "and", "or", "?"}
    words       = question.lower().replace("?", "").split()
    keywords    = [w for w in words if w not in stopwords]
    keyword_str = " ".join(keywords)

    variants = []
    if keyword_str and keyword_str != question.lower():
        variants.append(keyword_str)

    return variants


def expand_query(question: str,
                 use_llm: bool = True) -> List[str]:
    """
    Expand a question into multiple search queries.

    Args:
        question: original user question
        use_llm:  use Groq for expansion (falls back to rules)

    Returns:
        List starting with original question + variants
        e.g. ["original", "variant1", "variant2"]
    """
    variants = [question]   # always include original

    if use_llm and GROQ_API_KEY:
        try:
            llm_variants = _expand_with_llm(question)
            variants.extend(llm_variants)
            print(f"  Query expanded (LLM): {len(variants)} variants")
        except Exception as e:
            print(f"  Query expansion fallback to rules: {e}")
            variants.extend(_expand_with_rules(question))
    else:
        variants.extend(_expand_with_rules(question))
        print(f"  Query expanded (rules): {len(variants)} variants")

    # Deduplicate while preserving order
    seen   = set()
    unique = []
    for v in variants:
        if v.lower() not in seen:
            seen.add(v.lower())
            unique.append(v)

    return unique
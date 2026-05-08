from typing import List


PROMPT_TEMPLATE = """You are a helpful assistant.
Answer the question using ONLY the information provided in the context below.
If the answer is not found in the context, say:
"I could not find this information in the document."
Do not make up or infer anything not present in the context.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str,
                 retrieved_chunks: List[dict],
                 doc_type: str = None) -> str:
    """
    Build prompt from retrieved chunks + question.
    doc_type param kept for compatibility but no longer used.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Chunk {i} — Source: {chunk['source']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    return PROMPT_TEMPLATE.format(context=context, question=question)
import streamlit as st
from typing import List


def render_result_card(answer: str,
                       sources: List[dict],
                       chunks: List[dict],
                       latency: float,
                       doc_type: str = None,
                       eval_metrics: dict = None,
                       timing: dict = None,
                       expanded_queries: list = None,
                       from_cache: bool = False):

    # Answer text
    st.markdown(answer)

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⚡ Latency",   f"{latency}s")
    col2.metric("📦 Chunks",    len(chunks))
    col3.metric("🏆 Top score",
                f"{max(s['score'] for s in sources):.3f}"
                if sources else "—")
    col4.metric("📊 Quality",
                eval_metrics.get("quality_label", "—").split()[-1]
                if eval_metrics else "—")

    # Source chunks
    with st.expander("📎 View source chunks"):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(
                f"**Chunk {i}** — `{chunk['source']}` "
                f"· page `{chunk.get('page', '?')}` "
                f"· type `{chunk.get('region_type', 'text')}` "
                f"(score: `{chunk['score']:.3f}`)"
            )
            st.text(
                chunk["text"][:400] +
                ("..." if len(chunk["text"]) > 400 else "")
            )
            if i < len(chunks):
                st.divider()

    # Eval metrics panel
    from components.eval_metrics import render_eval_metrics
    render_eval_metrics(
        eval_metrics=eval_metrics,
        timing=timing,
        expanded_queries=expanded_queries,
        from_cache=from_cache
    )
import streamlit as st


def render_eval_metrics(eval_metrics: dict,
                        timing: dict = None,
                        expanded_queries: list = None,
                        from_cache: bool = False):
    """
    Renders answer quality metrics panel below each answer.
    """
    if not eval_metrics:
        return

    with st.expander("📊 Answer Quality Metrics", expanded=False):

        # ── Cache badge ────────────────────────────────
        if from_cache:
            st.success("⚡ Served from cache — instant response")
            st.divider()

        # ── Quality label ──────────────────────────────
        label = eval_metrics.get("quality_label", "")
        st.markdown(f"**Overall quality: {label}**")

        # ── Metric bars ────────────────────────────────
        col1, col2, col3 = st.columns(3)

        with col1:
            val = eval_metrics.get("retrieval_relevance", 0)
            st.caption("🔍 Retrieval Relevance")
            st.progress(val, text=f"{val:.0%}")
            st.caption("How well chunks match your question")

        with col2:
            val = eval_metrics.get("answer_grounding", 0)
            st.caption("📎 Answer Grounding")
            st.progress(val, text=f"{val:.0%}")
            st.caption("How much answer comes from document")

        with col3:
            val = eval_metrics.get("top_chunk_score", 0)
            st.caption("🏆 Top Chunk Score")
            st.progress(val, text=f"{val:.0%}")
            st.caption("Best semantic similarity found")

        # ── Confidence ─────────────────────────────────
        st.divider()
        conf     = eval_metrics.get("confidence", 0)
        conf_col = (
            "green"  if conf >= 0.75 else
            "orange" if conf >= 0.55 else
            "red"
        )
        st.markdown(
            f"**Combined confidence:** "
            f"<span style='color:{conf_col}; font-size:18px; "
            f"font-weight:700;'>{conf:.0%}</span>",
            unsafe_allow_html=True
        )

        # ── Timing breakdown ───────────────────────────
        if timing:
            st.divider()
            st.caption("⏱️ Latency breakdown")
            t_cols = st.columns(4)
            labels = [
                ("🔎 Expand",   "expansion_s"),
                ("📦 Retrieve", "retrieval_s"),
                ("🤖 Generate", "generation_s"),
                ("📊 Evaluate", "eval_s")
            ]
            for col, (lbl, key) in zip(t_cols, labels):
                val = timing.get(key, 0)
                col.metric(lbl, f"{val}s")

        # ── Expanded queries ───────────────────────────
        if expanded_queries and len(expanded_queries) > 1:
            st.divider()
            st.caption("🔍 Query variants used for retrieval:")
            for i, q in enumerate(expanded_queries):
                prefix = "Original" if i == 0 else f"Variant {i}"
                st.caption(f"  `{prefix}:` {q}")
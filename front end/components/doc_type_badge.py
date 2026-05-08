import streamlit as st


# Visual config per doc type
DOC_TYPE_CONFIG = {
    "contract": {
        "icon":  "🟦",
        "label": "Contract",
        "color": "#4e79a7",
        "desc":  "Legal agreement document"
    },
    "invoice": {
        "icon":  "🟨",
        "label": "Invoice",
        "color": "#f28e2b",
        "desc":  "Financial / billing document"
    },
    "research": {
        "icon":  "🟩",
        "label": "Research / Report",
        "color": "#59a14f",
        "desc":  "Academic or analytical document"
    },
    "unknown": {
        "icon":  "⬜",
        "label": "Unknown",
        "color": "#aaaaaa",
        "desc":  "Could not classify document"
    }
}


def render_doc_type_badge(doc_type: str, confidence: float,
                          all_scores: dict = None):
    """
    Renders a styled document type card with:
    - Type label + icon
    - Confidence progress bar
    - All class scores breakdown
    """
    config = DOC_TYPE_CONFIG.get(doc_type, DOC_TYPE_CONFIG["unknown"])

    st.markdown(f"""
    <div style="
        background: {config['color']}22;
        border-left: 4px solid {config['color']};
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    ">
        <div style="font-size:18px; font-weight:700; color:{config['color']}">
            {config['icon']} {config['label']}
        </div>
        <div style="font-size:12px; color:#888; margin-top:2px">
            {config['desc']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bar
    st.caption("Classification confidence")
    st.progress(confidence, text=f"{confidence:.1%}")

    # All scores breakdown
    if all_scores:
        with st.expander("📊 All class scores"):
            for label, score in sorted(
                all_scores.items(), key=lambda x: x[1], reverse=True
            ):
                cfg = DOC_TYPE_CONFIG.get(label, DOC_TYPE_CONFIG["unknown"])
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.caption(f"{cfg['icon']} {label}")
                with col2:
                    st.progress(score, text=f"{score:.1%}")
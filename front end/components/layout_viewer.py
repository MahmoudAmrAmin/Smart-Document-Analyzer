import streamlit as st
import os
import json

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

LAYOUTS_PATH = os.path.join(
    ROOT_DIR, "vector_db", "faiss_index", "layouts.json"
)

# Icons per region type
REGION_ICONS = {
    "title":     "📌",
    "paragraph": "📝",
    "table":     "📊",
    "list":      "📋",
    "figure":    "🖼️",
    "section":   "📂",
    "unknown":   "❓"
}

REGION_COLORS = {
    "title":     "#4e79a7",
    "paragraph": "#59a14f",
    "table":     "#f28e2b",
    "list":      "#e15759",
    "figure":    "#b07aa1",
    "section":   "#76b7b2",
    "unknown":   "#aaaaaa"
}


def render_layout_viewer(layout_details: dict = None):
    """
    Renders the document layout structure panel.
    Shows region breakdown + individual regions with type badges.
    """
    st.markdown("### 🗂️ Document Layout")

    # Load from disk if not passed directly
    if not layout_details:
        if os.path.exists(LAYOUTS_PATH):
            with open(LAYOUTS_PATH) as f:
                layout_details = json.load(f)
        else:
            st.caption("No layout data yet. Index a document first.")
            return

    for filename, data in layout_details.items():
        summary = data.get("summary", {})
        regions = data.get("regions", [])

        with st.expander(
            f"📄 {filename} — {summary.get('regions', 0)} regions "
            f"({summary.get('method', 'unknown')} method)",
            expanded=True
        ):
            # ── Summary metrics ────────────────────────
            breakdown = summary.get("breakdown", {})
            cols      = st.columns(len(breakdown) if breakdown else 1)
            for col, (rtype, count) in zip(cols, breakdown.items()):
                icon = REGION_ICONS.get(rtype, "❓")
                col.metric(f"{icon} {rtype}", count)

            # ── Detection method badge ─────────────────
            method = summary.get("method", "unknown")
            color  = "#4e79a7" if method == "layoutparser" else "#f28e2b"
            st.markdown(
                f'<span style="background:{color}22; color:{color}; '
                f'padding:3px 10px; border-radius:12px; font-size:12px;">'
                f'{"🤖 LayoutParser (DL)" if method == "layoutparser" else "📐 Rule-based (fallback)"}'
                f'</span>',
                unsafe_allow_html=True
            )

            st.divider()

            # ── Region list ────────────────────────────
            if regions:
                st.caption("Document regions (first 50):")
                for region in regions:
                    rtype  = region.get("type", "unknown")
                    icon   = REGION_ICONS.get(rtype, "❓")
                    color  = REGION_COLORS.get(rtype, "#aaaaaa")
                    text   = region.get("text", "")[:120]
                    page   = region.get("page", 0)

                    st.markdown(
                        f'<div style="border-left: 3px solid {color}; '
                        f'padding: 6px 10px; margin: 4px 0; '
                        f'border-radius: 0 6px 6px 0; '
                        f'background: {color}11;">'
                        f'<span style="color:{color}; font-size:11px; '
                        f'font-weight:600;">{icon} {rtype.upper()}'
                        f' · pg {page}</span><br>'
                        f'<span style="font-size:12px; color: var(--text-color);">'
                        f'{text}{"..." if len(region.get("text","")) > 120 else ""}'
                        f'</span></div>',
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No region details available.")
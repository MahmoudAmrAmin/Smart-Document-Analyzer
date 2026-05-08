import streamlit as st
from typing import List


def render_chat_history(messages: List[dict]):
    """
    Renders the full conversation history.

    Each message dict:
    {
        "role":    "user" | "assistant",
        "content": str,
        "sources": [...],   # only for assistant
        "latency": float    # only for assistant
    }
    """
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show sources + latency under assistant replies
            if msg["role"] == "assistant" and msg.get("sources"):
                cols = st.columns([3, 1])
                with cols[0]:
                    source_names = list({s["source"] for s in msg["sources"]})
                    badges = "  ".join([f"`📎 {s}`" for s in source_names])
                    st.markdown(badges)
                with cols[1]:
                    if msg.get("latency"):
                        st.caption(f"⚡ {msg['latency']}s")


def render_query_input() -> str | None:
    """
    Renders the chat input box at the bottom.
    Returns the user's query string, or None if empty.
    """
    return st.chat_input("Ask a question about your document...")
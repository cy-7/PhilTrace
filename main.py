"""PhilTrace Streamlit application entrypoint."""

from __future__ import annotations

import streamlit as st

from ui.style import apply_all_styles
from ui.view_chat import render_chat_view
from ui.view_drift import render_drift_view

APP_VERSION = "1.1.0"


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:0.5rem 0 0.8rem 0;">
              <h2 style="margin:0; letter-spacing:0.03em;">PhilTrace</h2>
              <p style="margin:0.35rem 0 0 0; font-size:0.82rem; opacity:0.82;">
                哲学概念历时分析平台
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "功能导航",
            options=["concept_drift", "free_chat"],
            format_func=lambda item: {
                "concept_drift": "概念漂移研究",
                "free_chat": "自由对话",
            }[item],
            key="nav_radio",
        )

        st.markdown("---")
        st.caption(f"Version {APP_VERSION}")
        st.caption("Bauhaus UI Edition")

    return page


def main() -> None:
    st.set_page_config(
        page_title="PhilTrace | 哲学概念分析",
        page_icon="●",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                "PhilTrace: a concept drift and philosophy chat platform. "
                "Built for research, education, and competition demos."
            ),
            "Report a bug": "https://github.com/cy-7/PhilTrace/issues",
        },
    )

    apply_all_styles()
    current_page = render_sidebar()

    if current_page == "concept_drift":
        render_drift_view()
    else:
        render_chat_view()


if __name__ == "__main__":
    main()

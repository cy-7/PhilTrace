"""Robust Bauhaus UI styles with guaranteed text visibility."""

import streamlit as st


def apply_base_styles() -> None:
    st.markdown(
        """
    <style>
    :root {
        --bau-red: #E53935;
        --bau-yellow: #FDD835;
        --bau-blue: #1E88E5;
        --bau-black: #212121;
        --bau-white: #FAFAFA;
        --bau-gray: #9E9E9E;

        --phil-primary: var(--bau-blue);
        --phil-accent: var(--bau-red);
        --phil-bg-dark: var(--bau-white);
        --phil-bg-card: #FFFFFF;
        --phil-text-primary: var(--bau-black);
        --phil-text-secondary: #444444;
        --phil-text-muted: #777777;
        --phil-border: var(--bau-black);
    }

    html, body, [class*="css"] {
        color: var(--bau-black);
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: var(--bau-white) !important;
        color: var(--bau-black) !important;
    }

    /* Ensure main content text is always visible */
    .stApp p,
    .stApp span,
    .stApp div,
    .stApp label,
    .stApp li,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6,
    .stMarkdown,
    .stText,
    .stCaption {
        color: var(--bau-black);
    }

    .block-container {
        padding: 1.5rem 2.2rem;
        max-width: 1280px;
        position: relative;
        z-index: 2;
    }

    /* Decorative geometric shapes */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0;
        right: 0;
        width: min(36vw, 520px);
        height: min(25vh, 220px);
        background: var(--bau-yellow);
        clip-path: polygon(100% 0, 38% 0, 100% 100%);
        z-index: 0;
        pointer-events: none;
    }

    [data-testid="stAppViewContainer"]::after {
        content: "";
        position: fixed;
        left: 20px;
        bottom: 20px;
        width: 80px;
        height: 80px;
        background: var(--bau-blue);
        border-radius: 50%;
        z-index: 0;
        pointer-events: none;
    }

    a {
        color: var(--bau-blue);
        text-decoration: none;
        border-bottom: 1px solid transparent;
    }

    a:hover {
        border-bottom-color: var(--bau-blue);
    }

    section[data-testid="stSidebar"] {
        background: #F3F3F3;
        border-right: 3px solid var(--bau-red);
    }

    section[data-testid="stSidebar"] * {
        color: var(--bau-black) !important;
    }


    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
        color: var(--bau-black) !important;
    }

    section[data-testid="stSidebar"] .stRadio > div {
        gap: 0.25rem;
    }

    section[data-testid="stSidebar"] .stRadio label {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        background: transparent !important;
        padding: 0.15rem 0.05rem !important;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: transparent !important;
    }

    .phil-title {
        font-family: Futura, "Avenir Next", "Century Gothic", "Segoe UI", sans-serif;
        font-size: clamp(1.8rem, 3.6vw, 3.2rem);
        font-weight: 800;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: var(--bau-black) !important;
        margin: 0 0 0.4rem 0;
    }

    .phil-subtitle {
        font-family: Futura, "Avenir Next", "Century Gothic", "Segoe UI", sans-serif;
        color: var(--phil-text-secondary) !important;
        border-left: 4px solid var(--bau-blue);
        padding-left: 0.7rem;
        margin-bottom: 1.2rem;
        max-width: 64ch;
    }

    .phil-section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--bau-black) !important;
        margin-bottom: 0.8rem;
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
    }

    .phil-section-title::before {
        content: "";
        width: 14px;
        height: 14px;
        background: var(--bau-red);
        transform: rotate(45deg);
    }

    .phil-card {
        background: #FFFFFF;
        border: 2px solid var(--bau-black);
        border-radius: 0;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .period-card {
        border-left: 8px solid var(--bau-red);
    }

    .period-card h3,
    .period-card p,
    .period-badge {
        color: var(--bau-black) !important;
    }

    .period-badge {
        border: 1px solid var(--bau-black);
        border-radius: 0;
        padding: 0.15rem 0.4rem;
        font-size: 0.72rem;
        font-weight: 700;
        background: #FFFFFF !important;
    }

    .stTextInput > div > div,
    .stTextArea > div > div,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #FFFFFF !important;
        border: 2px solid var(--bau-black) !important;
        border-radius: 0 !important;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox * {
        color: var(--bau-black) !important;
    }

    .stButton > button {
        border-radius: 0;
        border: 2px solid var(--bau-black);
        background: var(--bau-blue);
        color: #FFFFFF !important;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: var(--bau-red);
    }

    .streamlit-expander {
        border: 2px solid var(--bau-black);
        border-radius: 0;
        background: #FFFFFF;
    }

    .metric-card {
        background: #FFFFFF;
        border: 2px solid var(--bau-black);
        border-radius: 0;
        padding: 0.8rem 1rem;
        position: relative;
    }

    .metric-card::after {
        content: "";
        position: absolute;
        right: 0;
        top: 0;
        width: 10px;
        height: 100%;
        background: var(--bau-yellow);
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: var(--bau-black) !important;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #444 !important;
        margin-top: 0.2rem;
        text-transform: uppercase;
    }

    .chart-container {
        background: #FFFFFF;
        border: 2px solid var(--bau-black);
        border-radius: 0;
        padding: 0.9rem;
        margin-bottom: 1rem;
    }

    .chart-title {
        font-size: 0.92rem;
        font-weight: 800;
        color: var(--bau-black) !important;
        margin-bottom: 0.6rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid var(--bau-black);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stChatMessage"] {
        border: 2px solid var(--bau-black);
        border-radius: 0;
    }

    .welcome-message {
        text-align: left;
        padding: 2rem 1.3rem;
        color: var(--bau-black) !important;
        border: 2px dashed var(--bau-black);
        background: #FFFFFF;
    }

    .welcome-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: var(--bau-black) !important;
        text-transform: uppercase;
    }

    .welcome-hint {
        color: #444 !important;
    }



    section[data-testid="stSidebar"] .stRadio,
    section[data-testid="stSidebar"] .stRadio > div,
    section[data-testid="stSidebar"] [role="radiogroup"],
    section[data-testid="stSidebar"] fieldset {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        background: transparent !important;
        padding: 0 !important;
    }

    @media (max-width: 960px) {
        .block-container {
            padding: 1rem;
        }

        [data-testid="stAppViewContainer"]::before {
            width: 48vw;
            height: 18vh;
        }

        [data-testid="stAppViewContainer"]::after {
            width: 56px;
            height: 56px;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def apply_drift_view_styles() -> None:
    st.markdown(
        """
    <style>
    .concept-input-section {
        text-align: left;
        padding: 1.1rem;
        background: #FFFFFF;
        border: 2px solid var(--bau-black);
        margin-bottom: 1.2rem;
        position: relative;
    }

    .concept-input-section::before {
        content: "";
        position: absolute;
        top: -2px;
        right: -2px;
        width: 20px;
        height: 20px;
        background: var(--bau-red);
    }

    .periods-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.9rem;
        margin-bottom: 1.4rem;
    }

    .charts-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1.4rem;
    }

    .quote-card {
        background: #FFFFFF;
        border: 2px solid var(--bau-black);
        border-left: 8px solid var(--bau-blue);
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.55rem;
    }

    .quote-text,
    .quote-source {
        color: var(--bau-black) !important;
    }

    @media (max-width: 1200px) {
        .periods-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 1024px) {
        .charts-row {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 768px) {
        .periods-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def apply_chat_view_styles() -> None:
    st.markdown(
        """
    <style>
    .settings-panel {
        background: #FFFFFF;
        border: 2px solid var(--bau-black);
        padding: 0.95rem;
        margin-bottom: 1rem;
    }

    .settings-title {
        font-size: 0.8rem;
        font-weight: 800;
        color: var(--bau-black) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.65rem;
    }

    .settings-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
    }

    @media (max-width: 900px) {
        .settings-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def apply_all_styles() -> None:
    apply_base_styles()
    apply_drift_view_styles()
    apply_chat_view_styles()

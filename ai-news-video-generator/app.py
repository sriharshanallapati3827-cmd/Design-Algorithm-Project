"""
AI NEWS Video Generator — Streamlit Frontend
==============================================
Main application file implementing the UI from FRONTEND_SPEC.md.

Run with:  streamlit run app.py
"""

import time
import io
import streamlit as st
from utils import (
    calculate_scene_range,
    format_timestamp,
    generate_demo_scenes,
    load_demo_article,
    _create_placeholder_image,
)
from ingestion import extract_text_from_url, extract_text_from_pdf, clean_text_input
from director import generate_storyboard, MODEL_MAP

# Phase 3 — Local GPU image engine (optional; falls back to placeholders)
try:
    from generator import (
        generate_scene_image,
        image_to_bytes,
        get_gpu_info,
        get_pipeline,
    )
    GPU_ENGINE_AVAILABLE = True
except ImportError:
    GPU_ENGINE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI NEWS Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for the spec's dark-mode palette & gold accents
st.markdown(
    """
    <style>
    /* ---- Google Font Import ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---- Global overrides & Design Tokens ---- */
    :root {
        --bg-primary: #0B0F17;
        --bg-secondary: #121824;
        --bg-card: #161F30;
        --accent: #F5A623;
        --accent-hover: #FFAF38;
        --text-primary: #F0F4F8;
        --text-muted: #94A3B8;
        --border-color: #1E293B;
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Global typography application (preserving Streamlit icon fonts) */
    html, body, .stApp {
        font-family: var(--font-family);
        letter-spacing: -0.01em;
    }

    p, h1, h2, h3, h4, h5, h6, input, textarea, select, .scene-badge, .pro-tips, .warning-note {
        font-family: var(--font-family) !important;
    }

    /* Preserve Material Symbols & Streamlit icon ligatures */
    span[data-testid="stIconMaterial"],
    [class*="material-symbols"],
    [class*="material-icons"],
    button[data-testid="collapsedControl"] *,
    button[kind="header"] *,
    [data-testid="stSidebarCollapseButton"] * {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
    }

    /* Clean, modern styling for the collapsed sidebar double arrow */
    div[data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        z-index: 99999 !important;
        margin: 10px 0 0 10px !important;
    }

    button[data-testid="collapsedControl"],
    div[data-testid="stSidebarCollapsedControl"] button {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 6px 10px !important;
        cursor: pointer !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35) !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    button[data-testid="collapsedControl"]:hover,
    div[data-testid="stSidebarCollapsedControl"] button:hover {
        background-color: var(--bg-card) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 0 14px rgba(245, 166, 35, 0.3) !important;
    }

    button[data-testid="collapsedControl"] span,
    div[data-testid="stSidebarCollapsedControl"] button span {
        color: var(--accent) !important;
        font-size: 1.3rem !important;
    }

    /* Collapse button when sidebar is open */
    div[data-testid="stSidebarCollapseButton"] button {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stSidebarCollapseButton"] button:hover {
        border-color: var(--border-color) !important;
        background-color: var(--bg-primary) !important;
    }

    div[data-testid="stSidebarCollapseButton"] button span {
        color: var(--accent) !important;
    }

    p, span:not([data-testid="stIconMaterial"]), li, div {
        line-height: 1.55;
    }

    /* Full width flexible main containers */
    .stApp {
        background-color: var(--bg-primary);
        width: 100%;
        max-width: 100%;
        overflow-x: hidden;
    }

    .main {
        width: 100%;
        overflow-x: hidden;
    }

    .main .block-container {
        padding: clamp(1rem, 3vw, 2.5rem) clamp(0.75rem, 2.5vw, 2rem) !important;
        max-width: 100% !important;
        width: 100% !important;
        overflow-x: hidden;
        min-height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }

    /* ------------------------------------------------------------- */
    /* 2. Sidebar Transitions & Layout                              */
    /* ------------------------------------------------------------- */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1),
                    width 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: clamp(1rem, 2.5vw, 2rem) !important;
        padding-left: clamp(0.75rem, 2vw, 1.25rem) !important;
        padding-right: clamp(0.75rem, 2vw, 1.25rem) !important;
    }

    /* ------------------------------------------------------------- */
    /* 3. Input Type Container (Flexbox)                             */
    /* ------------------------------------------------------------- */
    div[data-testid="stTabs"] [role="tablist"],
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: clamp(8px, 1.5vw, 16px) !important;
        width: 100% !important;
        flex-wrap: wrap !important;
        border-bottom: 1px solid var(--border-color) !important;
        padding-bottom: 6px !important;
        margin-bottom: 12px !important;
    }

    div[data-testid="stTabs"] [role="tab"],
    .stTabs [data-baseweb="tab"] {
        flex: 1 1 auto !important;
        min-width: 75px !important;
        text-align: center !important;
        justify-content: center !important;
        padding: clamp(6px, 1.2vw, 10px) clamp(8px, 1.5vw, 14px) !important;
        border-radius: 6px !important;
        transition: all 0.2s ease-in-out !important;
        font-size: clamp(0.76rem, 1vw + 0.45rem, 0.88rem) !important;
        font-weight: 500 !important;
    }

    /* Flexible components with percentage widths */
    div[data-testid="stTextInput"],
    div[data-testid="stTextArea"],
    div[data-testid="stFileUploader"],
    div[data-testid="stSelectbox"],
    div[data-testid="stSlider"],
    div[data-testid="stButton"] {
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* ------------------------------------------------------------- */
    /* 4. Fluid Typography & Styled Components                       */
    /* ------------------------------------------------------------- */
    /* Brand header */
    .brand-header {
        text-align: center;
        padding: clamp(0.5rem, 2vw, 1rem) 0 clamp(0.25rem, 1vw, 0.5rem);
        width: 100%;
    }
    .brand-header h1 {
        color: var(--accent) !important;
        font-size: clamp(1.15rem, 2.5vw + 0.5rem, 1.6rem) !important;
        font-weight: 700 !important;
        letter-spacing: clamp(1px, 0.2vw, 2px);
        margin-bottom: 0 !important;
        word-break: break-word;
    }
    .brand-header p {
        color: var(--text-muted);
        font-size: clamp(0.72rem, 1.2vw + 0.35rem, 0.84rem);
        margin-top: 4px;
        font-weight: 400;
    }

    /* Scene estimation badge */
    .scene-badge {
        background: var(--bg-primary);
        border: 1px solid var(--accent);
        border-radius: 6px;
        padding: clamp(6px, 1.5vw, 10px) clamp(10px, 2vw, 16px);
        text-align: center;
        color: var(--accent);
        font-weight: 600;
        font-size: clamp(0.78rem, 1.2vw + 0.45rem, 0.92rem);
        margin-top: 4px;
        width: 100%;
        box-sizing: border-box;
    }

    /* Pro-tips container */
    .pro-tips {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: clamp(10px, 2vw, 16px);
        margin-top: 12px;
        width: 100%;
        box-sizing: border-box;
    }
    .pro-tips h4 {
        color: var(--accent) !important;
        font-size: clamp(0.78rem, 1.2vw + 0.45rem, 0.9rem) !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
    .pro-tips li {
        color: var(--text-muted);
        font-size: clamp(0.72rem, 1vw + 0.4rem, 0.84rem);
        margin-bottom: 4px;
        line-height: 1.5;
    }

    /* Warning note */
    .warning-note {
        background: rgba(245, 166, 35, 0.08);
        border-left: 3px solid var(--accent);
        padding: clamp(6px, 1.5vw, 10px) clamp(8px, 2vw, 14px);
        font-size: clamp(0.72rem, 1vw + 0.4rem, 0.84rem);
        color: var(--text-muted);
        border-radius: 0 6px 6px 0;
        margin-top: 6px;
        width: 100%;
        box-sizing: border-box;
        line-height: 1.5;
    }

    /* Top Navigation Header & DEPLOY button */
    .top-deploy-nav {
        position: fixed;
        top: 14px;
        right: 24px;
        z-index: 99999;
        display: flex;
        align-items: center;
    }

    .deploy-btn {
        font-family: var(--font-family) !important;
        color: #FFFFFF !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
        padding: 5px 12px !important;
        border-radius: 4px !important;
        background: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        transition: all 0.2s ease !important;
    }

    .deploy-btn:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: rgba(245, 166, 35, 0.08) !important;
    }

    /* Centered Empty state in both height and width */
    .empty-state-container {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        min-height: calc(75vh - 40px) !important;
        width: 100% !important;
        text-align: center !important;
        margin: 0 auto !important;
    }

    .empty-state {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        padding: clamp(1.5rem, 4vw, 3rem) clamp(1rem, 3vw, 2rem) !important;
        width: 100% !important;
        max-width: 600px !important;
        margin: 0 auto !important;
        box-sizing: border-box !important;
    }
    .empty-state .icon {
        font-size: clamp(2.5rem, 6vw, 4rem);
        margin-bottom: clamp(0.75rem, 2vw, 1.25rem);
    }
    .empty-state h2 {
        color: var(--accent) !important;
        letter-spacing: clamp(1px, 0.3vw, 2.5px);
        font-size: clamp(1.3rem, 2.8vw + 0.5rem, 1.95rem) !important;
        font-weight: 800 !important;
        line-height: 1.25 !important;
        text-shadow: 0 0 24px rgba(245, 166, 35, 0.35), 0 0 48px rgba(245, 166, 35, 0.15);
        word-break: break-word;
        margin-bottom: 0.75rem !important;
    }
    .empty-state p {
        color: var(--text-muted);
        max-width: 520px;
        width: 100%;
        margin: 0 auto;
        font-size: clamp(0.85rem, 1.2vw + 0.45rem, 1rem);
        line-height: 1.6;
    }

    /* Scene card */
    .scene-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: clamp(12px, 2.5vw, 18px);
        margin-bottom: clamp(12px, 2.5vw, 18px);
        width: 100%;
        box-sizing: border-box;
    }
    .scene-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
        width: 100%;
    }
    .scene-card-header h3 {
        color: var(--accent) !important;
        font-size: clamp(0.92rem, 1.5vw + 0.45rem, 1.1rem) !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    .scene-card-header span {
        color: var(--text-muted);
        font-size: clamp(0.75rem, 1vw + 0.45rem, 0.88rem);
    }

    /* Loading pipeline steps */
    .pipeline-step {
        padding: clamp(4px, 1vw, 8px) 0;
        font-size: clamp(0.8rem, 1.2vw + 0.45rem, 0.95rem);
    }
    .pipeline-step.done {
        color: #4CAF50;
    }
    .pipeline-step.active {
        color: var(--accent);
    }
    .pipeline-step.pending {
        color: var(--text-muted);
    }

    /* ------------------------------------------------------------- */
    /* 1. Mobile Media Queries (<= 768px)                            */
    /* ------------------------------------------------------------- */
    @media (max-width: 768px) {
        /* Sidebar full width when active on mobile */
        section[data-testid="stSidebar"] {
            width: 100% !important;
            max-width: 100% !important;
        }

        /* Streamlit columns stack vertically on mobile */
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 1rem !important;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }

        /* Adjust scene card layout on mobile */
        .scene-card-header {
            flex-direction: column;
            align-items: flex-start;
        }

        /* Full width buttons */
        div[data-testid="stButton"] button {
            width: 100% !important;
        }

        /* Compact padding on mobile */
        .empty-state {
            padding: 2.5rem 0.75rem;
        }
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: var(--bg-primary) !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "scenes" not in st.session_state:
    st.session_state.scenes = []
if "generating" not in st.session_state:
    st.session_state.generating = False
if "demo_text" not in st.session_state:
    st.session_state.demo_text = ""
if "generation_done" not in st.session_state:
    st.session_state.generation_done = False
# Persisted inputs — survive st.rerun() during generation
if "input_url" not in st.session_state:
    st.session_state.input_url = ""
if "input_pdf" not in st.session_state:
    st.session_state.input_pdf = None
if "input_raw_text" not in st.session_state:
    st.session_state.input_raw_text = ""
if "input_duration" not in st.session_state:
    st.session_state.input_duration = 90
if "input_model" not in st.session_state:
    st.session_state.input_model = "Gemini 2.5 Flash"
if "article_text" not in st.session_state:
    st.session_state.article_text = ""
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "saved_api_key" not in st.session_state:
    st.session_state.saved_api_key = ""

# ---------------------------------------------------------------------------
# LEFT SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    # ---- A. Brand Header ----
    st.markdown(
        """
        <div class="brand-header">
            <h1>🎬 AI NEWS Video Generator</h1>
            <p>Automated News Video Pipeline</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ---- B. Multi-Source Ingestion ----
    st.markdown("##### 📥 News Source Input")

    tab_url, tab_pdf, tab_text = st.tabs(["🔗 URL", "📄 PDF Upload", "📝 Raw Text"])

    with tab_url:
        url_input = st.text_input(
            "Article URL",
            placeholder="https://example.com/news-article",
            label_visibility="collapsed",
        )

    with tab_pdf:
        pdf_file = st.file_uploader(
            "Upload a PDF article",
            type=["pdf"],
            help="Accepted: .pdf up to 5MB (~1–5 pages)",
            label_visibility="collapsed",
        )
        st.caption("Accepted: .pdf up to 5MB (~1–5 pages)")
        if pdf_file is not None and pdf_file.size > 5 * 1024 * 1024:
            st.error("⚠️ File size exceeds 5MB limit. Please upload a smaller PDF.")

    with tab_text:
        raw_text = st.text_area(
            "Paste article text",
            value=st.session_state.demo_text,
            height=160,
            placeholder="Paste your news article here…",
            label_visibility="collapsed",
        )

    # Constraint warning
    st.markdown(
        '<div class="warning-note">⚠️ Long articles will automatically be '
        "summarized into a concise news script tailored for a 1–2 minute "
        "target video.</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- C. Duration Control ----
    st.markdown("##### ⏱️ Target Duration")
    duration = st.slider(
        "Duration (seconds)",
        min_value=60,
        max_value=120,
        value=90,
        step=5,
        format="%d sec",
        label_visibility="collapsed",
    )

    min_sc, max_sc = calculate_scene_range(duration)
    st.markdown(
        f'<div class="scene-badge">🎞️ Estimated Scenes: '
        f"{min_sc} – {max_sc} Frames</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- E. Director Model Selector ----
    st.markdown("##### 🤖 Director Model")
    model = st.selectbox(
        "Select model",
        options=[
            "Gemini 3.6 Flash",
            "Gemini 3.5 Flash Lite",
            "Gemini 2.5 Flash",
            "Gemini 1.5 Flash",
            "Claude 3.5 Sonnet",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # ---- API Key ----
    st.markdown("##### 🔑 Gemini API Key")
    sidebar_api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste key here (overrides .env)",
        label_visibility="collapsed",
        help="Optional. If left blank, the key from .env / GEMINI_API_KEY env var is used.",
        key="api_key_input",
    )

    st.divider()

    # ---- F. Action Buttons ----
    generate_clicked = st.button(
        "🎬  GENERATE AI NEWS STORYBOARD",
        use_container_width=True,
        type="primary",
    )

    demo_clicked = st.button(
        "📰  LOAD DEMO NEWS ARTICLE",
        use_container_width=True,
    )

    # ---- Pro Tips ----
    st.markdown(
        """
        <div class="pro-tips">
            <h4>💡 PRO TIPS</h4>
            <ul>
                <li>Focus on key headlines for sharper scenes</li>
                <li>Use 1080p landscape prompts for best visuals</li>
                <li>Shorter articles (≤500 words) yield tighter scripts</li>
                <li>Try different director models for varied styles</li>
                <li>Edit voiceover scripts before final rendering</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- GPU Status Indicator ----
    st.divider()
    st.markdown("##### 🖥️ GPU Engine Status")
    if GPU_ENGINE_AVAILABLE:
        gpu = get_gpu_info()
        if gpu["cuda_available"]:
            st.success(f"✅ **{gpu['device_name']}**", icon="🟢")
            st.caption(
                f"VRAM: {gpu['vram_used_mb']} / {gpu['vram_total_mb']} MB  \n"
                f"Model: {gpu['model_loaded'] or 'Not loaded yet'}"
            )
        else:
            st.warning("⚠️ CUDA not available — CPU mode", icon="🟡")
            st.caption("Images will generate slowly on CPU.")
    else:
        st.info("📦 GPU engine not installed", icon="ℹ️")
        st.caption("Install torch + diffusers for AI images.")

# ---------------------------------------------------------------------------
# Handle button actions
# ---------------------------------------------------------------------------

if demo_clicked:
    st.session_state.demo_text = load_demo_article()
    st.session_state.scenes = []
    st.session_state.generation_done = False
    st.session_state.last_error = ""
    st.rerun()

if generate_clicked:
    # Claude toast guard
    if model == "Claude 3.5 Sonnet":
        st.toast("⚠️ Anthropic models are currently inactive. Please select a Gemini model.", icon="🚫")
    else:
        # Check that at least one input source has content
        has_input = bool(url_input) or bool(pdf_file) or bool(raw_text)
        if not has_input:
            st.sidebar.error("Please provide a URL, PDF, or text before generating.")
        elif pdf_file is not None and pdf_file.size > 5 * 1024 * 1024:
            st.sidebar.error("PDF file exceeds the 5MB limit. Please upload a smaller file.")
        else:
            # Persist inputs into session state so they survive the rerun
            st.session_state.input_url = url_input or ""
            st.session_state.input_pdf = pdf_file
            st.session_state.input_raw_text = raw_text or ""
            st.session_state.input_duration = duration
            st.session_state.input_model = model
            st.session_state.saved_api_key = sidebar_api_key or ""
            st.session_state.last_error = ""
            st.session_state.generating = True
            st.session_state.generation_done = False
            st.session_state.scenes = []
            st.rerun()

# ---------------------------------------------------------------------------
# MAIN WORKSPACE
# ---------------------------------------------------------------------------

if st.session_state.generating:
    # ---- Loading / Pipeline State ----
    st.markdown("## ⚙️ Generating Your News Storyboard…")

    progress_bar = st.progress(0)
    status_container = st.empty()

    pipeline_labels = [
        "📥 Ingesting source content…",
        "📝 Generating news script via Gemini…",
        "🎨 Generating AI scene images…" if GPU_ENGINE_AVAILABLE else "🖼️ Preparing visual placeholders…",
        "✅ Assembling storyboard…",
    ]
    num_steps = len(pipeline_labels)

    def _show_pipeline(active_idx: int):
        """Render the pipeline step indicators."""
        lines = []
        for j, label in enumerate(pipeline_labels):
            if j < active_idx:
                lines.append(f'<div class="pipeline-step done">✅ {label}</div>')
            elif j == active_idx:
                lines.append(f'<div class="pipeline-step active">⏳ {label}</div>')
            else:
                lines.append(f'<div class="pipeline-step pending">⬜ {label}</div>')
        status_container.markdown("".join(lines), unsafe_allow_html=True)

    error_occurred = False

    # ── Step 1: Ingest ────────────────────────────────────────────────────
    _show_pipeline(0)
    progress_bar.progress(1 / num_steps)

    try:
        saved_url = st.session_state.input_url
        saved_pdf = st.session_state.input_pdf
        saved_raw = st.session_state.input_raw_text

        if saved_url:
            article_text = extract_text_from_url(saved_url)
        elif saved_pdf is not None:
            article_text = extract_text_from_pdf(saved_pdf)
        elif saved_raw:
            article_text = clean_text_input(saved_raw)
        else:
            raise ValueError("No input content found. Please provide a URL, PDF, or text.")

        st.session_state.article_text = article_text

    except (ValueError, Exception) as exc:
        st.session_state.last_error = f"Ingestion error: {exc}"
        error_occurred = True

    # ── Step 2: Generate storyboard via Gemini ────────────────────────────
    if not error_occurred:
        _show_pipeline(1)
        progress_bar.progress(2 / num_steps)

        try:
            api_key = st.session_state.saved_api_key if st.session_state.saved_api_key else None
            scenes_raw = generate_storyboard(
                article_text=st.session_state.article_text,
                duration_sec=st.session_state.input_duration,
                api_key=api_key,
                model_name=st.session_state.input_model,
            )
        except (ValueError, RuntimeError, Exception) as exc:
            st.session_state.last_error = f"Gemini Director error: {exc}"
            error_occurred = True

    # ── Step 3: Generate AI images (or placeholders) ────────────────────────
    if not error_occurred:
        _show_pipeline(2)
        progress_bar.progress(3 / num_steps)

        scenes = []
        total_scenes = len(scenes_raw)
        for i, s in enumerate(scenes_raw):
            visual_prompt = s.get("visual_prompt", "")
            scene_num = s.get("scene_number", i + 1)
            img_bytes = None

            # Try AI generation if GPU engine is available
            if GPU_ENGINE_AVAILABLE and visual_prompt:
                try:
                    st.toast(
                        f"🎨 Rendering scene {scene_num}/{total_scenes}…",
                        icon="🖼️",
                    )
                    ai_image = generate_scene_image(prompt=visual_prompt)
                    if ai_image is not None:
                        img_bytes = image_to_bytes(ai_image)
                except Exception as exc:
                    st.toast(
                        f"⚠️ Scene {scene_num} AI render failed, using placeholder.",
                        icon="⚠️",
                    )

            # Fallback to placeholder
            if img_bytes is None:
                img_bytes = _create_placeholder_image(scene_num)

            scenes.append({
                "scene_number": scene_num,
                "timestamp": s.get("timestamp", ""),
                "voiceover": s.get("narration", ""),
                "visual_prompt": visual_prompt,
                "image_bytes": img_bytes,
            })

        st.session_state.scenes = scenes

    # ── Step 4: Done ──────────────────────────────────────────────────────
    if not error_occurred:
        _show_pipeline(3)
        progress_bar.progress(1.0)
        time.sleep(0.4)

    st.session_state.generating = False

    if error_occurred:
        st.session_state.generation_done = False
    else:
        st.session_state.generation_done = True

    st.rerun()

elif st.session_state.generation_done and st.session_state.scenes:
    # ---- Generated Scene Grid ----
    st.markdown("## 🎬 News Storyboard Timeline")
    st.caption(
        f"{len(st.session_state.scenes)} scenes · "
        f"{duration} seconds · Model: {model}"
    )
    st.divider()

    for scene in st.session_state.scenes:
        with st.container():
            st.markdown(
                f"""
                <div class="scene-card">
                    <div class="scene-card-header">
                        <h3>🎞️ SCENE {scene['scene_number']}</h3>
                        <span>{scene['timestamp']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_img, col_details = st.columns([1, 1.4])

            with col_img:
                # Visual Frame
                if scene.get("image_bytes"):
                    st.image(
                        scene["image_bytes"],
                        caption=f"Scene {scene['scene_number']} — AI Generated",
                        use_container_width=True,
                    )
                else:
                    st.info("🖼️ Image placeholder (Pillow not installed)")

                rerender_key = f"rerender_{scene['scene_number']}"
                if GPU_ENGINE_AVAILABLE:
                    if st.button(
                        "🔄 Re-render Image",
                        key=rerender_key,
                        use_container_width=True,
                    ):
                        with st.spinner(f"🎨 Re-rendering scene {scene['scene_number']}…"):
                            new_image = generate_scene_image(prompt=scene["visual_prompt"])
                            if new_image is not None:
                                scene["image_bytes"] = image_to_bytes(new_image)
                                st.toast(f"✅ Scene {scene['scene_number']} re-rendered!", icon="🎨")
                                st.rerun()
                            else:
                                st.toast(f"⚠️ Re-render failed for scene {scene['scene_number']}.", icon="⚠️")
                else:
                    st.button(
                        "🔄 Re-render Image",
                        key=rerender_key,
                        use_container_width=True,
                        disabled=True,
                        help="Install torch + diffusers to enable AI re-rendering",
                    )

            with col_details:
                # Voiceover Script (editable)
                st.markdown("**📝 Voiceover Script**")
                st.text_area(
                    "Voiceover",
                    value=scene["voiceover"],
                    height=100,
                    key=f"vo_{scene['scene_number']}",
                    label_visibility="collapsed",
                )

                # Visual Prompt
                st.markdown("**🎨 Visual Prompt**")
                st.code(scene["visual_prompt"], language="text")

                # Audio Preview (placeholder)
                st.markdown("**🔊 Audio Preview**")
                st.info(
                    "🎧 Audio preview will appear here after TTS generation.",
                    icon="🔊",
                )

            st.divider()

else:
    # ---- Error State (if any) ----
    if st.session_state.last_error:
        st.error(f"⚠️ **Generation Failed:** {st.session_state.last_error}")
        col_err1, col_err2 = st.columns([1, 1])
        with col_err1:
            if st.button("💡 Render with Demo Storyboard Instead", use_container_width=True):
                min_s, _ = calculate_scene_range(duration)
                st.session_state.scenes = generate_demo_scenes(min_s, duration)
                st.session_state.generation_done = True
                st.session_state.last_error = ""
                st.rerun()
        with col_err2:
            if st.button("✕ Dismiss", use_container_width=True):
                st.session_state.last_error = ""
                st.rerun()
        st.divider()

    # ---- Empty State ----
    st.markdown(
        """
        <div class="top-deploy-nav">
            <span class="deploy-btn">DEPLOY</span>
        </div>
        <div class="empty-state-container">
            <div class="empty-state">
                <div class="icon">🎬</div>
                <!-- YOUR NEWS STORYBOARD AWAITS -->
                <h2>YOUR NEWS STORYBOARD<br>AWAITS</h2>
                <p>
                    Enter a URL, PDF, or text on the left, pick your duration, and watch<br>
                    your video scenes generate in real-time.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

"""
AI NEWS Generator — Streamlit Frontend
========================================
Main application file implementing the UI from FRONTEND_SPEC.md.

Run with:  streamlit run app.py
"""

import time
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

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI NEWS Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for the spec's dark-mode palette & gold accents
st.markdown(
    """
    <style>
    /* ---- Global overrides ---- */
    :root {
        --bg-primary: #0B0F17;
        --bg-secondary: #121824;
        --accent: #F5A623;
        --text-primary: #EAEAEA;
        --text-muted: #8892A4;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
    }

    /* Brand header */
    .brand-header {
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .brand-header h1 {
        color: var(--accent) !important;
        font-size: 1.55rem !important;
        letter-spacing: 2px;
        margin-bottom: 0 !important;
    }
    .brand-header p {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin-top: 2px;
    }

    /* Scene estimation badge */
    .scene-badge {
        background: var(--bg-primary);
        border: 1px solid var(--accent);
        border-radius: 6px;
        padding: 8px 14px;
        text-align: center;
        color: var(--accent);
        font-weight: 600;
        font-size: 0.88rem;
        margin-top: 4px;
    }

    /* Pro-tips container */
    .pro-tips {
        background: var(--bg-primary);
        border: 1px solid #1E2636;
        border-radius: 8px;
        padding: 14px 16px;
        margin-top: 12px;
    }
    .pro-tips h4 {
        color: var(--accent) !important;
        font-size: 0.85rem !important;
        margin-bottom: 8px !important;
    }
    .pro-tips li {
        color: var(--text-muted);
        font-size: 0.8rem;
        margin-bottom: 4px;
    }

    /* Warning note */
    .warning-note {
        background: rgba(245,166,35,0.08);
        border-left: 3px solid var(--accent);
        padding: 8px 12px;
        font-size: 0.8rem;
        color: var(--text-muted);
        border-radius: 0 6px 6px 0;
        margin-top: 6px;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 6rem 2rem;
    }
    .empty-state .icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    .empty-state h2 {
        color: var(--accent) !important;
        letter-spacing: 2px;
        font-size: 1.6rem !important;
    }
    .empty-state p {
        color: var(--text-muted);
        max-width: 540px;
        margin: 0.8rem auto 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Scene card */
    .scene-card {
        background: var(--bg-secondary);
        border: 1px solid #1E2636;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 18px;
    }
    .scene-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .scene-card-header h3 {
        color: var(--accent) !important;
        font-size: 1rem !important;
        margin: 0 !important;
    }
    .scene-card-header span {
        color: var(--text-muted);
        font-size: 0.85rem;
    }

    /* Loading pipeline steps */
    .pipeline-step {
        padding: 6px 0;
        font-size: 0.9rem;
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
            <h1>🎬 AI NEWS Generator</h1>
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
        "🖼️ Preparing visual placeholders…",
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

    # ── Step 3: Build scene objects with placeholders ─────────────────────
    if not error_occurred:
        _show_pipeline(2)
        progress_bar.progress(3 / num_steps)

        scenes = []
        for i, s in enumerate(scenes_raw):
            scenes.append({
                "scene_number": s.get("scene_number", i + 1),
                "timestamp": s.get("timestamp", ""),
                "voiceover": s.get("narration", ""),
                "visual_prompt": s.get("visual_prompt", ""),
                "image_bytes": _create_placeholder_image(i + 1),
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

                st.button(
                    "🔄 Re-render Image",
                    key=f"rerender_{scene['scene_number']}",
                    use_container_width=True,
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
        <div class="empty-state">
            <div class="icon">🎬</div>
            <h2>YOUR NEWS STORYBOARD AWAITS</h2>
            <p>
                Enter a URL, PDF, or text on the left, pick your duration,
                and watch your video scenes generate in real-time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

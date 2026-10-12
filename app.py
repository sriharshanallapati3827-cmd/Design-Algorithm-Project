"""
AI NEWS Video Generator — Streamlit Frontend
==============================================
Main application file implementing the UI from FRONTEND_SPEC.md.
Features real-time WebSocket heartbeat yielding, resilient cloud execution,
dual-mode visual engine (Local GPU / Pollinations Cloud), Neural TTS,
and automated MoviePy 2.x video assembly.

Run with:  streamlit run app.py
"""

import os
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

# Phase 3 — Visual engine (Local GPU or Pollinations Cloud API)
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

# Phase 4 — Neural TTS audio engine
try:
    from audio import (
        generate_scene_audio,
        get_voice_options,
        VOICES,
        DEFAULT_VOICE,
    )
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    DEFAULT_VOICE = "en-IN-NeerjaNeural"
    VOICES = {
        "Female – Indian News Anchor (Neerja)": "en-IN-NeerjaNeural",
        "Female – Expressive Indian Anchor (Neerja Expressive)": "en-IN-NeerjaExpressiveNeural",
        "Male – Indian News Anchor (Prabhat)": "en-IN-PrabhatNeural",
    }

    def get_voice_options():
        return dict(VOICES)

# Phase 5 — Video rendering engine
try:
    from video_engine import (
        assemble_full_video,
        is_available as _video_engine_available,
    )
    VIDEO_ENGINE_AVAILABLE = _video_engine_available()
except ImportError:
    VIDEO_ENGINE_AVAILABLE = False

    def assemble_full_video(*_a, **_kw):
        raise ImportError("moviepy is not installed. Run: pip install moviepy imageio-ffmpeg")

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI NEWS Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

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

    html, body, .stApp {
        font-family: var(--font-family);
        letter-spacing: -0.01em;
    }

    p, h1, h2, h3, h4, h5, h6, input, textarea, select, .scene-badge, .pro-tips, .warning-note {
        font-family: var(--font-family) !important;
    }

    span[data-testid="stIconMaterial"],
    [class*="material-symbols"],
    [class*="material-icons"],
    button[data-testid="collapsedControl"] *,
    button[kind="header"] *,
    [data-testid="stSidebarCollapseButton"] * {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
    }

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

    .stApp {
        background-color: var(--bg-primary);
        width: 100%;
        max-width: 100%;
        overflow-x: hidden;
    }

    .main .block-container {
        padding: clamp(1rem, 3vw, 2.5rem) clamp(0.75rem, 2.5vw, 2rem) !important;
        max-width: 100% !important;
        width: 100% !important;
        overflow-x: hidden;
        min-height: 100vh !important;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
    }

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
    }
    .brand-header p {
        color: var(--text-muted);
        font-size: clamp(0.72rem, 1.2vw + 0.35rem, 0.84rem);
        margin-top: 4px;
    }

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
        font-size: clamp(1.3rem, 2.8vw + 0.5rem, 1.95rem) !important;
        font-weight: 800 !important;
        line-height: 1.25 !important;
        text-shadow: 0 0 24px rgba(245, 166, 35, 0.35);
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

    .pipeline-step {
        padding: clamp(4px, 1vw, 8px) 0;
        font-size: clamp(0.8rem, 1.2vw + 0.45rem, 0.95rem);
    }
    .pipeline-step.done { color: #4CAF50; }
    .pipeline-step.active { color: var(--accent); }
    .pipeline-step.pending { color: var(--text-muted); }

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
if "input_url" not in st.session_state:
    st.session_state.input_url = ""
if "input_pdf" not in st.session_state:
    st.session_state.input_pdf = None
if "input_raw_text" not in st.session_state:
    st.session_state.input_raw_text = ""
if "input_duration" not in st.session_state:
    st.session_state.input_duration = 90
if "input_model" not in st.session_state:
    st.session_state.input_model = "Gemini 3.6 Flash"
if "article_text" not in st.session_state:
    st.session_state.article_text = ""
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "audio_voice" not in st.session_state:
    st.session_state.audio_voice = DEFAULT_VOICE
if "scene_audio" not in st.session_state:
    st.session_state.scene_audio = {}
if "video_bytes" not in st.session_state:
    st.session_state.video_bytes = None
if "video_render_error" not in st.session_state:
    st.session_state.video_render_error = ""

# ---------------------------------------------------------------------------
# LEFT SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
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

    st.markdown(
        '<div class="warning-note">⚠️ Long articles will automatically be '
        "summarized into a concise news script tailored for a 1–2 minute "
        "target video.</div>",
        unsafe_allow_html=True,
    )

    st.divider()

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

    st.markdown("##### 🤖 Director Model")
    model = st.selectbox(
        "Select model",
        options=[
            "Gemini 3.6 Flash",
            "Gemini 3.5 Flash Lite",
            "Claude 3.5 Sonnet",
        ],
        label_visibility="collapsed",
        help="Gemini 3.6 Flash is primary; Gemini 3.5 Flash Lite and Groq Qwen act as fallbacks.",
    )

    st.divider()

    st.markdown("##### 🎙️ Voiceover Voice")
    _voice_options = get_voice_options()
    _voice_display_names = list(_voice_options.keys())
    _current_voice_id = st.session_state.audio_voice
    _default_idx = next(
        (i for i, v in enumerate(_voice_options.values()) if v == _current_voice_id),
        0,
    )
    _selected_voice_name = st.selectbox(
        "Select voiceover voice",
        options=_voice_display_names,
        index=_default_idx,
        label_visibility="collapsed",
        disabled=not TTS_AVAILABLE,
    )
    _selected_voice_id = _voice_options[_selected_voice_name]
    if _selected_voice_id != st.session_state.audio_voice:
        st.session_state.audio_voice = _selected_voice_id
        st.session_state.scene_audio = {}

    if TTS_AVAILABLE:
        st.caption(f"🎙️ {_selected_voice_name}")
    else:
        st.caption("⚠️ edge-tts not installed — audio disabled")

    st.divider()

    generate_clicked = st.button(
        "🎬  GENERATE AI NEWS STORYBOARD",
        use_container_width=True,
        type="primary",
    )

    demo_clicked = st.button(
        "📰  LOAD DEMO NEWS ARTICLE",
        use_container_width=True,
    )

    # Status Indicator
    st.divider()
    st.markdown("##### 🖥️ Visual Engine Status")
    if GPU_ENGINE_AVAILABLE:
        gpu = get_gpu_info()
        if gpu.get("cuda_available"):
            st.success(f"✅ **{gpu['device_name']} (Local GPU)**", icon="🟢")
            st.caption(f"VRAM: {gpu['vram_used_mb']} / {gpu['vram_total_mb']} MB")
        else:
            st.info(f"☁️ **{gpu.get('device_name', 'Cloud Pollinations API')}**", icon="🌐")
            st.caption("Streaming AI visuals from cloud API.")
    else:
        st.info("📦 Visual engine using placeholder generator", icon="ℹ️")

# ---------------------------------------------------------------------------
# Button Actions
# ---------------------------------------------------------------------------

if demo_clicked:
    st.session_state.demo_text = load_demo_article()
    st.session_state.scenes = []
    st.session_state.generation_done = False
    st.session_state.last_error = ""
    st.session_state.scene_audio = {}
    st.session_state.video_bytes = None
    st.session_state.video_render_error = ""
    st.rerun()

if generate_clicked:
    if model == "Claude 3.5 Sonnet":
        st.toast("⚠️ Anthropic models are currently inactive. Please select a Gemini model.", icon="🚫")
    else:
        has_input = bool(url_input) or bool(pdf_file) or bool(raw_text)
        if not has_input:
            st.sidebar.error("Please provide a URL, PDF, or text before generating.")
        elif pdf_file is not None and pdf_file.size > 5 * 1024 * 1024:
            st.sidebar.error("PDF file exceeds the 5MB limit. Please upload a smaller file.")
        else:
            st.session_state.input_url = url_input or ""
            st.session_state.input_pdf = pdf_file
            st.session_state.input_raw_text = raw_text or ""
            st.session_state.input_duration = duration
            st.session_state.input_model = model
            st.session_state.last_error = ""
            st.session_state.generating = True
            st.session_state.generation_done = False
            st.session_state.scenes = []
            st.session_state.scene_audio = {}
            st.session_state.video_bytes = None
            st.session_state.video_render_error = ""
            st.rerun()

# ---------------------------------------------------------------------------
# MAIN WORKSPACE
# ---------------------------------------------------------------------------

if st.session_state.generating:
    st.markdown("## ⚙️ Generating Your News Storyboard…")

    progress_bar = st.progress(0.0)
    status_container = st.empty()

    pipeline_labels = [
        "📥 Ingesting source content…",
        "📝 Generating news script via Gemini…",
        "🎨 Generating AI scene visuals…",
        "✅ Assembling storyboard scenes…",
        "🎙️ Synthesizing neural voiceover…",
        "🎬 Rendering final MP4 video…",
    ]
    num_steps = len(pipeline_labels)

    def _show_pipeline(active_idx: int, detail: str = ""):
        lines = []
        for j, label in enumerate(pipeline_labels):
            if j < active_idx:
                lines.append(f'<div class="pipeline-step done">✅ {label}</div>')
            elif j == active_idx:
                lbl = f"{label} ({detail})" if detail else label
                lines.append(f'<div class="pipeline-step active">⏳ {lbl}</div>')
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
            raise ValueError("No input content found.")

        st.session_state.article_text = article_text

    except Exception as exc:
        st.session_state.last_error = f"Ingestion error: {exc}"
        error_occurred = True

    # ── Step 2: Generate storyboard ───────────────────────────────────────
    if not error_occurred:
        _show_pipeline(1)
        progress_bar.progress(2 / num_steps)

        try:
            api_key = os.getenv("GEMINI_API_KEY")
            scenes_raw = generate_storyboard(
                article_text=st.session_state.article_text,
                duration_sec=st.session_state.input_duration,
                api_key=api_key,
                model_name=st.session_state.input_model,
            )
        except Exception as exc:
            st.session_state.last_error = f"Gemini Director error: {exc}"
            error_occurred = True

    # ── Step 3: Visual Generation with Heartbeat Yielding ──────────────────
    if not error_occurred:
        scenes = []
        total_scenes = len(scenes_raw)

        for i, s in enumerate(scenes_raw):
            visual_prompt = s.get("visual_prompt", "")
            scene_num = s.get("scene_number", i + 1)
            img_bytes = None

            # Streamlit Heartbeat: update progress and status on EVERY iteration!
            sub_progress = (2.0 + (i / total_scenes)) / num_steps
            progress_bar.progress(min(sub_progress, 0.99))
            _show_pipeline(2, f"Rendering scene {scene_num}/{total_scenes}")

            if GPU_ENGINE_AVAILABLE and visual_prompt:
                try:
                    search_q = s.get("search_query") or visual_prompt
                    ai_image = generate_scene_image(prompt=visual_prompt, search_query=search_q)
                    if ai_image is not None:
                        img_bytes = image_to_bytes(ai_image)
                except Exception as _render_exc:
                    st.toast(f"⚠️ Scene {scene_num} render fallback: {_render_exc}", icon="⚠️")

            if img_bytes is None:
                img_bytes = _create_placeholder_image(scene_num)

            scenes.append({
                "scene_number": scene_num,
                "timestamp": s.get("timestamp", ""),
                "voiceover": s.get("narration", ""),
                "visual_prompt": visual_prompt,
                "search_query": s.get("search_query", visual_prompt),
                "image_bytes": img_bytes,
            })

        st.session_state.scenes = scenes

    # ── Step 4: Assembling storyboard done marker ─────────────────────────
    if not error_occurred:
        _show_pipeline(3)
        progress_bar.progress(4 / num_steps)
        time.sleep(0.1)

    # ── Step 5: Auto-generate TTS audio with Heartbeat Yielding ───────────
    if not error_occurred:
        _show_pipeline(4)
        progress_bar.progress(4.5 / num_steps)

        if TTS_AVAILABLE:
            _voice = st.session_state.audio_voice
            total_s = len(st.session_state.scenes)
            for idx, scene in enumerate(st.session_state.scenes):
                _snum = scene["scene_number"]
                _text = scene.get("voiceover", "")
                if not _text.strip():
                    continue

                sub_progress = (4.0 + (idx / total_s)) / num_steps
                progress_bar.progress(min(sub_progress, 0.99))
                _show_pipeline(4, f"Synthesizing voiceover {_snum}/{total_s}")

                try:
                    _audio = generate_scene_audio(text=_text, voice=_voice)
                    st.session_state.scene_audio[_snum] = _audio
                except Exception as _audio_exc:
                    st.toast(f"⚠️ Audio skipped for scene {_snum}: {_audio_exc}", icon="🔇")
        else:
            st.toast("⚠️ edge-tts not installed — skipping audio.", icon="🔇")

    # ── Step 6: Render MP4 with Real-Time Callback Heartbeat ──────────────
    if not error_occurred:
        _show_pipeline(5)
        progress_bar.progress(5.2 / num_steps)

        if VIDEO_ENGINE_AVAILABLE:
            import tempfile as _tempfile
            import video_engine as _ve

            _out_path = os.path.join(_tempfile.gettempdir(), "ai_news_output.mp4")

            def _vid_render_heartbeat(frac: float, label: str):
                overall_frac = (5.0 + frac * 0.95) / num_steps
                progress_bar.progress(min(overall_frac, 0.99))
                _show_pipeline(5, label)

            try:
                _ve.assemble_full_video(
                    scenes=st.session_state.scenes,
                    scene_audio=st.session_state.scene_audio,
                    output_path=_out_path,
                    progress_callback=_vid_render_heartbeat,
                )
                with open(_out_path, "rb") as _f:
                    st.session_state.video_bytes = _f.read()
                try:
                    os.unlink(_out_path)
                except OSError:
                    pass
            except Exception as _vid_exc:
                st.session_state.video_render_error = str(_vid_exc)
                st.toast(f"⚠️ Video assembly notice: {_vid_exc}", icon="⚠️")
        else:
            st.toast("📦 moviepy not installed — skipping video rendering.", icon="ℹ️")

    # ── Done ──────────────────────────────────────────────────────────────
    progress_bar.progress(1.0)
    time.sleep(0.3)

    st.session_state.generating = False
    st.session_state.generation_done = not error_occurred
    st.rerun()

elif st.session_state.generation_done and st.session_state.scenes:
    _total_scenes = len(st.session_state.scenes)
    _scenes_with_audio = [
        s["scene_number"]
        for s in st.session_state.scenes
        if st.session_state.scene_audio.get(s["scene_number"])
    ]

    st.markdown("## 🎬 Generated News Video")
    st.caption(
        f"{_total_scenes} scenes · {st.session_state.input_duration} seconds · "
        f"Model: {st.session_state.input_model}"
    )

    if st.session_state.video_bytes:
        st.video(st.session_state.video_bytes)

        _vid_col1, _vid_col2, _vid_col3 = st.columns([3, 2, 2])
        with _vid_col1:
            st.download_button(
                label="⬇️ Download MP4",
                data=st.session_state.video_bytes,
                file_name="news_video.mp4",
                mime="video/mp4",
                use_container_width=True,
                type="primary",
            )
        with _vid_col2:
            _rerender_clicked = st.button(
                "🔄 Re-render Updated Video",
                use_container_width=True,
                disabled=not VIDEO_ENGINE_AVAILABLE,
            )
        with _vid_col3:
            st.caption(
                f"🎙️ Audio: {len(_scenes_with_audio)}/{_total_scenes} scenes"
                if _scenes_with_audio
                else "🔇 No audio attached"
            )
    else:
        _rerender_clicked = False
        if st.session_state.video_render_error:
            st.error(f"⚠️ Auto-render notification: {st.session_state.video_render_error}")

        if VIDEO_ENGINE_AVAILABLE:
            _rerender_clicked = st.button("🎬 Render MP4 Video", use_container_width=True, type="primary")

    if _rerender_clicked and VIDEO_ENGINE_AVAILABLE:
        import tempfile as _tempfile
        import video_engine as _ve

        _render_progress = st.progress(0.0, text="Starting render…")
        _render_status = st.empty()

        def _manual_cb(frac: float, label: str):
            _render_progress.progress(min(frac, 1.0), text=label)
            _render_status.caption(label)

        for _sc in st.session_state.scenes:
            _key = f"vo_{_sc['scene_number']}"
            if _key in st.session_state:
                _sc["voiceover"] = st.session_state[_key]

        _out_path = os.path.join(_tempfile.gettempdir(), "ai_news_output.mp4")
        try:
            _ve.assemble_full_video(
                scenes=st.session_state.scenes,
                scene_audio=st.session_state.scene_audio,
                output_path=_out_path,
                progress_callback=_manual_cb,
            )
            with open(_out_path, "rb") as _f:
                st.session_state.video_bytes = _f.read()
            try:
                os.unlink(_out_path)
            except OSError:
                pass
            st.toast("✅ Video re-rendered!", icon="🎬")
            st.rerun()
        except Exception as _exc:
            st.session_state.video_render_error = str(_exc)
            st.error(f"⚠️ Render failed: {_exc}")

    # Scene Editor
    st.divider()
    st.markdown("### 🎞️ Scene-by-Scene Editor")
    st.caption("Edit voiceovers or regenerate images below, then re-render.")

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
                if scene.get("image_bytes"):
                    st.image(scene["image_bytes"], use_container_width=True)

                if GPU_ENGINE_AVAILABLE:
                    if st.button("🔄 Re-render Visual", key=f"rerender_{scene['scene_number']}", use_container_width=True):
                        with st.spinner("Rendering visual…"):
                            new_img = generate_scene_image(
                                prompt=scene["visual_prompt"],
                                search_query=scene.get("search_query")
                            )
                            if new_img is not None:
                                scene["image_bytes"] = image_to_bytes(new_img)
                                st.rerun()

            with col_details:
                st.markdown("**📝 Voiceover Script**")
                st.text_area(
                    "Voiceover",
                    value=scene["voiceover"],
                    height=100,
                    key=f"vo_{scene['scene_number']}",
                    label_visibility="collapsed",
                )

                st.markdown("**🎨 Visual Prompt**")
                st.code(scene["visual_prompt"], language="text")

                _snum = scene["scene_number"]
                _cached_audio = st.session_state.scene_audio.get(_snum)

                if TTS_AVAILABLE:
                    if _cached_audio:
                        st.audio(_cached_audio, format="audio/mp3")
                        if st.button("🔄 Regenerate Audio", key=f"audio_regen_{_snum}", use_container_width=True):
                            _live_text = st.session_state.get(f"vo_{_snum}", scene["voiceover"])
                            with st.spinner("Synthesizing audio…"):
                                try:
                                    st.session_state.scene_audio[_snum] = generate_scene_audio(
                                        text=_live_text,
                                        voice=st.session_state.audio_voice,
                                    )
                                    st.rerun()
                                except Exception as _exc:
                                    st.error(f"TTS error: {_exc}")
                    else:
                        if st.button("🎙️ Generate Audio", key=f"audio_gen_{_snum}", use_container_width=True):
                            _live_text = st.session_state.get(f"vo_{_snum}", scene["voiceover"])
                            with st.spinner("Synthesizing audio…"):
                                try:
                                    st.session_state.scene_audio[_snum] = generate_scene_audio(
                                        text=_live_text,
                                        voice=st.session_state.audio_voice,
                                    )
                                    st.rerun()
                                except Exception as _exc:
                                    st.error(f"TTS error: {_exc}")
            st.divider()

else:
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

    st.markdown(
        """
        <div class="empty-state-container">
            <div class="empty-state">
                <div class="icon">🎬</div>
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

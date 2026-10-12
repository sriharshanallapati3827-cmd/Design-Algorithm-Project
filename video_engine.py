"""
video_engine.py — Phase 5 Video Assembly & Rendering Engine
=============================================================
Assembles storyboard scene images, TTS audio, and subtitle overlays
into a final rendered MP4 news video using moviepy 2.x.
Includes dynamic subtitle auto-scaling, progress callbacks, and safe
file handling for Windows and cloud container environments.

Requires:
    pip install moviepy imageio-ffmpeg pillow
"""

from __future__ import annotations

import io
import os
import tempfile
import textwrap
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Availability guard — moviepy 2.x
# ---------------------------------------------------------------------------

try:
    from moviepy import (
        ImageClip,
        AudioFileClip,
        CompositeVideoClip,
        concatenate_videoclips,
    )
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from utils import fit_to_broadcast_canvas
    _VIDEO_ENGINE_AVAILABLE = True
except ImportError:
    _VIDEO_ENGINE_AVAILABLE = False


def is_available() -> bool:
    """Return True if moviepy + Pillow are installed and ready."""
    return _VIDEO_ENGINE_AVAILABLE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SIZE: tuple[int, int] = (1024, 576)   # 16:9  width × height
FPS: int = 24
VIDEO_CODEC: str = "libx264"
AUDIO_CODEC: str = "aac"

# Subtitle bar defaults
SUBTITLE_BAR_H_FRAC: float = 0.18
SUBTITLE_FONT_SIZE: int = 22
SUBTITLE_MAX_CHARS_PER_LINE: int = 80
SUBTITLE_BG_ALPHA: int = 185
SUBTITLE_TEXT_COLOR: tuple = (255, 255, 255)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_scene_frame(
    image_bytes: bytes,
    subtitle_text: str,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:
    """Decode image bytes, scale to target_size, and overlay subtitle banner.

    Dynamically adjusts banner height to fit wrapped voiceover text so it
    never spills vertically over the canvas.
    """
    w, h = target_size

    # Decode base scene image
    try:
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception:
        base_img = Image.new("RGBA", (w, h), (20, 30, 50, 255))

    # Scale to fill target canvas cleanly (e.g. 1024x576) preserving faces and titles
    canvas = fit_to_broadcast_canvas(base_img, target_width=w, target_height=h)
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")

    # Overlay subtitle banner if text is provided
    if subtitle_text and subtitle_text.strip():
        # Font resolution
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        font_resolved = False
        for font_name in ("arial.ttf", "segoeui.ttf", "calibri.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                font = ImageFont.truetype(font_name, SUBTITLE_FONT_SIZE)
                font_resolved = True
                break
            except (OSError, IOError):
                continue
        if not font_resolved:
            font = ImageFont.load_default()

        wrapped = textwrap.fill(subtitle_text.strip(), width=SUBTITLE_MAX_CHARS_PER_LINE)
        lines = wrapped.split("\n")

        line_h = SUBTITLE_FONT_SIZE + 6
        total_text_h = line_h * len(lines)

        # Dynamic bar height: ensure banner is at least 18% of frame, but grows
        # to contain all wrapped lines plus padding if necessary
        default_bar_h = int(h * SUBTITLE_BAR_H_FRAC)
        bar_h = max(default_bar_h, total_text_h + 20)
        bar_y = h - bar_h

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Semi-transparent dark banner
        draw.rectangle([(0, bar_y), (w, h)], fill=(0, 0, 0, SUBTITLE_BG_ALPHA))

        text_y = bar_y + max(8, (bar_h - total_text_h) // 2)
        padding_x = int(w * 0.04)

        for line in lines:
            # Shadow for enhanced readability
            draw.text(
                (padding_x + 1, text_y + 1),
                line,
                font=font,
                fill=(0, 0, 0, 240),
            )
            draw.text(
                (padding_x, text_y),
                line,
                font=font,
                fill=(*SUBTITLE_TEXT_COLOR, 255),
            )
            text_y += line_h

        canvas = Image.alpha_composite(canvas, overlay)

    return np.array(canvas.convert("RGB"))


def _make_silent_audio_clip(duration: float) -> tuple["AudioFileClip", str]:
    """Write a short silent WAV and return an AudioFileClip for it."""
    import wave

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00" * (n_samples * 2))
    return AudioFileClip(tmp.name), tmp.name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_scene_clip(
    image_bytes: bytes,
    audio_bytes: Optional[bytes] = None,
    subtitle_text: str = "",
    target_size: tuple[int, int] = TARGET_SIZE,
) -> "ImageClip":
    """Build a single scene VideoClip with subtitles and audio."""
    if not _VIDEO_ENGINE_AVAILABLE:
        raise ImportError("moviepy is not installed. Run: pip install moviepy imageio-ffmpeg")

    _tmp_audio_path: Optional[str] = None

    if audio_bytes:
        tmp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_audio.write(audio_bytes)
        tmp_audio.flush()
        tmp_audio.close()
        _tmp_audio_path = tmp_audio.name
        audio_clip = AudioFileClip(_tmp_audio_path)
        duration = audio_clip.duration
    else:
        audio_clip, _tmp_audio_path = _make_silent_audio_clip(5.0)
        duration = 5.0

    frame_array = _render_scene_frame(
        image_bytes=image_bytes,
        subtitle_text=subtitle_text,
        target_size=target_size,
    )
    scene_clip = ImageClip(frame_array, duration=duration)
    scene_clip = scene_clip.with_audio(audio_clip)

    scene_clip._tmp_audio_path = _tmp_audio_path
    return scene_clip


def assemble_full_video(
    scenes: list[dict],
    scene_audio: dict[int, bytes],
    output_path: str = "output_video.mp4",
    target_size: tuple[int, int] = TARGET_SIZE,
    fps: int = FPS,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """Assemble all storyboard scenes into a final MP4 news video.

    Calls progress_callback regularly to yield heartbeat progress
    back to the parent runtime (e.g. Streamlit WebSocket).
    """
    if not _VIDEO_ENGINE_AVAILABLE:
        raise ImportError("moviepy is not installed. Run: pip install moviepy imageio-ffmpeg")

    if not scenes:
        raise ValueError("Cannot assemble video: no scenes provided.")

    def _progress(frac: float, label: str) -> None:
        if progress_callback:
            try:
                progress_callback(frac, label)
            except Exception:
                pass

    total = len(scenes)
    clips = []
    tmp_paths: list[str] = []

    # ── Build per-scene clips ──────────────────────────────────────────────
    for idx, scene in enumerate(scenes):
        scene_num = scene.get("scene_number", idx + 1)
        image_bytes: Optional[bytes] = scene.get("image_bytes")
        voiceover: str = (
            scene.get("voiceover") or scene.get("narration") or ""
        )
        audio_bytes: Optional[bytes] = scene_audio.get(scene_num)

        _progress((idx + 0.1) / (total + 1), f"Composing scene {scene_num} of {total}...")

        # Fall back to a placeholder if image is missing
        if not image_bytes:
            w, h = target_size
            placeholder = Image.new("RGB", (w, h), color=(20, 30, 50))
            buf = io.BytesIO()
            placeholder.save(buf, format="PNG")
            image_bytes = buf.getvalue()

        clip = create_scene_clip(
            image_bytes=image_bytes,
            audio_bytes=audio_bytes,
            subtitle_text=voiceover,
            target_size=target_size,
        )
        clips.append(clip)

        _tmp = getattr(clip, "_tmp_audio_path", None)
        if _tmp:
            tmp_paths.append(_tmp)

    # ── Concatenate ────────────────────────────────────────────────────────
    _progress(total / (total + 1), "Stitching scene clips...")
    final_clip = concatenate_videoclips(clips, method="compose")

    # ── Write MP4 ─────────────────────────────────────────────────────────
    _progress(total / (total + 1), f"Encoding MP4 -> {os.path.basename(output_path)}...")
    try:
        final_clip.write_videofile(
            output_path,
            fps=fps,
            codec=VIDEO_CODEC,
            audio_codec=AUDIO_CODEC,
            logger=None,
            threads=2,
        )
    except Exception as exc:
        raise RuntimeError(f"moviepy write_videofile failed: {exc}") from exc
    finally:
        # Explicitly close each clip's audio handle before unlinking temp files
        try:
            final_clip.close()
        except Exception:
            pass

        for clip in clips:
            try:
                if hasattr(clip, "audio") and clip.audio is not None:
                    clip.audio.close()
                clip.close()
            except Exception:
                pass

        for path in tmp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    _progress(1.0, "Video rendered successfully!")
    return os.path.abspath(output_path)

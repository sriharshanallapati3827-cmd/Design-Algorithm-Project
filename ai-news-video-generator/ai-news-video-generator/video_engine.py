"""
video_engine.py — Phase 5 Video Assembly & Rendering Engine
=============================================================
Assembles storyboard scene images, TTS audio, and subtitle overlays
into a final rendered MP4 news video using moviepy 2.x.

Usage:
    from video_engine import assemble_full_video, is_available

    def my_progress(frac, label):
        print(f"{frac:.0%} — {label}")

    output_path = assemble_full_video(
        scenes=st.session_state.scenes,
        scene_audio=st.session_state.scene_audio,
        output_path="output_video.mp4",
        progress_callback=my_progress,
    )

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
    from PIL import Image, ImageDraw, ImageFont
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

# Subtitle bar: bottom 18% of frame height
SUBTITLE_BAR_H_FRAC: float = 0.18
SUBTITLE_FONT_SIZE: int = 22
SUBTITLE_MAX_CHARS_PER_LINE: int = 80
SUBTITLE_BG_ALPHA: int = 175          # 0-255 transparency of black band
SUBTITLE_TEXT_COLOR: tuple = (255, 255, 255)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_scene_frame(
    image_bytes: bytes,
    subtitle_text: str,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:  # type: ignore[type-arg]
    """
    Decode image bytes (PNG/JPEG), scale to target_size (1024x576),
    and overlay a semi-transparent lower-third subtitle banner with voiceover text.
    Returns an RGB numpy array (H, W, 3) ready for MoviePy ImageClip.
    """
    import numpy as np
    from PIL import ImageOps

    w, h = target_size

    # Decode base scene image
    try:
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception:
        # Fallback if image bytes are corrupted or invalid
        base_img = Image.new("RGBA", (w, h), (20, 30, 50, 255))

    # Scale to fill 1024x576 target canvas cleanly
    canvas = ImageOps.fit(base_img, (w, h), method=Image.LANCZOS)
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")

    # Overlay subtitle banner if text is provided
    if subtitle_text and subtitle_text.strip():
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        bar_h = int(h * SUBTITLE_BAR_H_FRAC)
        bar_y = h - bar_h

        # Semi-transparent dark banner (alpha=185 for contrast)
        draw.rectangle([(0, bar_y), (w, h)], fill=(0, 0, 0, SUBTITLE_BG_ALPHA))

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

        line_h = SUBTITLE_FONT_SIZE + 4
        total_text_h = line_h * len(lines)
        text_y = bar_y + max(4, (bar_h - total_text_h) // 2)

        padding_x = int(w * 0.04)
        for line in lines:
            # Shadow for readability
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
    """
    Build a single scene VideoClip by:
      • Rendering the scene visual with lower-third subtitle overlay.
      • Attaching the audio voiceover (which sets the clip duration).

    If ``audio_bytes`` is None / empty, the clip uses a 5-second silent
    placeholder so the video still renders cleanly.

    Parameters
    ----------
    image_bytes   : Raw PNG/JPEG bytes of the scene visual.
    audio_bytes   : Raw MP3 bytes from edge-tts, or None.
    subtitle_text : Voiceover text to display in the subtitle bar.
    target_size   : (width, height) in pixels — default 1024 × 576 (16:9).

    Returns
    -------
    ImageClip
        A moviepy clip ready to be concatenated.
    """
    if not _VIDEO_ENGINE_AVAILABLE:
        raise ImportError("moviepy is not installed. Run: pip install moviepy imageio-ffmpeg")

    _tmp_audio_path: Optional[str] = None

    # ── Audio ──────────────────────────────────────────────────────────────
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

    # ── Render Frame with Subtitles ─────────────────────────────────────────
    frame_array = _render_scene_frame(
        image_bytes=image_bytes,
        subtitle_text=subtitle_text,
        target_size=target_size,
    )
    scene_clip = ImageClip(frame_array, duration=duration)
    scene_clip = scene_clip.with_audio(audio_clip)

    # Store temp path for cleanup after rendering
    scene_clip._tmp_audio_path = _tmp_audio_path  # type: ignore[attr-defined]

    return scene_clip


def assemble_full_video(
    scenes: list[dict],
    scene_audio: dict[int, bytes],
    output_path: str = "output_video.mp4",
    target_size: tuple[int, int] = TARGET_SIZE,
    fps: int = FPS,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Assemble all storyboard scenes into a final MP4 news video.

    Parameters
    ----------
    scenes          : List of scene dicts from ``st.session_state.scenes``.
                      Each dict must have keys: ``scene_number``, ``image_bytes``,
                      ``voiceover`` (or ``narration``).
    scene_audio     : Dict mapping ``scene_number`` → MP3 bytes
                      (from ``st.session_state.scene_audio``).
    output_path     : File path for the rendered MP4.
    target_size     : (width, height) — default 1024 × 576.
    fps             : Frames per second — default 24.
    progress_callback : Optional callable ``(fraction: float, label: str) -> None``
                        called as each scene is assembled (0 → 1) and again
                        when writing is complete.

    Returns
    -------
    str
        Absolute path to the rendered MP4 file.

    Raises
    ------
    ImportError  : If moviepy / Pillow are not installed.
    ValueError   : If ``scenes`` is empty.
    RuntimeError : If moviepy fails to write the video.
    """
    if not _VIDEO_ENGINE_AVAILABLE:
        raise ImportError("moviepy is not installed. Run: pip install moviepy imageio-ffmpeg")

    if not scenes:
        raise ValueError("Cannot assemble video: no scenes provided.")

    def _progress(frac: float, label: str) -> None:
        if progress_callback:
            progress_callback(frac, label)

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

        _progress(idx / total, f"Building scene {scene_num} of {total}...")

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

        # Track temp audio files for cleanup
        _tmp = getattr(clip, "_tmp_audio_path", None)
        if _tmp:
            tmp_paths.append(_tmp)

    # ── Concatenate ────────────────────────────────────────────────────────
    _progress(total / (total + 1), "Concatenating scenes...")
    final_clip = concatenate_videoclips(clips, method="compose")

    # ── Write MP4 ─────────────────────────────────────────────────────────
    _progress(total / (total + 1), f"Rendering MP4 -> {os.path.basename(output_path)}...")
    try:
        final_clip.write_videofile(
            output_path,
            fps=fps,
            codec=VIDEO_CODEC,
            audio_codec=AUDIO_CODEC,
            logger=None,          # suppress moviepy console output
            threads=2,
        )
    except Exception as exc:
        raise RuntimeError(f"moviepy write_videofile failed: {exc}") from exc
    finally:
        # Close all clips to release file handles, then delete temp audio files
        final_clip.close()
        for clip in clips:
            try:
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


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Video engine available:", is_available())

    if not is_available():
        print("Install moviepy: pip install moviepy imageio-ffmpeg pillow")
        sys.exit(1)

    # Build a tiny 2-scene test video with placeholder images
    import numpy as np

    def _solid_png(color: tuple, size=(1024, 576)) -> bytes:
        img = Image.new("RGB", size, color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    test_scenes = [
        {
            "scene_number": 1,
            "image_bytes": _solid_png((20, 40, 80)),
            "voiceover": "Breaking news: The video engine has been successfully integrated.",
            "timestamp": "0:00",
        },
        {
            "scene_number": 2,
            "image_bytes": _solid_png((40, 20, 60)),
            "voiceover": "Reporters on the ground confirm the rendering pipeline is fully operational.",
            "timestamp": "0:05",
        },
    ]

    out = "test_output.mp4"

    def prog(frac, label):
        print(f"  [{frac:5.1%}] {label}")

    print("\nAssembling test video (no audio — silent clips)…")
    try:
        path = assemble_full_video(
            scenes=test_scenes,
            scene_audio={},        # no audio → silent placeholders
            output_path=out,
            progress_callback=prog,
        )
        size = os.path.getsize(path)
        print(f"\nOK  Written {size:,} bytes -> {path}")
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)

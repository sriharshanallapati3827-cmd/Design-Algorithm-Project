"""
audio.py — Phase 4 Neural TTS Engine
======================================
Generates MP3 voiceover audio for storyboard scenes using Microsoft's
``edge-tts`` neural speech synthesis engine.

Usage (sync, from Streamlit):
    from audio import generate_scene_audio, get_voice_options, VOICES

    audio_bytes = generate_scene_audio(
        text="Breaking news from the capital...",
        voice="en-US-AvaNeural",
        rate="+0%",
    )
    # audio_bytes is raw MP3 data ready for st.audio()

Requires:
    pip install edge-tts
"""

from __future__ import annotations

import asyncio
import io
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# Voice catalogue
# ---------------------------------------------------------------------------

VOICES: dict[str, str] = {
    "Female - Professional News Anchor": "en-US-AvaNeural",
    "Male - Authoritative News Anchor": "en-US-AndrewNeural",
    "Male - Standard Broadcast": "en-US-GuyNeural",
    "Female - British News": "en-GB-SoniaNeural",
}

# Default voice
DEFAULT_VOICE = "en-US-AvaNeural"

# ---------------------------------------------------------------------------
# edge-tts availability guard
# ---------------------------------------------------------------------------

try:
    import edge_tts  # noqa: F401 — presence check only
    _EDGE_TTS_AVAILABLE = True
except ImportError:
    _EDGE_TTS_AVAILABLE = False


def is_available() -> bool:
    """Return True if edge-tts is installed and ready to use."""
    return _EDGE_TTS_AVAILABLE


# ---------------------------------------------------------------------------
# Core async generator
# ---------------------------------------------------------------------------


async def generate_scene_audio_async(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
) -> bytes:
    """
    Stream TTS audio from edge-tts and return raw MP3 bytes.

    Parameters
    ----------
    text  : The narration text to synthesise.
    voice : An edge-tts voice ID (e.g. "en-US-AvaNeural").
    rate  : Speaking rate adjustment, e.g. "+10%" or "-5%".

    Returns
    -------
    bytes
        Raw MP3 audio data.

    Raises
    ------
    ImportError
        If edge-tts is not installed.
    ValueError
        If text is empty.
    RuntimeError
        If edge-tts synthesis fails.
    """
    if not _EDGE_TTS_AVAILABLE:
        raise ImportError(
            "edge-tts is not installed. Run: pip install edge-tts"
        )

    text = (text or "").strip()
    if not text:
        raise ValueError("Cannot synthesise audio for empty text.")

    import edge_tts  # local import — only reached when available

    buffer = io.BytesIO()
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])
    except Exception as exc:
        raise RuntimeError(f"edge-tts synthesis failed: {exc}") from exc

    audio_bytes = buffer.getvalue()
    if not audio_bytes:
        raise RuntimeError(
            "edge-tts returned no audio data. "
            "Check that the voice ID is valid and the text is non-empty."
        )

    return audio_bytes


# ---------------------------------------------------------------------------
# Sync wrapper (safe for Streamlit''s event-loop environment)
# ---------------------------------------------------------------------------


def _run_in_thread(text: str, voice: str, rate: str) -> bytes:
    """Run the async coroutine in a dedicated thread with its own event loop."""
    result: list = []
    error: list = []

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            audio = loop.run_until_complete(
                generate_scene_audio_async(text, voice, rate)
            )
            result.append(audio)
        except Exception as exc:
            error.append(exc)
        finally:
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()

    if error:
        raise error[0]

    return result[0]


def generate_scene_audio(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
) -> bytes:
    """
    Synchronous TTS synthesis — safe to call from Streamlit.

    Internally spawns a worker thread with its own event loop so that it
    works correctly regardless of whether an event loop is already running
    in the calling thread (e.g. inside Streamlit or Jupyter).

    Parameters
    ----------
    text  : The narration text to synthesise.
    voice : An edge-tts voice ID (e.g. "en-US-AvaNeural").
    rate  : Speaking rate adjustment, e.g. "+10%" or "-5%".

    Returns
    -------
    bytes
        Raw MP3 audio data, suitable for st.audio(data, format="audio/mp3").
    """
    return _run_in_thread(text, voice, rate)


# ---------------------------------------------------------------------------
# UI helper
# ---------------------------------------------------------------------------


def get_voice_options() -> dict[str, str]:
    """
    Return the voice catalogue as {display_name: voice_id}.

    Suitable for use as the options argument of st.selectbox.
    """
    return dict(VOICES)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("edge-tts available:", is_available())
    print("Voice options:")
    for name, vid in get_voice_options().items():
        print(f"  {name!r:50s} -> {vid}")

    # Quick synthesis test
    sample = "Breaking news: AI systems now generate voiceovers autonomously."
    print(f"\nSynthesising sample text with default voice ({DEFAULT_VOICE})...")
    try:
        data = generate_scene_audio(sample)
        out_path = "sample_audio.mp3"
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"OK  Written {len(data):,} bytes -> {out_path}")
    except Exception as e:
        print(f"FAIL  Synthesis failed: {e}", file=sys.stderr)
        sys.exit(1)

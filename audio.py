"""
audio.py — Phase 4 Neural TTS Engine with Resilient gTTS Fallback
==================================================================
Generates MP3 voiceover audio for storyboard scenes using Microsoft's
``edge-tts`` neural speech synthesis engine, with automatic fallback to
``gTTS`` (Google Translate TTS) on 403 Forbidden / WebSocket drops, and
a silent MP3 track fallback if all network TTS engines are unreachable.

Usage (sync, from Streamlit):
    from audio import generate_scene_audio, get_voice_options, VOICES

    audio_bytes = generate_scene_audio(
        text="Breaking news from the capital...",
        voice="en-US-AvaNeural",
        rate="+0%",
    )
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice catalogue
# ---------------------------------------------------------------------------

VOICES: dict[str, str] = {
    # Standard tested US broadcast voices (Primary)
    "Female – US News Anchor (Ava)": "en-US-AvaNeural",
    "Male – US Standard Broadcast (Guy)": "en-US-GuyNeural",
    "Female – US General Broadcast (Aria)": "en-US-AriaNeural",
    # Regional & International Anchors
    "Female – Indian News Anchor (Neerja)": "en-IN-NeerjaNeural",
    "Female – Expressive Indian Anchor (Neerja Expressive)": "en-IN-NeerjaExpressiveNeural",
    "Male – Indian News Anchor (Prabhat)": "en-IN-PrabhatNeural",
    "Female – UK News Anchor (Sonia)": "en-GB-SoniaNeural",
}

DEFAULT_VOICE = "en-US-AvaNeural"
FALLBACK_EDGE_VOICE = "en-US-AriaNeural"

# ---------------------------------------------------------------------------
# Engine availability guards
# ---------------------------------------------------------------------------

try:
    import edge_tts
    _EDGE_TTS_AVAILABLE = True
except ImportError:
    _EDGE_TTS_AVAILABLE = False

try:
    from gtts import gTTS
    _GTTS_AVAILABLE = True
except ImportError:
    _GTTS_AVAILABLE = False


def is_available() -> bool:
    """Return True if at least one TTS engine (edge-tts or gTTS) is available."""
    return _EDGE_TTS_AVAILABLE or _GTTS_AVAILABLE


# ---------------------------------------------------------------------------
# Text sanitization & silent audio utilities
# ---------------------------------------------------------------------------

def sanitize_tts_text(text: str) -> str:
    """Sanitize input text for edge-tts and gTTS to prevent 403 and SSML errors.

    Removes XML/HTML tags (<...>), converts raw & into 'and', strips quotes
    that disrupt XML attributes, removes unprintable characters, and collapses
    whitespace.
    """
    if not text:
        return ""
    # Strip HTML/XML tags
    clean = re.sub(r"<[^>]+>", " ", text)
    # Replace ampersands (breaking SSML)
    clean = clean.replace("&", " and ")
    # Replace double quotes that disrupt XML attributes
    clean = clean.replace('"', "'")
    # Keep only printable characters + standard newlines/tabs
    clean = "".join(ch for ch in clean if ch.isprintable() or ch in "\n\r\t")
    # Normalize duplicate whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def generate_silent_mp3(duration_sec: float) -> bytes:
    """Generate valid silent MP3 audio bytes matching the estimated duration.

    Uses standard MPEG-1 Layer 3 128kbps/44.1kHz silent frames (417 bytes each),
    where each frame equals ~0.026122 seconds (1152 samples / 44100).
    """
    frame = b"\xff\xfb\x90\x00" + b"\x00" * 413
    frame_count = max(1, int(round(duration_sec * 38.28125)))
    return frame * frame_count


def generate_gtts_audio(text: str, lang: str = "en") -> bytes:
    """Generate MP3 audio bytes using gTTS (Google Translate TTS)."""
    if not _GTTS_AVAILABLE:
        raise ImportError("gtts is not installed. Run: pip install gtts")
    tts = gTTS(text=text, lang=lang)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Core async edge-tts synthesizer
# ---------------------------------------------------------------------------

async def _synthesize_edge_tts(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    max_retries: int = 2,
) -> bytes:
    """Call edge-tts with automatic fallback to en-US-AriaNeural on 403 or failure."""
    if not _EDGE_TTS_AVAILABLE:
        raise ImportError("edge-tts is not installed. Run: pip install edge-tts")

    import edge_tts

    voices_to_try = [voice]
    if voice != FALLBACK_EDGE_VOICE:
        voices_to_try.append(FALLBACK_EDGE_VOICE)

    last_exc: Optional[Exception] = None

    for v in voices_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                buffer = io.BytesIO()
                communicate = edge_tts.Communicate(text=text, voice=v, rate=rate)
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        buffer.write(chunk["data"])

                audio_bytes = buffer.getvalue()
                if audio_bytes:
                    return audio_bytes
                raise RuntimeError(f"edge-tts returned 0 bytes for voice {v}")

            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                logger.warning(
                    "edge-tts attempt %d with voice %s failed: %s",
                    attempt, v, exc
                )
                # If 403 Forbidden or invalid response, break retry on this voice
                if "403" in err_str or "forbidden" in err_str or "invalid response" in err_str:
                    break
                if attempt < max_retries:
                    await asyncio.sleep(1.0)

    raise last_exc or RuntimeError("edge-tts failed on all voices")


# ---------------------------------------------------------------------------
# Multi-tier resilient TTS generation (edge-tts -> gTTS -> Silent MP3)
# ---------------------------------------------------------------------------

async def generate_scene_audio_async(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
) -> bytes:
    """Synthesise narration text into MP3 audio with multi-tier fallback:

      1. Microsoft edge-tts (primary voice)
      2. Microsoft edge-tts (en-US-AriaNeural fallback voice)
      3. Google TTS (gTTS fallback on 403 / network / WebSocket errors)
      4. Silent MP3 track matching estimated duration (safe MoviePy fallback)
    """
    clean_text = sanitize_tts_text(text)
    if not clean_text:
        return generate_silent_mp3(2.0)

    # Estimate duration: ~140 WPM -> ~2.33 words/sec
    word_count = len(clean_text.split())
    estimated_duration = max(2.5, word_count / 2.33)

    # ── Tier 1 & 2: edge-tts ──────────────────────────────────────────────
    if _EDGE_TTS_AVAILABLE:
        try:
            return await _synthesize_edge_tts(clean_text, voice=voice, rate=rate)
        except Exception as edge_exc:
            logger.warning(
                "edge-tts failed (%s). Triggering gTTS fallback...", edge_exc
            )

    # ── Tier 3: gTTS fallback ────────────────────────────────────────────
    if _GTTS_AVAILABLE:
        try:
            logger.info("Synthesizing audio via gTTS fallback...")
            return generate_gtts_audio(clean_text, lang="en")
        except Exception as gtts_exc:
            logger.warning(
                "gTTS fallback failed (%s). Falling back to silent MP3...", gtts_exc
            )

    # ── Tier 4: Silent MP3 fallback ──────────────────────────────────────
    logger.error(
        "All TTS engines failed. Generating %.1fs silent MP3 fallback.",
        estimated_duration,
    )
    return generate_silent_mp3(estimated_duration)


# ---------------------------------------------------------------------------
# Sync wrapper (isolated event loop thread for Streamlit safety)
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
    """Synchronous TTS synthesis — safe to call from Streamlit."""
    return _run_in_thread(text, voice, rate)


def get_voice_options() -> dict[str, str]:
    """Return the voice catalogue as {display_name: voice_id}."""
    return dict(VOICES)

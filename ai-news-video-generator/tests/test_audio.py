"""
tests/test_audio.py
===================
Unit test suite for Phase 4 audio TTS generation and resilient fallback:
  1. test_edge_tts_primary_generation
  2. test_gtts_fallback_on_403_failure
  3. test_audio_moviepy_compatibility
"""

import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest

from audio import (
    generate_scene_audio,
    generate_gtts_audio,
    sanitize_tts_text,
    generate_silent_mp3,
)

try:
    from moviepy import AudioFileClip
    _MOVIEPY_AVAILABLE = True
except ImportError:
    _MOVIEPY_AVAILABLE = False


def test_edge_tts_primary_generation():
    """Test Case 1: Primary edge-tts generation returns valid MP3 bytes."""
    sample_text = "Breaking news: Scientists confirm Artemis IV has entered lunar orbit."
    audio_bytes = generate_scene_audio(sample_text, voice="en-US-AvaNeural")

    assert isinstance(audio_bytes, bytes), "Expected raw audio bytes"
    assert len(audio_bytes) > 500, f"Expected non-trivial audio payload, got {len(audio_bytes)} bytes"


def test_gtts_fallback_on_403_failure():
    """Test Case 2: 403 Forbidden on edge-tts gracefully falls back to gTTS without error."""
    sample_text = "Headline: Global markets react with historic surge as new tech index opens."

    with patch("audio._synthesize_edge_tts", side_effect=RuntimeError("403 Forbidden: Invalid response")):
        # Should not raise; must seamlessly fall back to gTTS
        audio_bytes = generate_scene_audio(sample_text, voice="en-US-AvaNeural")

        assert isinstance(audio_bytes, bytes), "Expected raw audio bytes from fallback"
        assert len(audio_bytes) > 100, f"Expected non-empty gTTS audio payload, got {len(audio_bytes)} bytes"


@pytest.mark.skipif(not _MOVIEPY_AVAILABLE, reason="MoviePy required for audio compatibility test")
def test_audio_moviepy_compatibility():
    """Test Case 3: Generated audio bytes can be loaded into MoviePy AudioFileClip with duration > 0s."""
    sample_text = "Weather update: Clear skies forecasted across all central regions."
    audio_bytes = generate_scene_audio(sample_text)

    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        clip = AudioFileClip(tmp_path)
        assert clip.duration is not None, "Clip duration should not be None"
        assert clip.duration > 0.0, f"Expected duration > 0s, got {clip.duration}"
        clip.close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

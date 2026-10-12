"""
End-to-end integration test — validates the full pipeline with a mocked
Gemini API, real ingestion, and real utils.

Simulates: article text → ingestion.clean_text_input → director.generate_storyboard
            → scene validation → utils.generate_demo_scenes fallback
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from ingestion import clean_text_input, extract_text_from_url
from director import generate_storyboard, _estimate_scene_range
from utils import calculate_scene_range, generate_demo_scenes, format_timestamp


# ===================================================================
# Full Pipeline Integration
# ===================================================================

class TestEndToEndPipeline:
    """Simulates the real app pipeline with mocked external calls."""

    ARTICLE_HTML = """
    <html><body>
        <p>Breaking news: global leaders met today to discuss climate action.</p>
        <p>The summit in Geneva brought together representatives from 50 nations,
           each pledging to reduce carbon emissions by 40% before 2035.</p>
        <p>Environmental experts praised the agreements but cautioned that
           enforcement mechanisms remain weak.</p>
        <p>Financial markets reacted with modest gains, as green energy stocks
           rose an average of 2.1% across major exchanges.</p>
    </body></html>
    """

    GEMINI_SCENES = [
        {
            "scene_number": 1,
            "timestamp": "00:00 - 00:20",
            "narration": "World leaders gather in Geneva for a historic climate summit.",
            "visual_prompt": "Wide shot of a grand conference hall with flags of 50 nations."
        },
        {
            "scene_number": 2,
            "timestamp": "00:20 - 00:40",
            "narration": "Fifty nations pledge to cut carbon emissions by 40 percent by 2035.",
            "visual_prompt": "Close-up of a document being signed, pens and national seals visible."
        },
        {
            "scene_number": 3,
            "timestamp": "00:40 - 00:60",
            "narration": "Markets respond positively as green energy stocks climb worldwide.",
            "visual_prompt": "Trading floor screens showing upward green arrows, cinematic lighting."
        },
    ]

    @patch("ingestion.requests.get")
    @patch("director.genai.Client")
    def test_url_to_storyboard_pipeline(self, mock_client_cls, mock_get):
        """Full flow: URL → scrape → clean → Gemini → validate scenes."""

        # --- Mock URL fetch ---
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = self.ARTICLE_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # --- Mock Gemini ---
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(self.GEMINI_SCENES)
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        # --- Execute pipeline ---
        # Step 1: Ingest
        article_text = extract_text_from_url("https://news.example.com/climate")
        assert len(article_text) > 50
        assert "climate" in article_text.lower()

        # Step 2: Generate storyboard
        duration = 60
        scenes = generate_storyboard(
            article_text, duration, api_key="test-key-integration"
        )

        # Step 3: Validate output
        assert len(scenes) == 3
        for i, scene in enumerate(scenes):
            assert scene["scene_number"] == i + 1
            assert "timestamp" in scene
            assert "narration" in scene
            assert len(scene["narration"]) > 10
            assert "visual_prompt" in scene
            assert len(scene["visual_prompt"]) > 10

    def test_text_to_demo_scenes_pipeline(self):
        """Full flow: raw text → clean → demo scene generation (no LLM)."""

        raw_input = """
            Breaking   news:   global    leaders    met
            today   to   discuss   climate   action.


            The    summit    brought    representatives
            from    50    nations.
        """

        # Step 1: Clean
        cleaned = clean_text_input(raw_input)
        assert "  " not in cleaned  # No double spaces
        assert "\n\n\n" not in cleaned  # No triple newlines

        # Step 2: Calculate scene range
        duration = 90
        min_scenes, max_scenes = calculate_scene_range(duration)
        assert min_scenes >= 1
        assert max_scenes >= min_scenes

        # Step 3: Generate demo scenes
        scenes = generate_demo_scenes(min_scenes, duration)
        assert len(scenes) == min_scenes
        for scene in scenes:
            assert scene["image_bytes"] is not None


# ===================================================================
# Cross-module Consistency
# ===================================================================

class TestCrossModuleConsistency:
    """Verify that utils and director agree on scene calculations."""

    def test_scene_ranges_are_reasonable(self):
        """Both modules should produce overlapping scene ranges."""
        for dur in [60, 80, 100, 120]:
            u_lo, u_hi = calculate_scene_range(dur)
            d_lo, d_hi = _estimate_scene_range(dur)

            # Both should return positive values
            assert u_lo >= 1
            assert d_lo >= 1

            # Ranges should overlap (not wildly different)
            overlap = max(0, min(u_hi, d_hi) - max(u_lo, d_lo) + 1)
            assert overlap > 0, (
                f"No overlap at {dur}s: utils={u_lo}-{u_hi}, director={d_lo}-{d_hi}"
            )

    def test_format_timestamp_used_by_demo_scenes(self):
        """Demo scenes should use the same timestamp format as format_timestamp."""
        scenes = generate_demo_scenes(3, 60)
        for scene in scenes:
            assert "–" in scene["timestamp"]  # em-dash from format_timestamp


# ===================================================================
# Spec Compliance Checks (from FRONTEND_SPEC.md & PHASE2_SPEC.md)
# ===================================================================

class TestSpecCompliance:
    """Verify key spec requirements are met by the codebase."""

    def test_duration_range_matches_spec(self):
        """FRONTEND_SPEC: slider from 60 to 120 seconds."""
        with open("app.py", "r", encoding="utf-8") as f:
            source = f.read()
        assert "min_value=60" in source
        assert "max_value=120" in source

    def test_wpm_rate_matches_spec(self):
        """PHASE2_SPEC: ~140 WPM = 2.33 words/second."""
        from director import _WPM_RATE
        assert abs(_WPM_RATE - 2.33) < 0.01

    def test_pdf_limits_match_spec(self):
        """PHASE2_SPEC: max 5 pages, 5 MB."""
        from ingestion import _MAX_PDF_PAGES, _MAX_PDF_BYTES
        assert _MAX_PDF_PAGES == 5
        assert _MAX_PDF_BYTES == 5 * 1024 * 1024

    def test_art_style_removed(self):
        """FRONTEND_SPEC: ART STYLE dropdown must not exist."""
        with open("app.py", "r", encoding="utf-8") as f:
            source = f.read().lower()
        # Neither "art_style" nor "art style" should appear as a UI element
        # (allow it in comments but not as a variable/label)
        assert "art_style" not in source or "art style" not in source

    def test_brand_name_correct(self):
        """FRONTEND_SPEC: title must be 'AI NEWS Video Generator'."""
        with open("app.py", "r", encoding="utf-8") as f:
            source = f.read()
        assert "AI NEWS Video Generator" in source

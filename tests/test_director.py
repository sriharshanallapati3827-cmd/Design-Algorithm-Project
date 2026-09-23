"""
Tests for director.py — scene estimation, JSON parsing, and Gemini integration.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from director import (
    _estimate_scene_range,
    _parse_scenes_json,
    generate_storyboard,
    MODEL_MAP,
)


# ===================================================================
# _estimate_scene_range
# ===================================================================

class TestEstimateSceneRange:
    """Verify the linear interpolation of scene counts from the spec table."""

    def test_60s_boundary(self):
        lo, hi = _estimate_scene_range(60)
        assert lo == 3
        assert hi == 4

    def test_120s_boundary(self):
        lo, hi = _estimate_scene_range(120)
        assert lo == 7
        assert hi == 9

    def test_100s_boundary(self):
        lo, hi = _estimate_scene_range(100)
        assert lo == 5
        assert hi == 7

    def test_below_minimum_clamps(self):
        """Durations below 60s should clamp to the 60s row."""
        lo, hi = _estimate_scene_range(30)
        assert lo == 3
        assert hi == 4

    def test_above_maximum_clamps(self):
        """Durations above 120s should clamp to the 120s row."""
        lo, hi = _estimate_scene_range(180)
        assert lo == 7
        assert hi == 9

    def test_midpoint_interpolation(self):
        """80s is midway between 60 and 100; should interpolate."""
        lo, hi = _estimate_scene_range(80)
        assert 3 <= lo <= 5
        assert 4 <= hi <= 7

    def test_returns_ints(self):
        lo, hi = _estimate_scene_range(90)
        assert isinstance(lo, int)
        assert isinstance(hi, int)

    def test_min_never_exceeds_max(self):
        """For any duration, min_scenes <= max_scenes."""
        for dur in range(30, 200, 5):
            lo, hi = _estimate_scene_range(dur)
            assert lo <= hi, f"Failed at {dur}s: {lo} > {hi}"


# ===================================================================
# _parse_scenes_json
# ===================================================================

class TestParseScenesJson:
    """Verify JSON extraction from various LLM response formats."""

    def test_clean_json_array(self, sample_gemini_response):
        scenes = _parse_scenes_json(sample_gemini_response)
        assert len(scenes) == 3
        assert scenes[0]["scene_number"] == 1
        assert "Artemis" in scenes[0]["narration"]

    def test_fenced_json(self, sample_gemini_response_fenced):
        scenes = _parse_scenes_json(sample_gemini_response_fenced)
        assert len(scenes) == 1
        assert scenes[0]["scene_number"] == 1

    def test_json_with_trailing_prose(self):
        raw = """[{"scene_number": 1, "timestamp": "00:00 - 00:30",
                   "narration": "Hello", "visual_prompt": "World"}]
        I hope this helps!"""
        scenes = _parse_scenes_json(raw)
        assert len(scenes) == 1

    def test_raises_on_non_json(self):
        with pytest.raises(RuntimeError, match="Failed to parse"):
            _parse_scenes_json("This is not JSON at all.")

    def test_raises_on_json_object_not_array(self):
        with pytest.raises(RuntimeError, match="Failed to parse"):
            _parse_scenes_json('{"key": "value"}')

    def test_raises_on_empty_string(self):
        with pytest.raises(RuntimeError, match="Failed to parse"):
            _parse_scenes_json("")


# ===================================================================
# generate_storyboard (mocked Gemini API)
# ===================================================================

class TestGenerateStoryboard:
    """Integration-style tests with a mocked Gemini client."""

    def _make_mock_response(self, scenes_json: str):
        mock_response = MagicMock()
        mock_response.text = scenes_json
        return mock_response

    @patch("director.genai.Client")
    def test_successful_generation(self, mock_client_cls, sample_article_text,
                                   sample_gemini_response):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            self._make_mock_response(sample_gemini_response)
        )
        mock_client_cls.return_value = mock_client

        scenes = generate_storyboard(
            sample_article_text, 60, api_key="test-key-123"
        )

        assert len(scenes) == 3
        assert scenes[0]["scene_number"] == 1
        assert "narration" in scenes[0]
        assert "visual_prompt" in scenes[0]
        assert "timestamp" in scenes[0]

        # Verify Gemini was called with the correct model
        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-3.6-flash"

    @patch("director.genai.Client")
    def test_fenced_response_handling(self, mock_client_cls,
                                      sample_article_text,
                                      sample_gemini_response_fenced):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            self._make_mock_response(sample_gemini_response_fenced)
        )
        mock_client_cls.return_value = mock_client

        scenes = generate_storyboard(
            sample_article_text, 60, api_key="test-key"
        )
        assert len(scenes) == 1

    def test_raises_without_api_key(self, sample_article_text):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No Gemini API key"):
                generate_storyboard(sample_article_text, 60, api_key="")

    def test_raises_on_unsupported_model(self, sample_article_text):
        with pytest.raises(ValueError, match="Unsupported model"):
            generate_storyboard(
                sample_article_text, 60,
                api_key="test-key",
                model_name="GPT-4o"
            )

    @patch("director.genai.Client")
    def test_raises_on_empty_response(self, mock_client_cls,
                                      sample_article_text):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            self._make_mock_response("[]")
        )
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="empty scene list"):
            generate_storyboard(
                sample_article_text, 60, api_key="test-key"
            )

    @patch("director.genai.Client")
    def test_raises_on_missing_keys(self, mock_client_cls,
                                    sample_article_text):
        # Scene missing "visual_prompt"
        bad_scene = json.dumps([{
            "scene_number": 1,
            "timestamp": "00:00 - 00:30",
            "narration": "Hello world."
        }])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            self._make_mock_response(bad_scene)
        )
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="missing required key"):
            generate_storyboard(
                sample_article_text, 60, api_key="test-key"
            )

    @patch("director.genai.Client")
    def test_uses_correct_model_id(self, mock_client_cls,
                                   sample_article_text,
                                   sample_gemini_response):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = (
            self._make_mock_response(sample_gemini_response)
        )
        mock_client_cls.return_value = mock_client

        generate_storyboard(
            sample_article_text, 60,
            api_key="test-key",
            model_name="Gemini 1.5 Flash"
        )

        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-3.5-flash-lite"

    def test_model_map_completeness(self):
        """All models in MODEL_MAP should have valid Gemini IDs."""
        for name, model_id in MODEL_MAP.items():
            assert model_id.startswith("gemini-"), f"{name} -> {model_id}"

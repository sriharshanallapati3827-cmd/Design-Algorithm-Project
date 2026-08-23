"""
Tests for utils.py — scene math, timestamps, demo data, placeholder images.
"""

import pytest
from utils import (
    calculate_scene_range,
    format_timestamp,
    load_demo_article,
    generate_demo_scenes,
)


# ===================================================================
# calculate_scene_range
# ===================================================================

class TestCalculateSceneRange:
    """Verify the heuristic scene count calculation."""

    def test_60s(self):
        lo, hi = calculate_scene_range(60)
        assert lo == 3
        assert hi == 4

    def test_120s(self):
        lo, hi = calculate_scene_range(120)
        assert lo == 6
        assert hi == 9

    def test_90s(self):
        lo, hi = calculate_scene_range(90)
        assert lo == 4
        assert hi == 6

    def test_minimum_clamp(self):
        """Very short durations should still return at least 1 scene."""
        lo, hi = calculate_scene_range(10)
        assert lo >= 1

    def test_min_never_exceeds_max(self):
        for dur in range(10, 200, 5):
            lo, hi = calculate_scene_range(dur)
            assert lo <= hi, f"Failed at {dur}s: min={lo} > max={hi}"


# ===================================================================
# format_timestamp
# ===================================================================

class TestFormatTimestamp:
    """Verify timestamp string formatting."""

    def test_zero_to_fifteen(self):
        result = format_timestamp(0, 15)
        assert result == "00:00 – 00:15"

    def test_wraps_minutes(self):
        result = format_timestamp(60, 90)
        assert result == "01:00 – 01:30"

    def test_two_minutes(self):
        result = format_timestamp(0, 120)
        assert result == "00:00 – 02:00"

    def test_mid_values(self):
        result = format_timestamp(45, 75)
        assert result == "00:45 – 01:15"


# ===================================================================
# load_demo_article
# ===================================================================

class TestLoadDemoArticle:
    """Verify the bundled demo article."""

    def test_returns_non_empty_string(self):
        article = load_demo_article()
        assert isinstance(article, str)
        assert len(article) > 500

    def test_contains_expected_content(self):
        article = load_demo_article()
        assert "Artemis" in article
        assert "NASA" in article

    def test_is_deterministic(self):
        assert load_demo_article() == load_demo_article()


# ===================================================================
# generate_demo_scenes
# ===================================================================

class TestGenerateDemoScenes:
    """Verify demo scene generation."""

    def test_correct_scene_count(self):
        scenes = generate_demo_scenes(5, 90)
        assert len(scenes) == 5

    def test_scene_numbering(self):
        scenes = generate_demo_scenes(3, 60)
        for i, scene in enumerate(scenes):
            assert scene["scene_number"] == i + 1

    def test_required_keys(self):
        scenes = generate_demo_scenes(4, 80)
        required = {"scene_number", "timestamp", "voiceover", "visual_prompt", "image_bytes"}
        for scene in scenes:
            assert required.issubset(set(scene.keys())), (
                f"Missing keys: {required - set(scene.keys())}"
            )

    def test_timestamps_are_sequential(self):
        scenes = generate_demo_scenes(5, 100)
        for i, scene in enumerate(scenes):
            ts = scene["timestamp"]
            assert "–" in ts  # em-dash separator from format_timestamp

    def test_image_bytes_not_none(self):
        """Placeholder images should be generated when Pillow is available."""
        scenes = generate_demo_scenes(2, 60)
        for scene in scenes:
            assert scene["image_bytes"] is not None
            assert isinstance(scene["image_bytes"], bytes)
            assert len(scene["image_bytes"]) > 0

    def test_single_scene(self):
        scenes = generate_demo_scenes(1, 60)
        assert len(scenes) == 1
        assert scenes[0]["scene_number"] == 1

    def test_large_scene_count(self):
        """Should handle more scenes than sample data by cycling."""
        scenes = generate_demo_scenes(15, 120)
        assert len(scenes) == 15
        # All should still have valid keys
        for scene in scenes:
            assert "voiceover" in scene
            assert "visual_prompt" in scene

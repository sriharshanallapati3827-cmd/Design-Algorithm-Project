"""
tests/test_image_generation.py
===============================
Tests for generator.py SerpApi Google Images engine, key pooling,
query cleaning, local GPU, and disk caching.
"""

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from generator import (
    clean_search_query,
    SerpApiKeyPool,
    generate_scene_image,
    save_scene_image,
)


def _tiny_png_bytes(width=100, height=100) -> bytes:
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestSearchQueryCleaner:
    def test_strips_diffusion_buzzwords(self):
        prompt = "Cinematic 1080p wide shot of NASA rover on Moon, dramatic lighting, photorealistic 8k"
        cleaned = clean_search_query(prompt)
        assert "cinematic" not in cleaned.lower()
        assert "1080p" not in cleaned.lower()
        assert "photorealistic" not in cleaned.lower()
        assert "8k" not in cleaned.lower()
        assert "NASA" in cleaned
        assert "Moon" in cleaned

    def test_handles_empty_string(self):
        assert clean_search_query("") == ""


class TestSerpApiKeyPool:
    def test_round_robin_rotation(self):
        pool = SerpApiKeyPool(["key1", "key2", "key3"])
        assert pool.get_next_key() == "key1"
        assert pool.get_next_key() == "key2"
        assert pool.get_next_key() == "key3"
        assert pool.get_next_key() == "key1"

    def test_quota_exhaustion_failover(self):
        pool = SerpApiKeyPool(["key1", "key2"])
        pool.mark_exhausted("key1", "Out of searches")
        assert pool.get_active_keys() == ["key2"]
        assert pool.get_next_key() == "key2"

    def test_all_exhausted_returns_none(self):
        pool = SerpApiKeyPool(["key1"])
        pool.mark_exhausted("key1", "Quota exceeded")
        assert pool.get_next_key() is None


class TestSerpApiImageEngine:
    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    @patch("generator.get_resilient_session")
    def test_generate_scene_image_with_serpapi(self, mock_get_sess, mock_dl, mock_cuda, tmp_path):
        import generator as gen_module
        orig_cache = gen_module._CACHE_DIR
        gen_module._CACHE_DIR = tmp_path / "cache"

        try:
            mock_session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "images_results": [
                    {"original": "https://example.com/test_image.jpg", "title": "Test Title"}
                ]
            }
            mock_session.get.return_value = mock_resp
            mock_get_sess.return_value = mock_session
            mock_dl.return_value = _tiny_png_bytes(200, 200)

            pool = SerpApiKeyPool(["test_key_123"])
            gen_module._KEY_POOL = pool

            img = generate_scene_image("Budget smartphones in retail showroom")
            assert img is not None
            assert img.size == (1024, 576)
            mock_session.get.assert_called()
            called_params = mock_session.get.call_args[1]["params"]
            assert called_params["engine"] == "google_images"
            assert called_params["api_key"] == "test_key_123"
        finally:
            gen_module._CACHE_DIR = orig_cache

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    @patch("generator.get_resilient_session")
    def test_disk_caching_avoids_api_calls(self, mock_get_sess, mock_dl, mock_cuda, tmp_path):
        import generator as gen_module
        orig_cache = gen_module._CACHE_DIR
        gen_module._CACHE_DIR = tmp_path / "cache"

        try:
            mock_session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "images_results": [{"original": "https://example.com/phone.jpg"}]
            }
            mock_session.get.return_value = mock_resp
            mock_get_sess.return_value = mock_session
            mock_dl.return_value = _tiny_png_bytes(200, 200)

            gen_module._KEY_POOL = SerpApiKeyPool(["test_key_cache"])

            # 1st call: hits API and writes to disk cache
            img1 = generate_scene_image("Flagship smartphone release")
            assert img1 is not None
            api_calls_count = mock_session.get.call_count

            # 2nd call with identical prompt: loads from disk cache, 0 new API calls
            img2 = generate_scene_image("Flagship smartphone release")
            assert img2 is not None
            assert mock_session.get.call_count == api_calls_count
        finally:
            gen_module._CACHE_DIR = orig_cache

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    def test_photographic_fallback_when_no_serpapi_keys(self, mock_dl, mock_cuda):
        import generator as gen_module
        orig_pool = gen_module._KEY_POOL
        gen_module._KEY_POOL = SerpApiKeyPool([])

        try:
            mock_dl.return_value = _tiny_png_bytes(200, 200)
            img = generate_scene_image("Space telescope observing galaxy")
            assert img is not None
            assert img.size == (1024, 576)
            called_url = mock_dl.call_args[1]["url"]
            assert "picsum.photos" in called_url
        finally:
            gen_module._KEY_POOL = orig_pool


class TestLocalGPUPath:
    @patch("generator._cuda_available", return_value=True)
    @patch("generator.get_pipeline")
    def test_local_gpu_mode_when_cuda(self, mock_pipeline, mock_cuda):
        import generator as gen_module
        orig_pool = gen_module._KEY_POOL
        gen_module._KEY_POOL = SerpApiKeyPool([])  # no cloud keys

        try:
            fake_image = Image.new("RGB", (1024, 576), color=(0, 255, 0))
            mock_result = MagicMock()
            mock_result.images = [fake_image]
            mock_pipe = MagicMock()
            mock_pipe.return_value = mock_result
            mock_pipeline.return_value = mock_pipe

            fake_torch = MagicMock()
            fake_torch.cuda.is_available.return_value = True
            gen_module._torch = fake_torch
            gen_module._pipeline_model_id = gen_module._PRIMARY_MODEL

            result = generate_scene_image("A coastal cityscape")
            assert result is not None
        finally:
            gen_module._KEY_POOL = orig_pool


class TestSaveSceneImage:
    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    def test_saves_file_to_output_dir(self, mock_stream, mock_cuda, tmp_path):
        import generator as gen_module
        orig_pool = gen_module._KEY_POOL
        gen_module._KEY_POOL = SerpApiKeyPool([])

        try:
            mock_stream.return_value = _tiny_png_bytes(200, 200)
            saved_path = save_scene_image(
                prompt="Cityscape at dusk",
                scene_index=3,
                output_dir=str(tmp_path),
            )
            assert saved_path is not None
            expected = tmp_path / "scene_3.png"
            assert expected.exists()
            img = Image.open(str(expected))
            assert img.size == (1024, 576)
        finally:
            gen_module._KEY_POOL = orig_pool

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    def test_output_dir_created_automatically(self, mock_stream, mock_cuda, tmp_path):
        import generator as gen_module
        orig_pool = gen_module._KEY_POOL
        gen_module._KEY_POOL = SerpApiKeyPool([])

        try:
            mock_stream.return_value = _tiny_png_bytes(200, 200)
            deep_dir = tmp_path / "sub" / "folder"
            assert not deep_dir.exists()

            saved_path = save_scene_image(
                prompt="Ocean waves",
                scene_index=1,
                output_dir=str(deep_dir),
            )
            assert deep_dir.exists()
            assert (deep_dir / "scene_1.png").exists()
        finally:
            gen_module._KEY_POOL = orig_pool

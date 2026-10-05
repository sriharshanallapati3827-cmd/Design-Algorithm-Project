"""
tests/test_image_generation.py
===============================
Tests for generator.py dual-mode image generation.
"""

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


def _tiny_png_bytes() -> bytes:
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestCloudFallback:
    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    def test_generate_scene_image_uses_pollinations_when_no_cuda(
        self, mock_stream, mock_cuda
    ):
        from generator import generate_scene_image

        mock_stream.return_value = _tiny_png_bytes()
        result = generate_scene_image("A lunar landscape, cinematic")

        assert result is not None
        assert isinstance(result, Image.Image)
        mock_stream.assert_called_once()
        called_url = mock_stream.call_args.kwargs.get("url") or mock_stream.call_args[1].get("url") or mock_stream.call_args[0][0]
        assert "image.pollinations.ai" in called_url

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    def test_prompt_is_url_encoded_in_request(self, mock_stream, mock_cuda):
        from generator import generate_scene_image

        mock_stream.return_value = _tiny_png_bytes()
        generate_scene_image("Astronaut & robot: 50/50 split, 1080p")

        called_url = mock_stream.call_args.kwargs.get("url") or mock_stream.call_args[1].get("url") or mock_stream.call_args[0][0]
        assert " " not in called_url
        assert "&" not in called_url

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    def test_http_error_returns_none_gracefully(self, mock_stream, mock_cuda):
        import requests as _req
        from generator import generate_scene_image

        mock_stream.side_effect = _req.exceptions.HTTPError("500 Server Error")
        result = generate_scene_image("Some prompt")
        assert result is None

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    def test_network_error_returns_none_gracefully(self, mock_stream, mock_cuda):
        import requests as _req
        from generator import generate_scene_image

        mock_stream.side_effect = _req.exceptions.ConnectionError("DNS failure")
        result = generate_scene_image("Lunar surface")
        assert result is None

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    @patch.dict(os.environ, {"POLLINATIONS_API_KEY": "test-secret-key"})
    def test_api_key_sent_as_bearer_token(self, mock_stream, mock_cuda):
        import generator as gen_module
        gen_module.POLLINATIONS_API_KEY = "test-secret-key"

        mock_stream.return_value = _tiny_png_bytes()
        gen_module.generate_scene_image("Space station")

        headers = mock_stream.call_args.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer test-secret-key"
        gen_module.POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    def test_no_api_key_sends_no_auth_header(self, mock_stream, mock_cuda):
        import generator as gen_module
        original_key = gen_module.POLLINATIONS_API_KEY
        gen_module.POLLINATIONS_API_KEY = None

        mock_stream.return_value = _tiny_png_bytes()
        gen_module.generate_scene_image("Rocket launch")

        headers = mock_stream.call_args.kwargs.get("headers", {})
        assert "Authorization" not in headers
        gen_module.POLLINATIONS_API_KEY = original_key


class TestLocalGPUPath:
    @patch("generator._cuda_available", return_value=True)
    @patch("generator.stream_download_bytes")
    @patch("generator.get_pipeline")
    def test_local_gpu_mode_does_not_call_pollinations(
        self, mock_pipeline, mock_stream, mock_cuda
    ):
        from generator import generate_scene_image

        fake_image = Image.new("RGB", (4, 4), color=(0, 255, 0))
        mock_result = MagicMock()
        mock_result.images = [fake_image]
        mock_pipe = MagicMock()
        mock_pipe.return_value = mock_result
        mock_pipeline.return_value = mock_pipe

        import generator as gen_module
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        gen_module._torch = fake_torch
        gen_module._pipeline_model_id = gen_module._PRIMARY_MODEL

        result = generate_scene_image("A coastal cityscape")
        assert result is not None
        mock_stream.assert_not_called()


class TestSaveSceneImage:
    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    def test_saves_file_to_output_dir_cloud_mode(
        self, mock_stream, mock_cuda, tmp_path
    ):
        from generator import save_scene_image

        mock_stream.return_value = _tiny_png_bytes()
        saved_path = save_scene_image(
            prompt="Cityscape at dusk",
            scene_index=3,
            output_dir=str(tmp_path),
        )

        assert saved_path is not None
        expected = tmp_path / "scene_3.png"
        assert expected.exists()
        img = Image.open(str(expected))
        assert img.mode == "RGB"

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    def test_default_output_dir_is_output_folder(self, mock_stream, mock_cuda, tmp_path):
        import generator as gen_module
        from generator import save_scene_image

        mock_stream.return_value = _tiny_png_bytes()
        orig_dir = gen_module._OUTPUT_DIR
        gen_module._OUTPUT_DIR = tmp_path / "output"

        try:
            saved_path = save_scene_image(prompt="Mountain sunset", scene_index=1)
        finally:
            gen_module._OUTPUT_DIR = orig_dir

        assert saved_path is not None
        assert Path(saved_path).name == "scene_1.png"

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    def test_returns_none_when_generation_fails(self, mock_stream, mock_cuda, tmp_path):
        import requests as _req
        from generator import save_scene_image

        mock_stream.side_effect = _req.exceptions.ConnectionError("offline")
        result = save_scene_image(
            prompt="Storm clouds", scene_index=2, output_dir=str(tmp_path)
        )
        assert result is None

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.stream_download_bytes")
    def test_output_dir_created_automatically(self, mock_stream, mock_cuda, tmp_path):
        from generator import save_scene_image

        mock_stream.return_value = _tiny_png_bytes()
        deep_dir = tmp_path / "new" / "nested" / "dir"
        assert not deep_dir.exists()

        save_scene_image(prompt="Ocean waves", scene_index=7, output_dir=str(deep_dir))
        assert deep_dir.exists()
        assert (deep_dir / "scene_7.png").exists()

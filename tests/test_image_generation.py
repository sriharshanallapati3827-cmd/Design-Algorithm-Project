"""
tests/test_image_generation.py
===============================
Tests for generator.py dual-mode image generation:

  * TestCloudFallback   — CUDA mocked to False  → Pollinations HTTP path
  * TestLocalGPUPath    — CUDA mocked to True   → local Stable Diffusion path
  * TestSaveSceneImage  — save_scene_image() writes output/scene_X.png
"""

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helper — build a tiny 1×1 red PIL image and return it as a response mock
# ---------------------------------------------------------------------------

def _tiny_png_bytes() -> bytes:
    """Return raw PNG bytes of a 1×1 red pixel."""
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_requests_response(png_bytes: bytes, status_code: int = 200):
    """Build a minimal mock that mimics a requests.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.content = png_bytes
    if status_code >= 400:
        from requests.exceptions import HTTPError
        mock_resp.raise_for_status.side_effect = HTTPError(
            f"{status_code} Error", response=mock_resp
        )
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


# ===========================================================================
# TestCloudFallback
# ===========================================================================

class TestCloudFallback:
    """
    Verify that when torch.cuda.is_available() returns False the app
    downloads the scene image over HTTP (Pollinations API) without crashing.
    """

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.requests.get")
    def test_generate_scene_image_uses_pollinations_when_no_cuda(
        self, mock_get, mock_cuda
    ):
        """generate_scene_image() hits Pollinations endpoint when CUDA is off."""
        from generator import generate_scene_image

        png = _tiny_png_bytes()
        mock_get.return_value = _make_requests_response(png)

        result = generate_scene_image("A lunar landscape, cinematic")

        assert result is not None, "Expected a PIL Image, got None"
        assert isinstance(result, Image.Image)
        mock_get.assert_called_once()

        # URL must point at the Pollinations endpoint
        called_url = mock_get.call_args[0][0]
        assert "image.pollinations.ai" in called_url, (
            f"Expected Pollinations URL, got: {called_url}"
        )

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.requests.get")
    def test_prompt_is_url_encoded_in_request(self, mock_get, mock_cuda):
        """Special characters in the prompt must be percent-encoded in the URL."""
        from generator import generate_scene_image

        mock_get.return_value = _make_requests_response(_tiny_png_bytes())
        generate_scene_image("Astronaut & robot: 50/50 split, 1080p")

        called_url = mock_get.call_args[0][0]
        # Space and special chars must NOT appear literally
        assert " " not in called_url
        assert "&" not in called_url

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.requests.get")
    def test_http_error_returns_none_gracefully(self, mock_get, mock_cuda):
        """A 500 from Pollinations must return None, not raise."""
        from generator import generate_scene_image

        mock_get.return_value = _make_requests_response(b"", status_code=500)

        result = generate_scene_image("Some prompt")
        assert result is None, "Expected None on HTTP 500, got an image"

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.requests.get")
    def test_network_error_returns_none_gracefully(self, mock_get, mock_cuda):
        """A ConnectionError from requests must return None, not propagate."""
        import requests as _req
        from generator import generate_scene_image

        mock_get.side_effect = _req.exceptions.ConnectionError("DNS failure")

        result = generate_scene_image("Lunar surface")
        assert result is None, "Expected None on ConnectionError"

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.requests.get")
    @patch.dict(os.environ, {"POLLINATIONS_API_KEY": "test-secret-key"})
    def test_api_key_sent_as_bearer_token(self, mock_get, mock_cuda):
        """When POLLINATIONS_API_KEY is set it must appear in the Auth header."""
        import generator as gen_module
        # Reload the module-level variable to pick up the patched env var
        gen_module.POLLINATIONS_API_KEY = "test-secret-key"

        mock_get.return_value = _make_requests_response(_tiny_png_bytes())
        gen_module.generate_scene_image("Space station")

        _, call_kwargs = mock_get.call_args
        headers = call_kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer test-secret-key", (
            "Expected Authorization header with Bearer token"
        )

        # Restore
        gen_module.POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.requests.get")
    def test_no_api_key_sends_no_auth_header(self, mock_get, mock_cuda):
        """Without a key, no Authorization header must be sent."""
        import generator as gen_module

        original_key = gen_module.POLLINATIONS_API_KEY
        gen_module.POLLINATIONS_API_KEY = None

        mock_get.return_value = _make_requests_response(_tiny_png_bytes())
        gen_module.generate_scene_image("Rocket launch")

        _, call_kwargs = mock_get.call_args
        headers = call_kwargs.get("headers", {})
        assert "Authorization" not in headers

        gen_module.POLLINATIONS_API_KEY = original_key


# ===========================================================================
# TestLocalGPUPath
# ===========================================================================

class TestLocalGPUPath:
    """When CUDA IS available, generate_scene_image() must NOT call requests.get."""

    @patch("generator._cuda_available", return_value=True)
    @patch("generator.requests.get")
    @patch("generator.get_pipeline")
    def test_local_gpu_mode_does_not_call_pollinations(
        self, mock_pipeline, mock_get, mock_cuda
    ):
        """Local GPU path must never hit the Pollinations API."""
        from generator import generate_scene_image

        # Build a fake pipeline that returns a tiny PIL image
        fake_image = Image.new("RGB", (4, 4), color=(0, 255, 0))
        mock_result = MagicMock()
        mock_result.images = [fake_image]
        mock_pipe = MagicMock()
        mock_pipe.return_value = mock_result
        mock_pipeline.return_value = mock_pipe

        # Provide a torch-like mock so _ensure_imports() doesn't crash
        import generator as gen_module
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        gen_module._torch = fake_torch
        gen_module._pipeline_model_id = gen_module._PRIMARY_MODEL

        result = generate_scene_image("A coastal cityscape")

        # Should return an image
        assert result is not None
        # Should NEVER have called the Pollinations endpoint
        mock_get.assert_not_called()


# ===========================================================================
# TestSaveSceneImage
# ===========================================================================

class TestSaveSceneImage:
    """
    Verify that save_scene_image() writes output/scene_X.png regardless of
    which backend was used (both paths are mocked here for isolation).
    """

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.requests.get")
    def test_saves_file_to_output_dir_cloud_mode(
        self, mock_get, mock_cuda, tmp_path
    ):
        """Cloud mode: save_scene_image writes a valid PNG to the output dir."""
        from generator import save_scene_image

        png = _tiny_png_bytes()
        mock_get.return_value = _make_requests_response(png)

        saved_path = save_scene_image(
            prompt="Cityscape at dusk",
            scene_index=3,
            output_dir=str(tmp_path),
        )

        assert saved_path is not None, "Expected a file path, got None"
        expected = tmp_path / "scene_3.png"
        assert expected.exists(), f"Expected {expected} to exist"

        # Ensure it's a readable image
        img = Image.open(str(expected))
        assert img.mode == "RGB"

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.requests.get")
    def test_default_output_dir_is_output_folder(self, mock_get, mock_cuda, tmp_path):
        """save_scene_image() defaults to 'output/' relative to cwd."""
        import generator as gen_module
        from generator import save_scene_image

        mock_get.return_value = _make_requests_response(_tiny_png_bytes())

        orig_dir = gen_module._OUTPUT_DIR
        gen_module._OUTPUT_DIR = tmp_path / "output"

        try:
            saved_path = save_scene_image(prompt="Mountain sunset", scene_index=1)
        finally:
            gen_module._OUTPUT_DIR = orig_dir

        assert saved_path is not None
        assert Path(saved_path).name == "scene_1.png"

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.requests.get")
    def test_returns_none_when_generation_fails(self, mock_get, mock_cuda, tmp_path):
        """If generation returns None, save_scene_image() must also return None."""
        import requests as _req
        from generator import save_scene_image

        mock_get.side_effect = _req.exceptions.ConnectionError("offline")

        result = save_scene_image(
            prompt="Storm clouds", scene_index=2, output_dir=str(tmp_path)
        )
        assert result is None

    @patch("generator._cuda_available", return_value=False)
    @patch("generator.requests.get")
    def test_output_dir_created_automatically(self, mock_get, mock_cuda, tmp_path):
        """save_scene_image() must mkdir the output dir if it doesn't exist."""
        from generator import save_scene_image

        mock_get.return_value = _make_requests_response(_tiny_png_bytes())
        deep_dir = tmp_path / "new" / "nested" / "dir"
        assert not deep_dir.exists()

        save_scene_image(prompt="Ocean waves", scene_index=7, output_dir=str(deep_dir))

        assert deep_dir.exists(), "Expected output dir to be created automatically"
        assert (deep_dir / "scene_7.png").exists()

"""
Image Generation Engine — AI NEWS Video Generator
==================================================
Dual-mode image generation with automatic GPU / cloud fallback:

  LOCAL MODE  (CUDA available)
      Uses Stable Diffusion (SDXL-Turbo primary, SD 1.5 fallback) on the
      local NVIDIA GPU.  Optimised for RTX 4050 / 6 GB VRAM.

  CLOUD MODE  (no CUDA / Cloud deployment)
      Downloads images via the Pollinations.ai REST API using resilient
      chunked streaming, automatic exponential backoff, and keep-alive
      sessions to prevent middle-mile network timeouts.

Public API:
    get_pipeline()           → lazily loads the diffusion pipeline (singleton)
    generate_scene_image()   → renders a single scene from a visual prompt
    save_scene_image()       → generate + save to output/scene_X.png
    get_gpu_info()           → returns GPU availability & model status dict
"""

import io
import logging
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Optional

from PIL import Image
from dotenv import load_dotenv

from network_utils import (
    get_resilient_session,
    stream_download_bytes,
    DEFAULT_TIMEOUT,
    NETWORK_EXCEPTIONS,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cloud / Pollinations configuration
# ---------------------------------------------------------------------------

POLLINATIONS_API_KEY: str | None = os.getenv("POLLINATIONS_API_KEY")
_POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"
_OUTPUT_DIR = Path("output")


def _log(msg: str) -> None:
    """Timestamped diagnostic line to stderr (visible in terminal & Streamlit)."""
    print(f"[Generator] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Lazy imports — torch / diffusers may not be installed yet
# ---------------------------------------------------------------------------

_torch = None
_diffusers = None


def _ensure_imports():
    """Import torch and diffusers on first use, raising ImportError if missing."""
    global _torch, _diffusers
    if _torch is None:
        import torch
        _torch = torch
    if _diffusers is None:
        import diffusers
        _diffusers = diffusers


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PRIMARY_MODEL = "stabilityai/sdxl-turbo"
_FALLBACK_MODEL = "runwayml/stable-diffusion-v1-5"

# Default resolution — 16:9 widescreen landscape
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 576

# Singleton pipeline reference
_pipeline = None
_pipeline_model_id: Optional[str] = None


# ---------------------------------------------------------------------------
# CUDA availability check
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    """Return True if a CUDA-capable GPU is accessible via torch."""
    try:
        _ensure_imports()
        return bool(_torch.cuda.is_available())
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# GPU diagnostics
# ---------------------------------------------------------------------------

def get_gpu_info() -> dict:
    """Return a dict describing GPU availability and loaded model status."""
    try:
        _ensure_imports()
    except ImportError:
        return {
            "cuda_available": False,
            "device_name": "torch not installed",
            "vram_total_mb": 0,
            "vram_used_mb": 0,
            "model_loaded": None,
            "mode": "cloud_pollinations",
        }

    cuda = _cuda_available()

    if cuda:
        props = _torch.cuda.get_device_properties(0)
        return {
            "cuda_available": True,
            "device_name": props.name,
            "vram_total_mb": props.total_memory // (1024 * 1024),
            "vram_used_mb": _torch.cuda.memory_allocated(0) // (1024 * 1024),
            "model_loaded": _pipeline_model_id,
            "mode": "local_gpu",
        }
    return {
        "cuda_available": False,
        "device_name": "Cloud (Pollinations API)",
        "vram_total_mb": 0,
        "vram_used_mb": 0,
        "model_loaded": None,
        "mode": "cloud_pollinations",
    }


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def get_pipeline():
    """Lazily load the Stable Diffusion pipeline (singleton)."""
    global _pipeline, _pipeline_model_id

    if _pipeline is not None:
        return _pipeline

    _ensure_imports()
    torch = _torch
    diffusers = _diffusers

    torch.backends.cudnn.benchmark = True

    cuda_available = _cuda_available()
    dtype = torch.float16 if cuda_available else torch.float32

    # ---- Try primary model: SDXL-Turbo ----
    try:
        logger.info("Loading primary model: %s (dtype=%s)", _PRIMARY_MODEL, dtype)
        from diffusers import AutoPipelineForText2Image

        pipe = AutoPipelineForText2Image.from_pretrained(
            _PRIMARY_MODEL,
            torch_dtype=dtype,
            variant="fp16" if cuda_available else None,
        )

        if cuda_available:
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
        else:
            pipe = pipe.to("cpu")

        _pipeline = pipe
        _pipeline_model_id = _PRIMARY_MODEL
        logger.info("✅ Primary model loaded successfully.")
        return pipe

    except Exception as exc:
        logger.warning(
            "Failed to load primary model %s: %s. Trying fallback…",
            _PRIMARY_MODEL, exc,
        )

    # ---- Fallback model: SD 1.5 ----
    try:
        logger.info("Loading fallback model: %s", _FALLBACK_MODEL)
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            _FALLBACK_MODEL,
            torch_dtype=dtype,
        )

        if cuda_available:
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
        else:
            pipe = pipe.to("cpu")

        _pipeline = pipe
        _pipeline_model_id = _FALLBACK_MODEL
        logger.info("✅ Fallback model loaded successfully.")
        return pipe

    except Exception as exc:
        logger.error("Failed to load fallback model %s: %s", _FALLBACK_MODEL, exc)
        raise RuntimeError(
            f"Could not load any diffusion model. "
            f"Primary ({_PRIMARY_MODEL}) and fallback ({_FALLBACK_MODEL}) both failed. "
            f"Last error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Cloud path — Pollinations API with Chunked Streaming & Resilient Session
# ---------------------------------------------------------------------------

def _fetch_pollinations_image(
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    timeout: tuple[int, int] = DEFAULT_TIMEOUT,
) -> Optional[Image.Image]:
    """Download a generated image from the Pollinations.ai REST API.

    Uses chunked streaming and resilient retry policies with keep-alive
    headers to survive cloud proxies and latency spikes.
    """
    encoded = urllib.parse.quote(prompt, safe="")
    url = f"{_POLLINATIONS_BASE}/{encoded}"

    params: dict[str, str | int] = {
        "width": width,
        "height": height,
        "nologo": "true",
        "model": "flux",
    }

    headers: dict[str, str] = {}
    if POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"
        _log("Pollinations: using authenticated request (API key set).")
    else:
        _log("Pollinations: using public endpoint (no API key set).")

    _log(f"Pollinations chunked request → {url}  params={params} timeout={timeout}")

    try:
        session = get_resilient_session(
            total_retries=5,
            backoff_factor=2.0,
            connect_timeout=timeout[0],
            read_timeout=timeout[1],
        )
        data = stream_download_bytes(
            url=url,
            params=params,
            headers=headers,
            session=session,
            timeout=timeout,
            chunk_size=8192,
            max_attempts=2,
            fallback_delay=3.0,
        )
        img = Image.open(io.BytesIO(data)).convert("RGB")
        _log(f"Pollinations OK — streamed {len(data):,} bytes, size={img.size}")
        return img
    except Exception as exc:
        logger.error("Pollinations image fetch failed: %s", exc)
        _log(f"Pollinations Error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Unified image generation entry point
# ---------------------------------------------------------------------------

def generate_scene_image(
    prompt: str,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = 4,
) -> Optional[Image.Image]:
    """Generate a single scene image from a visual prompt.

    Automatically selects the generation backend based on hardware:
    * Local GPU: when CUDA is available (RTX 4050 / SDXL-Turbo / SD 1.5).
    * Cloud Pollinations API: when CUDA is unavailable, using chunked streaming.
    """
    if not _cuda_available():
        _log("CUDA not available → using Pollinations cloud API.")
        return _fetch_pollinations_image(prompt, width=width, height=height)

    # ── Local GPU mode ───────────────────────────────────────────────────
    _log("CUDA available → using local Stable Diffusion pipeline.")
    try:
        _ensure_imports()
        torch = _torch
    except ImportError:
        logger.error("torch/diffusers not installed — cannot generate images locally.")
        return None

    try:
        pipe = get_pipeline()

        if _pipeline_model_id == _PRIMARY_MODEL:
            guidance_scale = 0.0
        else:
            guidance_scale = 7.5
            num_inference_steps = max(num_inference_steps, 20)

        if _pipeline_model_id == _FALLBACK_MODEL:
            height = min(height, 512)
            width = min(width, 512)

        logger.info(
            "Generating image: %dx%d, steps=%d, guidance=%.1f",
            width, height, num_inference_steps, guidance_scale,
        )

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

        image = result.images[0]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return image

    except Exception as exc:
        logger.error("Local GPU image generation failed: %s", exc, exc_info=True)
        try:
            if _torch and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# save_scene_image
# ---------------------------------------------------------------------------

def save_scene_image(
    prompt: str,
    scene_index: int,
    output_dir: Optional[str] = None,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = 4,
) -> Optional[str]:
    """Generate a scene image and save it to ``output/scene_<N>.png``."""
    dest_dir = Path(output_dir) if output_dir else _OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"scene_{scene_index}.png"

    _log(f"save_scene_image: scene={scene_index}  dest={dest_path}")

    image = generate_scene_image(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
    )

    if image is None:
        logger.error("save_scene_image: generation returned None for scene %d", scene_index)
        return None

    image.save(str(dest_path), format="PNG")
    _log(f"save_scene_image: saved {dest_path}  size={image.size}")
    return str(dest_path.resolve())


# ---------------------------------------------------------------------------
# Convenience: image → PNG bytes
# ---------------------------------------------------------------------------

def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL Image to raw bytes (PNG by default)."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()

"""
Image Generation Engine — AI NEWS Video Generator
==================================================
Dual-mode image generation with automatic GPU / cloud fallback:

  LOCAL MODE  (CUDA available)
      Uses Stable Diffusion (SDXL-Turbo primary, SD 1.5 fallback) on the
      local NVIDIA GPU.  Optimised for RTX 4050 / 6 GB VRAM.

  CLOUD MODE  (no CUDA)
      Downloads images via the Pollinations.ai REST API:
          GET https://image.pollinations.ai/prompt/{prompt}
      Authentication is optional: set POLLINATIONS_API_KEY in .env for
      higher rate-limits / private generations.

Public API:
    get_pipeline()           → lazily loads the diffusion pipeline (singleton)
    generate_scene_image()   → renders a single scene from a visual prompt
    save_scene_image()       → generate + save to output/scene_X.png
    get_gpu_info()           → returns GPU availability & model status dict

VRAM Budget (RTX 4050 — 6 GB):
    • torch.float16 precision throughout
    • pipe.enable_model_cpu_offload()   — keeps only the active sub-model on GPU
    • pipe.enable_attention_slicing()   — trades compute for ~30 % less peak VRAM
    • torch.backends.cudnn.benchmark   — auto-tunes conv kernels for fixed sizes
"""

import io
import logging
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cloud / Pollinations configuration
# ---------------------------------------------------------------------------

# API key is optional for Pollinations — omitting it uses the public endpoint.
POLLINATIONS_API_KEY: str | None = os.getenv("POLLINATIONS_API_KEY")

# Pollinations image endpoint (prompt goes in the URL path)
_POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"

# Default output directory — created automatically if absent
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
# CUDA availability check — single source of truth for both generation
# logic AND the UI status panel.  Mockable in tests via:
#   patch("generator._cuda_available", return_value=False)
# Also respects CUDA_VISIBLE_DEVICES="-1" set in the shell before startup.
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    """Return True if a CUDA-capable GPU is accessible via torch.

    This is the single authoritative check used by both the UI status panel
    (``get_gpu_info``) and the generation backend selector
    (``generate_scene_image``).  Keeping it in one place means that
    setting ``CUDA_VISIBLE_DEVICES=-1`` in the shell — or mocking it in
    tests — automatically affects both the sidebar display AND which
    image-generation path is taken.
    """
    try:
        _ensure_imports()
        return bool(_torch.cuda.is_available())  # type: ignore[union-attr]
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# GPU diagnostics
# ---------------------------------------------------------------------------

def get_gpu_info() -> dict:
    """Return a dict describing GPU availability and loaded model status.

    Uses ``_cuda_available()`` so the sidebar reflects the same truth as
    the image-generation backend selector.  Setting
    ``CUDA_VISIBLE_DEVICES=-1`` before starting Streamlit will therefore
    show "Cloud (Pollinations)" in the UI *and* route generation there.

    Keys:
        cuda_available (bool): Whether CUDA is available.
        device_name (str): GPU device name, "Cloud (Pollinations)", or
                           "torch not installed".
        vram_total_mb (int): Total VRAM in MB (0 if no CUDA).
        vram_used_mb (int): Currently allocated VRAM in MB.
        model_loaded (str | None): ID of the loaded diffusion model.
        mode (str): "local_gpu" or "cloud_pollinations".
    """
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

    # ── Use the shared wrapper — NOT _torch.cuda.is_available() directly ──
    cuda = _cuda_available()

    if cuda:
        props = _torch.cuda.get_device_properties(0)  # type: ignore[union-attr]
        return {
            "cuda_available": True,
            "device_name": props.name,
            "vram_total_mb": props.total_memory // (1024 * 1024),
            "vram_used_mb": _torch.cuda.memory_allocated(0) // (1024 * 1024),  # type: ignore[union-attr]
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
    """Lazily load the Stable Diffusion pipeline (singleton).

    Tries SDXL-Turbo first; falls back to SD 1.5 if VRAM is too tight
    or if the model fails to load.

    Returns
    -------
    pipe : diffusers.DiffusionPipeline
        Ready-to-use pipeline on the appropriate device.
    """
    global _pipeline, _pipeline_model_id

    if _pipeline is not None:
        return _pipeline

    _ensure_imports()
    torch = _torch
    diffusers = _diffusers

    # Enable cuDNN auto-tuner for fixed input sizes
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
# Image generation
# ---------------------------------------------------------------------------

# _cuda_available() is defined near the top of this file (before get_gpu_info)
# so it can be used by both the UI diagnostics and the generation logic.


# ---------------------------------------------------------------------------
# Cloud path — Pollinations API
# ---------------------------------------------------------------------------

def _fetch_pollinations_image(
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    timeout: int = 60,
) -> Optional[Image.Image]:
    """Download a generated image from the Pollinations.ai REST API.

    Parameters
    ----------
    prompt : str
        Visual description for the image.
    width, height : int
        Requested output dimensions (Pollinations accepts these as query params).
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    PIL.Image.Image or None
        Downloaded image, or ``None`` on any HTTP / IO error.
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

    _log(f"Pollinations request → {url}  params={params}")

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        _log(f"Pollinations OK — received {len(resp.content):,} bytes, size={img.size}")
        return img
    except requests.exceptions.RequestException as exc:
        logger.error("Pollinations HTTP request failed: %s", exc)
        return None
    except Exception as exc:
        logger.error("Pollinations image decode failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Unified local GPU path
# ---------------------------------------------------------------------------

def generate_scene_image(
    prompt: str,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = 4,
) -> Optional[Image.Image]:
    """Generate a single scene image from a visual prompt.

    Automatically selects the generation backend based on hardware:

    * **Local GPU** — when ``torch.cuda.is_available()`` returns ``True``.
      Uses Stable Diffusion (SDXL-Turbo → SD 1.5 fallback) on the GPU.
    * **Cloud (Pollinations API)** — when CUDA is unavailable.
      Downloads from ``https://image.pollinations.ai/prompt/{prompt}``.
      Set ``POLLINATIONS_API_KEY`` in ``.env`` for authenticated access.

    Parameters
    ----------
    prompt : str
        The visual description / image-generation prompt for this scene.
    height : int
        Output image height in pixels (default 576).
    width : int
        Output image width in pixels (default 1024).
    num_inference_steps : int
        Number of denoising steps — only used in local GPU mode
        (default 4 for SDXL-Turbo; auto-bumped to 20 for SD 1.5).

    Returns
    -------
    PIL.Image.Image or None
        The generated RGB image, or ``None`` if generation failed.
    """
    if not _cuda_available():
        # ── Cloud mode: Pollinations API ─────────────────────────────────
        _log("CUDA not available → using Pollinations cloud API.")
        return _fetch_pollinations_image(prompt, width=width, height=height)

    # ── Local GPU mode: Stable Diffusion ─────────────────────────────────
    _log("CUDA available → using local Stable Diffusion pipeline.")
    try:
        _ensure_imports()
        torch = _torch
    except ImportError:
        logger.error("torch/diffusers not installed — cannot generate images locally.")
        return None

    try:
        pipe = get_pipeline()

        # SDXL-Turbo works best without guidance (CFG=0), SD 1.5 needs CFG ~7.5
        if _pipeline_model_id == _PRIMARY_MODEL:
            guidance_scale = 0.0
        else:
            guidance_scale = 7.5
            # SD 1.5 needs more steps for decent quality
            num_inference_steps = max(num_inference_steps, 20)

        # SD 1.5 natively generates 512×512; clamp resolution for it
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

        # Free VRAM after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return image

    except Exception as exc:
        logger.error("Local GPU image generation failed: %s", exc, exc_info=True)
        # Attempt VRAM cleanup even on failure
        try:
            if _torch and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# save_scene_image — unified entry point that also persists to disk
# ---------------------------------------------------------------------------

def save_scene_image(
    prompt: str,
    scene_index: int,
    output_dir: Optional[str] = None,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = 4,
) -> Optional[str]:
    """Generate a scene image and save it to ``output/scene_<N>.png``.

    Both the local-GPU and cloud-API code paths converge here so that
    MoviePy's video assembly always finds the file at the same path
    regardless of which backend was used.

    Parameters
    ----------
    prompt : str
        Visual description for the scene.
    scene_index : int
        1-based scene number — used to build the filename
        (``output/scene_1.png``, ``output/scene_2.png``, …).
    output_dir : str or None
        Directory to write the PNG into.  Defaults to ``output/`` relative
        to the current working directory.
    height, width : int
        Output dimensions.
    num_inference_steps : int
        Denoising steps (local GPU mode only).

    Returns
    -------
    str or None
        Absolute path to the saved PNG, or ``None`` if generation failed.
    """
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

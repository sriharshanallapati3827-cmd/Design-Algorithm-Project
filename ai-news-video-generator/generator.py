"""
Local GPU Image Engine — AI NEWS Video Generator
=================================================
Generates AI scene images using Stable Diffusion (SDXL-Turbo primary,
SD 1.5 fallback) optimized for NVIDIA RTX 4050 with 6 GB VRAM.

Public API:
    get_pipeline()           → lazily loads the diffusion pipeline (singleton)
    generate_scene_image()   → renders a single scene from a visual prompt
    get_gpu_info()           → returns GPU availability & model status dict

VRAM Budget (RTX 4050 — 6 GB):
    • torch.float16 precision throughout
    • pipe.enable_model_cpu_offload()   — keeps only the active sub-model on GPU
    • pipe.enable_attention_slicing()   — trades compute for ~30 % less peak VRAM
    • torch.backends.cudnn.benchmark   — auto-tunes conv kernels for fixed sizes
"""

import io
import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

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
# GPU diagnostics
# ---------------------------------------------------------------------------

def get_gpu_info() -> dict:
    """Return a dict describing GPU availability and loaded model status.

    Keys:
        cuda_available (bool): Whether CUDA is available.
        device_name (str): GPU device name or "CPU".
        vram_total_mb (int): Total VRAM in MB (0 if CPU).
        vram_used_mb (int): Currently allocated VRAM in MB.
        model_loaded (str | None): ID of the loaded diffusion model.
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
        }

    cuda = _torch.cuda.is_available()
    if cuda:
        props = _torch.cuda.get_device_properties(0)
        return {
            "cuda_available": True,
            "device_name": props.name,
            "vram_total_mb": props.total_memory // (1024 * 1024),
            "vram_used_mb": _torch.cuda.memory_allocated(0) // (1024 * 1024),
            "model_loaded": _pipeline_model_id,
        }
    return {
        "cuda_available": False,
        "device_name": "CPU",
        "vram_total_mb": 0,
        "vram_used_mb": 0,
        "model_loaded": _pipeline_model_id,
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

    cuda_available = torch.cuda.is_available()
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

def generate_scene_image(
    prompt: str,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = 4,
) -> Optional[Image.Image]:
    """Generate a single scene image from a visual prompt.

    Parameters
    ----------
    prompt : str
        The visual description / image-generation prompt for this scene.
    height : int
        Output image height in pixels (default 576).
    width : int
        Output image width in pixels (default 1024).
    num_inference_steps : int
        Number of denoising steps (default 4 for SDXL-Turbo).

    Returns
    -------
    PIL.Image.Image or None
        The generated RGB image, or ``None`` if generation failed.
    """
    try:
        _ensure_imports()
        torch = _torch
    except ImportError:
        logger.error("torch/diffusers not installed — cannot generate images.")
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
        logger.error("Image generation failed: %s", exc, exc_info=True)
        # Attempt VRAM cleanup even on failure
        try:
            if _torch and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Convenience: image → PNG bytes
# ---------------------------------------------------------------------------

def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL Image to raw bytes (PNG by default)."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()

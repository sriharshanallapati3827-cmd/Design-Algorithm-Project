"""
Image Generation Engine — AI NEWS Video Generator
==================================================
Multi-mode image generation with SerpApi Google Images engine,
local CUDA GPU acceleration, and photographic fallback:

  1. SERPAPI GOOGLE IMAGES (Primary Cloud Engine)
      Searches real-world Google Images via SerpApi using entity-grounded
      keywords extracted from the news story. Supports multi-key pooling,
      round-robin load balancing, automatic quota-exhaustion failover,
      and disk caching in output/cache/.

  2. LOCAL GPU MODE (CUDA available)
      Uses Stable Diffusion (SDXL-Turbo primary, SD 1.5 fallback) on the
      local NVIDIA GPU. Optimised for RTX 4050 / 6 GB VRAM.

  3. PHOTOGRAPHIC FALLBACK (Network resilience)
      Deterministic high-resolution (1024x576) photographic CDN (Picsum)
      keyed by prompt MD5 hash so scenes are never left blank.

Public API:
    get_pipeline()           → lazily loads the diffusion pipeline (singleton)
    generate_scene_image()   → renders a single scene from a prompt or search query
    save_scene_image()       → generate + save to output/scene_X.png
    get_gpu_info()           → returns GPU availability & model status dict
    clean_search_query()     → sanitizes prompts into Google Images search keywords
    SerpApiKeyPool           → manages multi-key rotation and failover
"""

import hashlib
import io
import logging
import os
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps
from dotenv import load_dotenv

from network_utils import (
    get_resilient_session,
    stream_download_bytes,
    DEFAULT_TIMEOUT,
    NETWORK_EXCEPTIONS,
)
from utils import fit_to_broadcast_canvas

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directories & Constants
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path("output")
_CACHE_DIR = Path("output/cache")
_SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 576

_PRIMARY_MODEL = "stabilityai/sdxl-turbo"
_FALLBACK_MODEL = "runwayml/stable-diffusion-v1-5"


def _log(msg: str) -> None:
    """Timestamped diagnostic line to stderr (visible in terminal & Streamlit)."""
    print(f"[Generator] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# SerpApi Key Pooling & Automatic Quota Failover
# ---------------------------------------------------------------------------

def _load_serpapi_keys() -> list[str]:
    """Parse single or multiple comma-separated keys from .env."""
    raw = os.getenv("SERPAPI_API_KEYS") or os.getenv("SERPAPI_API_KEY") or ""
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys


class SerpApiKeyPool:
    """Manages round-robin rotation and quota-exhaustion failover for SerpApi keys."""

    def __init__(self, keys: Optional[list[str]] = None):
        if keys is not None:
            self._keys = list(keys)
        else:
            self._keys = _load_serpapi_keys()
        self._exhausted: set[str] = set()
        self._index: int = 0

    @property
    def has_keys(self) -> bool:
        return bool(self._keys)

    def reload(self) -> None:
        self._keys = _load_serpapi_keys()
        self._exhausted.clear()
        self._index = 0

    def get_active_keys(self) -> list[str]:
        return [k for k in self._keys if k not in self._exhausted]

    def get_next_key(self) -> Optional[str]:
        active = self.get_active_keys()
        if not active:
            return None
        self._index = self._index % len(active)
        key = active[self._index]
        self._index = (self._index + 1) % len(active)
        return key

    def mark_exhausted(self, key: str, reason: str = "") -> None:
        self._exhausted.add(key)
        masked = f"...{key[-6:]}" if len(key) >= 6 else key
        _log(f"SerpApi key {masked} marked exhausted: {reason} "
             f"({len(self.get_active_keys())}/{len(self._keys)} active keys remaining)")

    def reset_exhausted(self) -> None:
        self._exhausted.clear()


_KEY_POOL = SerpApiKeyPool()


# ---------------------------------------------------------------------------
# Search Query Cleaner
# ---------------------------------------------------------------------------

def clean_search_query(prompt: str) -> str:
    """Strip diffusion buzzwords and punctuation so Google Images gets clean keywords."""
    if not prompt:
        return ""
    buzzwords = [
        r"\b1080p\b", r"\b4k\b", r"\b8k\b", r"\bhd\b", r"\bhigh resolution\b",
        r"\bphotorealistic\b", r"\bhyperrealistic\b", r"\bcinematic\b",
        r"\bdramatic lighting\b", r"\bwide shot\b", r"\blandscape\b",
        r"\bdepth of field\b", r"\bunreal engine\b", r"\boctane render\b",
        r"\bmasterpiece\b", r"\btrending on artstation\b", r"\bclose up\b",
        r"\bclose-up\b", r"\baward winning\b",
    ]
    cleaned = prompt
    for bw in buzzwords:
        cleaned = re.sub(bw, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[,;:\-–—\(\)\[\]\"']+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    if len(words) > 8:
        cleaned = " ".join(words[:8])
    return cleaned


# ---------------------------------------------------------------------------
# Lazy imports — torch / diffusers
# ---------------------------------------------------------------------------

_torch = None
_diffusers = None
_pipeline = None
_pipeline_model_id: Optional[str] = None


def _ensure_imports():
    """Import torch and diffusers on first use, raising ImportError if missing."""
    global _torch, _diffusers
    if _torch is None:
        import torch
        _torch = torch
    if _diffusers is None:
        import diffusers
        _diffusers = diffusers


def _cuda_available() -> bool:
    """Return True if a CUDA-capable GPU is accessible via torch."""
    try:
        _ensure_imports()
        return bool(_torch.cuda.is_available())
    except (ImportError, Exception):
        return False


def get_gpu_info() -> dict:
    """Return a dict describing GPU availability and loaded model status."""
    try:
        _ensure_imports()
    except (ImportError, Exception):
        return {
            "cuda_available": False,
            "device_name": "torch not installed",
            "vram_total_mb": 0,
            "vram_used_mb": 0,
            "model_loaded": None,
            "mode": "serpapi_google_images",
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
        "device_name": "SerpApi Google Images Engine",
        "vram_total_mb": 0,
        "vram_used_mb": 0,
        "model_loaded": None,
        "mode": "serpapi_google_images",
    }


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

    # Try primary model: SDXL-Turbo
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
        logger.info("Primary model loaded successfully.")
        return pipe

    except Exception as exc:
        logger.warning(
            "Failed to load primary model %s: %s. Trying fallback…",
            _PRIMARY_MODEL, exc,
        )

    # Fallback model: SD 1.5
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
        logger.info("Fallback model loaded successfully.")
        return pipe

    except Exception as exc:
        logger.error("Failed to load fallback model %s: %s", _FALLBACK_MODEL, exc)
        raise RuntimeError(
            f"Could not load any diffusion model. "
            f"Primary ({_PRIMARY_MODEL}) and fallback ({_FALLBACK_MODEL}) both failed. "
            f"Last error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# SerpApi Google Images Fetcher
# ---------------------------------------------------------------------------

def _fetch_serpapi_google_image(
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    timeout: tuple[int, int] = DEFAULT_TIMEOUT,
    key_pool: Optional[SerpApiKeyPool] = None,
) -> Optional[Image.Image]:
    """Search Google Images via SerpApi with key pooling, disk cache, and 16:9 fitting."""
    pool = key_pool or _KEY_POOL
    if not pool.has_keys:
        _log("No SerpApi keys configured in .env.")
        return None

    query = clean_search_query(prompt)
    if not query:
        return None

    # Check local disk cache
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"{query}_{width}x{height}".encode("utf-8")).hexdigest()
    cache_path = _CACHE_DIR / f"{cache_key}.png"
    if cache_path.exists():
        try:
            cached_img = Image.open(str(cache_path)).convert("RGB")
            _log(f"SerpApi Cache HIT for query={query!r} -> {cache_path}")
            return cached_img
        except Exception:
            pass

    session = get_resilient_session(
        total_retries=3,
        backoff_factor=1.5,
        connect_timeout=timeout[0],
        read_timeout=timeout[1],
    )

    while pool.get_active_keys():
        api_key = pool.get_next_key()
        if not api_key:
            break

        params = {
            "engine": "google_images",
            "q": query,
            "api_key": api_key,
            "safe": "active",
            "tbs": "iar:w",  # Restrict to wide/landscape 16:9 images
        }

        masked_key = f"...{api_key[-6:]}" if len(api_key) >= 6 else api_key
        _log(f"SerpApi Google Images search: query={query!r} with key={masked_key}")

        try:
            resp = session.get(_SERPAPI_SEARCH_URL, params=params, timeout=timeout)

            if resp.status_code == 429:
                pool.mark_exhausted(api_key, "HTTP 429 Rate Limit")
                continue

            resp.raise_for_status()
            resp_json = resp.json()

            if "error" in resp_json:
                err_msg = str(resp_json["error"])
                if any(x in err_msg.lower() for x in ("run out of searches", "exhausted", "quota", "invalid api key")):
                    pool.mark_exhausted(api_key, err_msg)
                    continue
                else:
                    _log(f"SerpApi error: {err_msg}")
                    continue

            results = resp_json.get("images_results", [])
            if not results and "tbs" in params:
                _log(f"No wide results for {query!r}, retrying without aspect constraint...")
                fallback_params = {k: v for k, v in params.items() if k != "tbs"}
                try:
                    fb_resp = session.get(_SERPAPI_SEARCH_URL, params=fallback_params, timeout=timeout)
                    if fb_resp.ok:
                        results = fb_resp.json().get("images_results", [])
                except Exception:
                    pass

            if not results:
                _log(f"SerpApi returned 0 image results for query={query!r}")
                return None

            # Download and validate first good candidate
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            for item in results[:10]:
                img_url = item.get("original") or item.get("thumbnail")
                if not img_url:
                    continue
                try:
                    img_data = stream_download_bytes(
                        url=img_url,
                        headers=headers,
                        session=session,
                        timeout=(5, 15),
                        chunk_size=8192,
                        max_attempts=1,
                    )
                    raw_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    # Intelligently fit to 16:9 canvas preserving faces and titles
                    fitted_img = fit_to_broadcast_canvas(raw_img, target_width=width, target_height=height)
                    try:
                        fitted_img.save(str(cache_path), format="PNG")
                    except Exception:
                        pass
                    _log(f"SerpApi OK — downloaded & fitted image from {img_url[:60]}... ({fitted_img.size})")
                    return fitted_img
                except Exception as dl_err:
                    _log(f"Candidate image skipped ({dl_err}) -> trying next...")
                    continue

            _log(f"All image candidates failed to download for query={query!r}")
            return None

        except Exception as exc:
            _log(f"SerpApi request failed with key={masked_key}: {exc}")
            continue

    _log("All SerpApi keys exhausted or failed.")
    return None


# ---------------------------------------------------------------------------
# Photographic Fallback CDN (Picsum)
# ---------------------------------------------------------------------------

def _fetch_photographic_fallback(
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    timeout: tuple[int, int] = (10, 30),
) -> Optional[Image.Image]:
    """Download a high-resolution, deterministic photographic image fallback."""
    seed = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:10]
    fallback_url = f"https://picsum.photos/seed/{seed}/{width}/{height}"
    _log(f"Photographic fallback request → {fallback_url}")
    try:
        session = get_resilient_session(
            total_retries=3,
            backoff_factor=1.5,
            connect_timeout=timeout[0],
            read_timeout=timeout[1],
        )
        data = stream_download_bytes(
            url=fallback_url,
            session=session,
            timeout=timeout,
            chunk_size=8192,
            max_attempts=2,
            fallback_delay=1.0,
        )
        raw_img = Image.open(io.BytesIO(data)).convert("RGB")
        img = fit_to_broadcast_canvas(raw_img, target_width=width, target_height=height)
        _log(f"Photographic fallback OK — streamed {len(data):,} bytes, size={img.size}")
        return img
    except Exception as exc:
        logger.error("Photographic fallback failed: %s", exc)
        _log(f"Photographic fallback Error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Unified image generation entry point
# ---------------------------------------------------------------------------

def generate_scene_image(
    prompt: str,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = 4,
    search_query: Optional[str] = None,
) -> Optional[Image.Image]:
    """Generate a single scene image from a visual prompt or real news search query.

    Tier 1: SerpApi Google Images (real news photography matched to query).
    Tier 2: Local CUDA GPU (Stable Diffusion SDXL-Turbo / SD 1.5) if available.
    Tier 3: High-Res Deterministic Photographic Fallback (Picsum CDN).
    """
    effective_query = search_query or prompt

    # Tier 1: SerpApi Google Images Engine
    if _KEY_POOL.has_keys:
        _log(f"Querying SerpApi Google Images for: {effective_query[:60]}")
        img = _fetch_serpapi_google_image(effective_query, width=width, height=height)
        if img is not None:
            return img
        _log("SerpApi search returned no image -> trying next tier...")

    # Tier 2: Local GPU mode (if CUDA available)
    if _cuda_available():
        _log("CUDA available → using local Stable Diffusion pipeline.")
        try:
            _ensure_imports()
            pipe = get_pipeline()
            guidance_scale = 0.0 if _pipeline_model_id == _PRIMARY_MODEL else 7.5
            steps = num_inference_steps if _pipeline_model_id == _PRIMARY_MODEL else max(num_inference_steps, 20)
            h = min(height, 512) if _pipeline_model_id == _FALLBACK_MODEL else height
            w = min(width, 512) if _pipeline_model_id == _FALLBACK_MODEL else width

            with _torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    height=h,
                    width=w,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                )
            image = result.images[0]
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            return image
        except Exception as exc:
            logger.error("Local GPU image generation failed: %s", exc)
            if _torch and _torch.cuda.is_available():
                _torch.cuda.empty_cache()

    # Tier 3: Photographic Fallback
    _log("Using high-resolution photographic fallback.")
    return _fetch_photographic_fallback(effective_query, width=width, height=height)


# ---------------------------------------------------------------------------
# save_scene_image & image_to_bytes
# ---------------------------------------------------------------------------

def save_scene_image(
    prompt: str,
    scene_index: int,
    output_dir: Optional[str] = None,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = 4,
    search_query: Optional[str] = None,
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
        search_query=search_query,
    )

    if image is None:
        logger.error("save_scene_image: generation returned None for scene %d", scene_index)
        return None

    image.save(str(dest_path), format="PNG")
    _log(f"save_scene_image: saved {dest_path}  size={image.size}")
    return str(dest_path.resolve())


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL Image to raw bytes (PNG by default)."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()

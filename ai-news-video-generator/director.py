"""
Director LLM Engine — AI NEWS Generator
==========================================
Generates a structured news storyboard (JSON array of scenes) by calling the
Google Gemini API with the ingested article text and target duration.

Public API:
    generate_storyboard(article_text, duration_sec, *, api_key, model_id)
"""

import json
import re
import os

from google import genai
from dotenv import load_dotenv

# Load .env so GEMINI_API_KEY is available from the environment
load_dotenv()

# ---------------------------------------------------------------------------
# WPM & scene-count math  (from PHASE2_SPEC.md)
# ---------------------------------------------------------------------------

_WPM_RATE = 2.33  # ≈ 140 words per minute → 2.33 words per second

# Reference table from spec (duration_sec → (min_scenes, max_scenes))
_SCENE_TABLE = [
    (60,  3,  4),
    (100, 5,  7),
    (120, 7,  9),
]


def _estimate_scene_range(duration_sec: int) -> tuple[int, int]:
    """Linearly interpolate the scene-count range from the spec table."""
    if duration_sec <= _SCENE_TABLE[0][0]:
        return _SCENE_TABLE[0][1], _SCENE_TABLE[0][2]
    if duration_sec >= _SCENE_TABLE[-1][0]:
        return _SCENE_TABLE[-1][1], _SCENE_TABLE[-1][2]

    for i in range(len(_SCENE_TABLE) - 1):
        d0, min0, max0 = _SCENE_TABLE[i]
        d1, min1, max1 = _SCENE_TABLE[i + 1]
        if d0 <= duration_sec <= d1:
            frac = (duration_sec - d0) / (d1 - d0)
            lo = round(min0 + frac * (min1 - min0))
            hi = round(max0 + frac * (max1 - max0))
            return lo, hi

    # Fallback (shouldn't reach here)
    return 3, 5


# ---------------------------------------------------------------------------
# Gemini model-ID mapping
# ---------------------------------------------------------------------------

MODEL_MAP: dict[str, str] = {
    "Gemini 3.6 Flash": "gemini-3.6-flash",
    "Gemini 3.5 Flash Lite": "gemini-3.5-flash-lite",
    "Gemini 2.5 Flash": "gemini-3.6-flash",  # alias for backwards compatibility
    "Gemini 1.5 Flash": "gemini-3.5-flash-lite",  # alias for backwards compatibility
}


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a professional news video director. Given a news article and \
production parameters, you produce a structured storyboard for a short-form \
news video.

RULES:
1. The total narration must fit within {target_words} words (≈{duration_sec}s \
at 140 WPM).
2. Split the story into {min_scenes} to {max_scenes} scenes.
3. Each scene needs a concise, broadcast-quality narration and a detailed \
visual prompt suitable for AI image generation (480p landscape, cinematic).
4. Timestamps must be sequential and cover the full duration.
5. Return ONLY a valid JSON array — no markdown, no commentary.

OUTPUT FORMAT (JSON array):
[
  {{
    "scene_number": 1,
    "timestamp": "00:00 - 00:15",
    "narration": "Exact spoken narration text for this scene.",
    "visual_prompt": "480p landscape cinematic description for image generation."
  }}
]
"""

_USER_PROMPT = """\
NEWS ARTICLE:
\"\"\"
{article_text}
\"\"\"

PRODUCTION PARAMETERS:
- Target duration: {duration_sec} seconds
- Target word count: ~{target_words} words
- Scene count: {min_scenes}–{max_scenes} scenes

Generate the storyboard JSON now.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_storyboard(
    article_text: str,
    duration_sec: int,
    *,
    api_key: str | None = None,
    model_name: str = "Gemini 2.5 Flash",
) -> list[dict]:
    """Call Gemini to produce a list of scene dicts for the given article.

    Parameters
    ----------
    article_text : str
        The cleaned article body text.
    duration_sec : int
        Target video length in seconds (60–120).
    api_key : str | None
        Gemini API key.  Falls back to ``GEMINI_API_KEY`` env var.
    model_name : str
        Human-readable model label (must be a key in ``MODEL_MAP``).

    Returns
    -------
    list[dict]
        Each dict has keys: ``scene_number``, ``timestamp``, ``narration``,
        ``visual_prompt``.

    Raises
    ------
    ValueError
        If no API key is available or the model name is unsupported.
    RuntimeError
        If the Gemini response cannot be parsed as valid JSON.
    """
    # --- Resolve API key ---
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError(
            "No Gemini API key provided. Set GEMINI_API_KEY in your .env file "
            "or enter it in the sidebar."
        )

    # --- Resolve model ---
    model_id = MODEL_MAP.get(model_name)
    if model_id is None:
        low = model_name.lower()
        if "lite" in low or "1.5" in low:
            model_id = "gemini-3.5-flash-lite"
        elif "gemini" in low or "flash" in low:
            model_id = "gemini-3.6-flash"
        else:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Available: {', '.join(MODEL_MAP.keys())}"
            )

    # --- Compute production parameters ---
    target_words = int(duration_sec * _WPM_RATE)
    min_scenes, max_scenes = _estimate_scene_range(duration_sec)

    system_prompt = _SYSTEM_PROMPT.format(
        target_words=target_words,
        duration_sec=duration_sec,
        min_scenes=min_scenes,
        max_scenes=max_scenes,
    )

    user_prompt = _USER_PROMPT.format(
        article_text=article_text[:8000],   # truncate very long articles
        duration_sec=duration_sec,
        target_words=target_words,
        min_scenes=min_scenes,
        max_scenes=max_scenes,
    )

    # --- Call Gemini ---
    client = genai.Client(api_key=key)

    response = client.models.generate_content(
        model=model_id,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
        ),
    )

    raw = response.text

    # --- Parse JSON ---
    scenes = _parse_scenes_json(raw)

    # Basic sanity checks
    if not scenes:
        raise RuntimeError("Gemini returned an empty scene list.")

    for i, scene in enumerate(scenes):
        for required in ("scene_number", "timestamp", "narration", "visual_prompt"):
            if required not in scene:
                raise RuntimeError(
                    f"Scene {i+1} is missing required key '{required}'."
                )

    return scenes


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_scenes_json(raw_text: str) -> list[dict]:
    """Extract and parse a JSON array from the LLM response.

    Handles common quirks:
    - Markdown code fences (```json ... ```)
    - Leading/trailing prose around the array
    """
    text = raw_text.strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to find the first [ ... ] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    raise RuntimeError(
        "Failed to parse Gemini response as a JSON scene array.\n"
        f"Raw response:\n{raw_text[:500]}"
    )

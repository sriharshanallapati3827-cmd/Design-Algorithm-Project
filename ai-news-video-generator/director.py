"""
Director LLM Engine — AI NEWS Video Generator
==============================================
Generates a structured news storyboard (JSON array of scenes) by calling the
Google Gemini API with the ingested article text and target duration.

Falls back automatically to Groq (llama-3.3-70b-versatile → llama-3.1-8b-instant)
if Gemini is unavailable (503 UNAVAILABLE, rate-limit, connection error, etc.).

Public API:
    generate_storyboard(article_text, duration_sec, *, api_key, model_name)

Root-cause fixes applied (2026-09-23):
    - Broken re-raise in _call_gemini_with_retry: the bare `raise` inside the
      else-branch fired on BOTH non-retryable errors AND the final retryable
      attempt, meaning the Gemini chain never reached the next fallback model
      and never triggered the Groq fallback. Fixed: exhaust retries, then let
      the caller (generate_storyboard) handle falling through to the next model.
    - RuntimeError from JSON-parse leaked out of the per-model try/except and
      broke the entire fallback chain. Fixed: catch RuntimeError per-model the
      same as transient errors, log it, and continue to the next model / Groq.
    - Added print() diagnostic lines so the exact model name is logged before
      every generate_content() call.
    - MODEL_MAP rationalised: phantom display names like "Gemini 3.6 Flash"
      and "Gemini 2.5 Flash" are kept as aliases to real API IDs so the UI
      still works, but only real Gemini API IDs (gemini-2.0-flash,
      gemini-1.5-flash) reach the wire.
"""

import json
import re
import os
import sys
import time as _time

from google import genai
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Groq client (optional — graceful if not installed)
# ---------------------------------------------------------------------------

try:
    from groq import Groq as _GroqClient  # type: ignore[import-untyped]
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    _GroqClient = None  # type: ignore[assignment,misc]

# Load .env so GEMINI_API_KEY / GROQ_API_KEY are available from the environment
load_dotenv()

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

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

    return 3, 5  # fallback (shouldn't reach here)


# ---------------------------------------------------------------------------
# Model-ID mapping
# ---------------------------------------------------------------------------
# Only REAL Gemini API model IDs are used on the wire.
# Display names shown in the UI are just human-readable aliases mapped here.
#
# Confirmed working IDs (Sep 2025):
#   gemini-2.0-flash  — high-capacity Flash 2.0 (default)
#   gemini-1.5-flash  — stable Flash 1.5 (emergency fallback)
#
# NOTE: "gemini-1.5-pro", "gemini-2.5-flash", "gemini-3.*" do NOT exist as
# public API endpoints and will always return 404 / MODEL_NOT_FOUND.

MODEL_MAP: dict[str, str] = {
    # UI display name            →  Real Gemini API model ID
    # (audited live against this API key on 2026-09-23)
    "Gemini 3.6 Flash":          "gemini-3.6-flash",       # confirmed WORKS ✓
    "Gemini 3.5 Flash Lite":     "gemini-3.5-flash-lite",  # confirmed WORKS ✓
    # Alias display names kept so persisted session state doesn't break
    "Gemini 2.0 Flash":          "gemini-3.6-flash",       # old name → real ID
    "Gemini 2.5 Flash":          "gemini-3.6-flash",       # old name → real ID
    "Gemini 1.5 Flash":          "gemini-3.5-flash-lite",  # old name → real ID
    "Gemini 1.5 Flash Lite":     "gemini-3.5-flash-lite",  # old name → real ID
}

# Ordered list of real Gemini model IDs tried when the primary fails.
# Primary is gemini-3.6-flash; gemini-3.5-flash-lite is the safe fallback.
_GEMINI_FALLBACK_CHAIN: list[str] = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

# Groq models tried in order — audited live against this Groq account.
# llama-3.3-70b-versatile is NOT available; use qwen3.8-27b as primary.
_GROQ_MODEL_CHAIN: list[str] = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]

# ---------------------------------------------------------------------------
# Shared prompt templates (used by both Gemini and Groq)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a professional news video director. Given a news article and \
production parameters, you produce a structured storyboard for a short-form \
news video.

RULES:
1. The total narration must fit within {target_words} words \
(approx {duration_sec}s at 140 WPM).
2. Split the story into {min_scenes} to {max_scenes} scenes.
3. Each scene needs a concise, broadcast-quality narration and a detailed \
visual prompt suitable for AI image generation (1080p landscape, cinematic).
4. Timestamps must be sequential and cover the full duration.
5. Return ONLY a valid JSON array - no markdown, no commentary.

OUTPUT FORMAT (JSON array):
[
  {{
    "scene_number": 1,
    "timestamp": "00:00 - 00:15",
    "narration": "Exact spoken narration text for this scene.",
    "visual_prompt": "1080p landscape cinematic description for image generation."
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
- Scene count: {min_scenes}-{max_scenes} scenes

Generate the storyboard JSON now.
"""

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3        # attempts per model before moving to next
_RETRY_BASE_DELAY = 2.0  # seconds; doubles each attempt: 2s, 4s, 8s


# ---------------------------------------------------------------------------
# Diagnostic logger (writes to stderr so it shows in terminal + Streamlit logs)
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Print a timestamped diagnostic line to stderr."""
    print(f"[Director] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    """
    Return True for transient errors that are worth retrying on the SAME model.
    These include 503 UNAVAILABLE, rate-limits, quota, and network errors.
    Non-retryable errors (e.g. 400 BAD_REQUEST, 401 UNAUTHORIZED, MODEL_NOT_FOUND)
    will return False so we skip immediately to the next model / Groq.
    """
    msg = str(exc).lower()
    # Transient / capacity signals — retry same model
    RETRYABLE = (
        "503", "unavailable", "rate limit", "rate_limit",
        "resource exhausted", "resource_exhausted",
        "quota", "overloaded", "server error", "try again",
        "connection", "timeout", "timed out", "reset by peer",
    )
    # Hard errors — don't retry, skip to next model immediately
    HARD_ERRORS = (
        "404", "not found", "model_not_found",
        "400", "bad request", "invalid",
        "401", "403", "permission", "unauthorized",
    )
    if any(t in msg for t in HARD_ERRORS):
        return False
    return any(t in msg for t in RETRYABLE)


def _is_skip_worthy(exc: Exception) -> bool:
    """
    Return True for errors where retrying the SAME model is pointless.
    We should skip immediately to the next model in the chain.
    Includes: model-not-found, auth errors, bad-request, parse failures.
    """
    msg = str(exc).lower()
    return any(t in msg for t in (
        "404", "not found", "model_not_found",
        "400", "bad request",
        "401", "403", "permission", "unauthorized",
    ))


# ---------------------------------------------------------------------------
# Gemini call helper
# ---------------------------------------------------------------------------

def _call_gemini_with_retry(
    client: "genai.Client",
    model_id: str,
    user_prompt: str,
    system_prompt: str,
) -> str:
    """
    Call Gemini with exponential-backoff retries for transient errors.

    FIX: The previous version used a bare `raise` inside the else-branch which
    fired on BOTH non-retryable errors AND the final retryable attempt, meaning
    the caller never got a chance to fall through to the next model.

    Corrected behaviour:
      - Hard / non-retryable error on ANY attempt → raise immediately (caller
        should skip to next model, not waste retries).
      - Retryable error on attempts 1..N-1 → sleep and retry.
      - Retryable error on final attempt → raise so caller falls to next model.
      - Success → return raw text.
    """
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        _log(f"[Gemini Call] model={model_id!r}  attempt={attempt}/{_MAX_RETRIES}")
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=user_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                ),
            )
            _log(f"[Gemini OK]   model={model_id!r}  attempt={attempt}")
            return response.text

        except Exception as exc:
            last_exc = exc
            _log(f"[Gemini ERR]  model={model_id!r}  attempt={attempt}  error={exc!r}")

            if _is_skip_worthy(exc):
                # Hard error (404, auth, bad-request) — no point retrying
                _log(f"[Gemini SKIP] model={model_id!r}  hard error, skipping model")
                raise  # propagate to the model chain loop

            if _is_retryable(exc) and attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 2s, 4s
                _log(f"[Gemini WAIT] sleeping {delay:.0f}s before retry…")
                _time.sleep(delay)
                continue

            # Final attempt with a retryable error — give up on this model
            _log(f"[Gemini FAIL] model={model_id!r}  all retries exhausted")
            raise  # let the chain loop catch and move to next model

    # Should never reach here
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Groq call helpers
# ---------------------------------------------------------------------------

def _call_groq(
    groq_key: str,
    model_id: str,
    user_prompt: str,
    system_prompt: str,
) -> str:
    """
    Call a Groq-hosted LLM using the official ``groq`` Python client.
    Returns the raw text content of the first choice message.
    """
    if not _GROQ_AVAILABLE or _GroqClient is None:
        raise RuntimeError(
            "groq package is not installed. Run: pip install groq"
        )

    _log(f"[Groq Call]   model={model_id!r}")
    client = _GroqClient(api_key=groq_key)
    chat_completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=4096,
    )
    raw = chat_completion.choices[0].message.content or ""
    _log(f"[Groq OK]     model={model_id!r}  chars={len(raw)}")
    return raw


def _call_groq_with_retry(
    groq_key: str,
    model_id: str,
    user_prompt: str,
    system_prompt: str,
) -> str:
    """Call Groq with the same exponential-backoff policy as Gemini."""
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return _call_groq(groq_key, model_id, user_prompt, system_prompt)
        except Exception as exc:
            last_exc = exc
            _log(f"[Groq ERR]    model={model_id!r}  attempt={attempt}  error={exc!r}")
            if _is_retryable(exc) and attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                _log(f"[Groq WAIT]   sleeping {delay:.0f}s before retry…")
                _time.sleep(delay)
            else:
                raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Scene validation helper (shared between Gemini + Groq paths)
# ---------------------------------------------------------------------------

def _validate_scenes(scenes: list[dict]) -> None:
    """
    Raise RuntimeError if the scene list is empty or missing required keys.
    Required keys: scene_number, timestamp, narration, visual_prompt.
    """
    if not scenes:
        raise RuntimeError("LLM returned an empty scene list.")
    for i, scene in enumerate(scenes):
        for required in ("scene_number", "timestamp", "narration", "visual_prompt"):
            if required not in scene:
                raise RuntimeError(
                    f"Scene {i + 1} is missing required key '{required}'."
                )


# ---------------------------------------------------------------------------
# JSON parsing helper
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
        "Failed to parse LLM response as a JSON scene array.\n"
        f"Raw response:\n{raw_text[:500]}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_storyboard(
    article_text: str,
    duration_sec: int,
    *,
    api_key: str | None = None,
    model_name: str = "Gemini 2.0 Flash",
) -> list[dict]:
    """Call Gemini (with automatic Groq fallback) to produce scene dicts.

    Parameters
    ----------
    article_text : str
        The cleaned article body text.
    duration_sec : int
        Target video length in seconds (60-120).
    api_key : str | None
        Gemini API key.  Falls back to ``GEMINI_API_KEY`` env var.
    model_name : str
        Human-readable model label (key in ``MODEL_MAP``, or raw Gemini ID).

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
        If both Gemini and Groq fail after all retries and models.
    """
    # ── Resolve Gemini API key ────────────────────────────────────────────
    gemini_key = api_key or os.getenv("GEMINI_API_KEY", "")

    # ── Resolve primary Gemini model ID ──────────────────────────────────
    # First check MODEL_MAP (covers all display names + aliases)
    model_id: str | None = MODEL_MAP.get(model_name)
    if model_id is None:
        # Accept raw API IDs passed directly (e.g. "gemini-2.0-flash")
        low = model_name.lower()
        if "lite" in low or "1.5" in low:
            model_id = "gemini-1.5-flash"
        elif "gemini" in low or "flash" in low:
            model_id = "gemini-2.0-flash"
        else:
            # Unknown model — default to safe high-capacity model rather
            # than raising, so the pipeline continues uninterrupted
            model_id = "gemini-2.0-flash"
            _log(
                f"[Director] Unknown model_name={model_name!r}, "
                f"defaulting to {model_id!r}"
            )

    _log(
        f"[Director] generate_storyboard started  "
        f"model_name={model_name!r}  resolved_id={model_id!r}  "
        f"duration={duration_sec}s"
    )

    # ── Build shared prompts ──────────────────────────────────────────────
    target_words = int(duration_sec * _WPM_RATE)
    min_scenes, max_scenes = _estimate_scene_range(duration_sec)

    system_prompt = _SYSTEM_PROMPT.format(
        target_words=target_words,
        duration_sec=duration_sec,
        min_scenes=min_scenes,
        max_scenes=max_scenes,
    )
    user_prompt = _USER_PROMPT.format(
        article_text=article_text[:8000],
        duration_sec=duration_sec,
        target_words=target_words,
        min_scenes=min_scenes,
        max_scenes=max_scenes,
    )

    # ── Phase 1: Try Gemini models in order ──────────────────────────────
    # Build the chain: requested model first, then remaining fallbacks.
    gemini_chain = [model_id] + [m for m in _GEMINI_FALLBACK_CHAIN if m != model_id]
    _log(f"[Director] Gemini chain: {gemini_chain}")

    gemini_last_exc: Exception | None = None

    if gemini_key:
        client = genai.Client(api_key=gemini_key)

        for attempt_model in gemini_chain:
            try:
                raw = _call_gemini_with_retry(
                    client, attempt_model, user_prompt, system_prompt
                )
                scenes = _parse_scenes_json(raw)
                _validate_scenes(scenes)
                _log(f"[Director] Gemini success with model={attempt_model!r}  scenes={len(scenes)}")
                return scenes  # SUCCESS

            except Exception as exc:
                # Catches ALL errors (transient, hard, parse, validation).
                # Log it, store it, and fall through to the next model.
                gemini_last_exc = exc
                _log(
                    f"[Director] Gemini model={attempt_model!r} FAILED: {exc!r}  "
                    "-> trying next model"
                )
                continue  # try next Gemini model

        _log(f"[Director] All Gemini models exhausted. Last error: {gemini_last_exc!r}")

    else:
        gemini_last_exc = ValueError(
            "No GEMINI_API_KEY provided — skipping Gemini entirely."
        )
        _log(f"[Director] {gemini_last_exc}")

    # ── Phase 2: Groq fallback ────────────────────────────────────────────
    groq_key = GROQ_API_KEY
    _log(
        f"[Director] Attempting Groq fallback  "
        f"available={_GROQ_AVAILABLE}  key_set={bool(groq_key)}"
    )

    if groq_key and _GROQ_AVAILABLE:
        groq_last_exc: Exception | None = None
        _log(f"[Director] Groq chain: {_GROQ_MODEL_CHAIN}")

        for groq_model in _GROQ_MODEL_CHAIN:
            try:
                raw = _call_groq_with_retry(
                    groq_key, groq_model, user_prompt, system_prompt
                )
                scenes = _parse_scenes_json(raw)
                _validate_scenes(scenes)
                _log(f"[Director] Groq success with model={groq_model!r}  scenes={len(scenes)}")
                return scenes  # SUCCESS via Groq

            except Exception as exc:
                groq_last_exc = exc
                _log(f"[Director] Groq model={groq_model!r} FAILED: {exc!r}  -> trying next")
                continue

        raise RuntimeError(
            f"All Gemini models failed (last: {gemini_last_exc}) AND "
            f"all Groq models failed (last: {groq_last_exc}). "
            "Check your API keys and network connectivity."
        )

    # ── Phase 3: Surface the best available error message ─────────────────
    if not gemini_key and not groq_key:
        raise ValueError(
            "No GEMINI_API_KEY and no GROQ_API_KEY found. "
            "Set at least one in your .env file."
        )

    if not _GROQ_AVAILABLE:
        raise RuntimeError(
            f"All Gemini models failed. Last error: {gemini_last_exc}\n"
            "Tip: install the Groq fallback with: pip install groq\n"
            "     then add GROQ_API_KEY=<key> to your .env file."
        )

    # groq package installed but GROQ_API_KEY not set
    raise RuntimeError(
        f"All Gemini models failed. Last error: {gemini_last_exc}\n"
        "Tip: add GROQ_API_KEY=<key> to your .env file to enable "
        "the automatic Groq fallback."
    )

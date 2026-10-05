"""
Director LLM Engine — AI NEWS Video Generator
==============================================
Generates a structured news storyboard (JSON array of scenes) by calling the
Google Gemini API with the ingested article text and target duration.

Falls back automatically to Groq (qwen3.8-27b -> openai/gpt-oss-20b)
if Gemini is unavailable (503 UNAVAILABLE, rate-limit, connection error, etc.).

Public API:
    generate_storyboard(article_text, duration_sec, *, api_key, model_name)
"""

import json
import re
import os
import sys
import time as _time

from google import genai
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Groq client (optional)
# ---------------------------------------------------------------------------

try:
    from groq import Groq as _GroqClient
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    _GroqClient = None

load_dotenv()

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

# ---------------------------------------------------------------------------
# WPM & scene-count math
# ---------------------------------------------------------------------------

_WPM_RATE = 2.33  # ~140 words per minute -> 2.33 words per second

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

    return 3, 5


# ---------------------------------------------------------------------------
# Model-ID mapping
# ---------------------------------------------------------------------------

MODEL_MAP: dict[str, str] = {
    "Gemini 3.6 Flash":          "gemini-3.6-flash",
    "Gemini 3.5 Flash Lite":     "gemini-3.5-flash-lite",
    "Gemini 2.0 Flash":          "gemini-3.6-flash",
    "Gemini 2.5 Flash":          "gemini-3.6-flash",
    "Gemini 1.5 Flash":          "gemini-3.5-flash-lite",
    "Gemini 1.5 Flash Lite":     "gemini-3.5-flash-lite",
}

_GEMINI_FALLBACK_CHAIN: list[str] = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

_GROQ_MODEL_CHAIN: list[str] = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]

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

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 2.0
_CONNECT_RETRY_DELAY = 1.0


def _log(msg: str) -> None:
    print(f"[Director] {msg}", file=sys.stderr, flush=True)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    cls = type(exc).__name__.lower()
    combined = f"{cls} {msg}"

    HARD_ERRORS = (
        "404", "not found", "model_not_found",
        "400", "bad request", "invalid_argument",
        "401", "403", "permission", "unauthorized",
    )
    if any(t in combined for t in HARD_ERRORS):
        return False

    RETRYABLE = (
        "503", "unavailable", "rate limit", "rate_limit",
        "resource exhausted", "resource_exhausted",
        "quota", "overloaded", "server error", "try again",
        "connect", "network", "timeout", "timed out",
        "reset by peer", "getaddrinfo", "errno 11001", "errno 11004",
        "name or service", "temporary failure", "eof", "broken pipe", "ssl",
    )
    return any(t in combined for t in RETRYABLE)


def _is_skip_worthy(exc: Exception) -> bool:
    msg = str(exc).lower()
    cls = type(exc).__name__.lower()
    combined = f"{cls} {msg}"
    return any(t in combined for t in (
        "404", "not found", "model_not_found",
        "400", "bad request", "invalid_argument",
        "401", "403", "permission", "unauthorized",
    ))


def _call_gemini_with_retry(
    client: "genai.Client",
    model_id: str,
    user_prompt: str,
    system_prompt: str,
) -> str:
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
                _log(f"[Gemini SKIP] model={model_id!r}  hard error, skipping model")
                raise

            if _is_retryable(exc) and attempt < _MAX_RETRIES:
                _is_network_err = any(
                    t in f"{type(exc).__name__.lower()} {str(exc).lower()}"
                    for t in ("connect", "getaddrinfo", "errno 11", "network",
                              "timeout", "eof", "broken pipe", "ssl")
                )
                delay = (
                    _CONNECT_RETRY_DELAY * (1.5 ** (attempt - 1))
                    if _is_network_err
                    else _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                )
                _log(f"[Gemini WAIT] sleeping {delay:.1f}s before retry…")
                _time.sleep(delay)
                continue

            _log(f"[Gemini FAIL] model={model_id!r}  all retries exhausted")
            raise

    raise last_exc


def _call_groq(
    groq_key: str,
    model_id: str,
    user_prompt: str,
    system_prompt: str,
) -> str:
    if not _GROQ_AVAILABLE or _GroqClient is None:
        raise RuntimeError("groq package is not installed. Run: pip install groq")

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
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return _call_groq(groq_key, model_id, user_prompt, system_prompt)
        except Exception as exc:
            last_exc = exc
            _log(f"[Groq ERR]    model={model_id!r}  attempt={attempt}  error={exc!r}")
            if _is_retryable(exc) and attempt < _MAX_RETRIES:
                _is_network_err = any(
                    t in f"{type(exc).__name__.lower()} {str(exc).lower()}"
                    for t in ("connect", "getaddrinfo", "errno 11", "network",
                              "timeout", "eof", "broken pipe", "ssl")
                )
                delay = (
                    _CONNECT_RETRY_DELAY * (1.5 ** (attempt - 1))
                    if _is_network_err
                    else _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                )
                _log(f"[Groq WAIT]   sleeping {delay:.1f}s before retry…")
                _time.sleep(delay)
            else:
                raise
    raise last_exc


def _validate_scenes(scenes: list[dict]) -> None:
    if not scenes:
        raise RuntimeError("LLM returned an empty scene list.")
    for i, scene in enumerate(scenes):
        for required in ("scene_number", "timestamp", "narration", "visual_prompt"):
            if required not in scene:
                raise RuntimeError(
                    f"Scene {i + 1} is missing required key '{required}'."
                )


def _parse_scenes_json(raw_text: str) -> list[dict]:
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

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


def generate_storyboard(
    article_text: str,
    duration_sec: int,
    *,
    api_key: str | None = None,
    model_name: str = "Gemini 3.6 Flash",
) -> list[dict]:
    """Call Gemini (with automatic Groq fallback) to produce scene dicts."""
    gemini_key = api_key or os.getenv("GEMINI_API_KEY", "")

    # Resolve model ID
    model_id: str | None = MODEL_MAP.get(model_name)
    if model_id is None:
        if model_name.startswith("gemini-"):
            model_id = model_name
        else:
            raise ValueError(f"Unsupported model: '{model_name}'.")

    _log(
        f"[Director] generate_storyboard started  "
        f"model_name={model_name!r}  resolved_id={model_id!r}  "
        f"duration={duration_sec}s"
    )

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
                return scenes

            except Exception as exc:
                gemini_last_exc = exc
                _log(f"[Director] Gemini model={attempt_model!r} FAILED: {exc!r} -> trying next")
                continue

        _log(f"[Director] All Gemini models exhausted. Last error: {gemini_last_exc!r}")
    else:
        gemini_last_exc = ValueError("No Gemini API key provided — skipping Gemini.")
        _log(f"[Director] {gemini_last_exc}")

    # Groq Fallback
    groq_key = GROQ_API_KEY
    if groq_key and _GROQ_AVAILABLE:
        groq_last_exc: Exception | None = None
        for groq_model in _GROQ_MODEL_CHAIN:
            try:
                raw = _call_groq_with_retry(
                    groq_key, groq_model, user_prompt, system_prompt
                )
                scenes = _parse_scenes_json(raw)
                _validate_scenes(scenes)
                _log(f"[Director] Groq success with model={groq_model!r}  scenes={len(scenes)}")
                return scenes
            except Exception as exc:
                groq_last_exc = exc
                _log(f"[Director] Groq model={groq_model!r} FAILED: {exc!r} -> trying next")
                continue

        raise RuntimeError(
            f"All Gemini models failed (last: {gemini_last_exc}) AND "
            f"all Groq models failed (last: {groq_last_exc})."
        )

    if not gemini_key and not groq_key:
        raise ValueError("No Gemini API key provided and no GROQ_API_KEY found.")

    if not _GROQ_AVAILABLE:
        raise RuntimeError(f"All Gemini models failed: {gemini_last_exc}")

    raise RuntimeError(f"All Gemini models failed. Last error: {gemini_last_exc}")

"""
Full diagnostic smoke-test for the updated director.py + app.py model alignment.
No API calls are made.
"""
import sys, json

# ── 1. Imports ────────────────────────────────────────────────────────────────
from director import (
    MODEL_MAP,
    _GROQ_AVAILABLE,
    _GROQ_MODEL_CHAIN,
    _GEMINI_FALLBACK_CHAIN,
    _parse_scenes_json,
    _validate_scenes,
    _is_retryable,
    _is_skip_worthy,
    _estimate_scene_range,
)
print("1. Imports OK")

# ── 2. Groq availability ──────────────────────────────────────────────────────
assert _GROQ_AVAILABLE, "groq package should be installed"
print(f"2. Groq available: {_GROQ_AVAILABLE}  chain: {_GROQ_MODEL_CHAIN}")

# ── 3. MODEL_MAP only uses real Gemini API IDs ────────────────────────────────
VALID_GEMINI_IDS = {"gemini-2.0-flash", "gemini-1.5-flash"}
for display, api_id in MODEL_MAP.items():
    assert api_id in VALID_GEMINI_IDS, (
        f"MODEL_MAP[{display!r}] = {api_id!r} is NOT a real Gemini API ID!"
    )
print(f"3. MODEL_MAP verified: {dict(MODEL_MAP)}")

# ── 4. Gemini fallback chain uses real IDs ────────────────────────────────────
for m in _GEMINI_FALLBACK_CHAIN:
    assert m in VALID_GEMINI_IDS, f"_GEMINI_FALLBACK_CHAIN contains invalid ID {m!r}"
print(f"4. Gemini fallback chain OK: {_GEMINI_FALLBACK_CHAIN}")

# ── 5. Default model mapping ("Gemini 2.0 Flash" -> "gemini-2.0-flash") ───────
assert MODEL_MAP.get("Gemini 2.0 Flash") == "gemini-2.0-flash"
assert MODEL_MAP.get("Gemini 1.5 Flash") == "gemini-1.5-flash"
# Phantom aliases still resolve to real IDs
assert MODEL_MAP.get("Gemini 2.5 Flash") == "gemini-2.0-flash"
assert MODEL_MAP.get("Gemini 3.6 Flash") == "gemini-2.0-flash"
print("5. Model alias resolution OK")

# ── 6. _is_retryable correctly classifies errors ─────────────────────────────
class _Exc(Exception): pass

assert _is_retryable(_Exc("503 Service Unavailable"))        # retryable
assert _is_retryable(_Exc("rate limit exceeded"))            # retryable
assert _is_retryable(_Exc("resource_exhausted"))             # retryable
assert _is_retryable(_Exc("connection reset by peer"))       # retryable
assert not _is_retryable(_Exc("404 model_not_found"))        # hard error
assert not _is_retryable(_Exc("400 bad request"))            # hard error
assert not _is_retryable(_Exc("401 unauthorized"))           # hard error
print("6. _is_retryable classification OK")

# ── 7. _is_skip_worthy classifies hard errors for immediate skip ──────────────
assert _is_skip_worthy(_Exc("404 NOT_FOUND"))
assert _is_skip_worthy(_Exc("400 INVALID_ARGUMENT"))
assert not _is_skip_worthy(_Exc("503 UNAVAILABLE"))
assert not _is_skip_worthy(_Exc("rate limit"))
print("7. _is_skip_worthy classification OK")

# ── 8. JSON parser + validator ────────────────────────────────────────────────
sample = json.dumps([{
    "scene_number": 1,
    "timestamp": "00:00 - 00:15",
    "narration": "Breaking news tonight.",
    "visual_prompt": "Wide cinematic shot of city skyline.",
}])
scenes = _parse_scenes_json(sample)
_validate_scenes(scenes)
print(f"8. Parser + validator OK ({len(scenes)} scene)")

# Fenced markdown variant (common LLM output pattern)
fenced = "```json\n" + sample + "\n```"
_validate_scenes(_parse_scenes_json(fenced))
print("8b. Fenced-JSON parser OK")

# ── 9. Scene range math ───────────────────────────────────────────────────────
assert _estimate_scene_range(60)  == (3, 4)
assert _estimate_scene_range(120) == (7, 9)
assert _estimate_scene_range(90)  == (4, 6)
print("9. Scene range math OK")

# ── 10. app.py session state default model ────────────────────────────────────
with open("app.py", encoding="utf-8") as f:
    src = f.read()
assert "Gemini 2.0 Flash" in src,  "app.py session state default should be 'Gemini 2.0 Flash'"
assert "Gemini 2.5 Flash" not in src, "Phantom 'Gemini 2.5 Flash' should be removed from app.py"
assert "Gemini 3.6 Flash" not in src, "Phantom 'Gemini 3.6 Flash' should be removed from app.py"
assert "Gemini 3.5 Flash Lite" not in src, "Phantom 'Gemini 3.5 Flash Lite' should be removed from app.py"
print("10. app.py phantom model names removed OK")

print("\nAll diagnostic checks passed.")

# TASK SPECIFICATION: Phase 4 — Audio Narration & TTS Engine

## 1. Objective
Implement a text-to-speech audio module (`audio.py`) using `edge-tts` to generate neural news-anchor voiceovers for each storyboard scene and integrate audio preview playback directly into the Streamlit UI.

---

## 2. Dependencies (`requirements.txt`)
Add the following package to `requirements.txt`:
- `edge-tts`

---

## 3. Audio Engine Module (`audio.py`)

Create `audio.py` with the following features:

### Supported News Voices:
- `en-US-AvaNeural` (Female - Professional News Anchor)
- `en-US-AndrewNeural` (Male - Authoritative News Anchor)
- `en-US-GuyNeural` (Male - Standard Broadcast)
- `en-GB-SoniaNeural` (Female - British News)

### Key Function:
```python
async def generate_scene_audio_async(text: str, voice: str = "en-US-AvaNeural", rate: str = "+0%") -> bytes
def generate_scene_audio(text: str, voice: str = "en-US-AvaNeural", rate: str = "+0%") -> bytes
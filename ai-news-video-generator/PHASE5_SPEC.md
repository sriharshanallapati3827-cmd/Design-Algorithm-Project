# TASK SPECIFICATION: Phase 5 — Video Assembly & Rendering Engine

## 1. Objective
Implement a video synthesis module (`video_engine.py`) using `moviepy` to assemble generated scene images, audio voiceovers, and text caption overlays into a final rendered news video file (`output.mp4`), with interactive preview and download capabilities in Streamlit.

---

## 2. Dependencies (`requirements.txt`)
Add the following packages to `requirements.txt`:
- `moviepy`
- `imageio-ffmpeg`
- `pillow`

---

## 3. Video Engine Module (`video_engine.py`)

Create `video_engine.py` with the following core functionality:

### Key Functions:
```python
def create_scene_clip(image_bytes: bytes, audio_bytes: bytes, subtitle_text: str, target_size: tuple = (1024, 576)) -> VideoClip
def assemble_full_video(scenes: list, output_path: str = "output_video.mp4") -> str
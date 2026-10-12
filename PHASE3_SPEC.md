# TASK SPECIFICATION: Phase 3 — Local GPU Image Engine

## 1. Objective
Implement a local PyTorch & Diffusers image generation engine (`generator.py`) optimized for an NVIDIA RTX 4050 GPU (6GB VRAM), replacing placeholder visual cards with AI-rendered landscape images based on LLM visual prompts.

---

## 2. Dependencies (`requirements.txt`)
Add the following libraries to `requirements.txt`:
- `torch` (with CUDA support)
- `torchvision`
- `diffusers`
- `transformers`
- `accelerate`
- `safetensors`

---

## 3. Image Engine Module (`generator.py`)

Create `generator.py` with the following functionality:

### Model & Pipeline Selection:
- Primary Model: `ByteDance/SDXL-Lightning` (4-step SDXL) or `stabilityai/sdxl-turbo` / `runwayml/stable-diffusion-v1-5`.
- Target Resolution: `1024x576` (16:9 widescreen landscape) or `512x512`.

### GPU & VRAM Optimizations for RTX 4050 (6GB VRAM):
- Enforce `torch.float16` precision.
- Check CUDA availability with `torch.cuda.is_available()`.
- Enable memory optimizations:
  - `pipe.enable_model_cpu_offload()` (or `pipe.enable_sequential_cpu_offload()` if VRAM is tight)
  - `pipe.enable_attention_slicing()`
- Enable `torch.backends.cudnn.benchmark = True`.

### Key Function:
```python
def generate_scene_image(prompt: str, height: int = 576, width: int = 1024, num_inference_steps: int = 4) -> Image.Image
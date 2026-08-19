"""
Utility helpers for the AI NEWS Generator frontend.

Provides scene estimation, demo data generation, timestamp formatting,
and placeholder image creation for the Streamlit UI.
"""

import io
import math
from typing import List, Dict, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None  # Graceful fallback; placeholder images will be skipped


# ---------------------------------------------------------------------------
# Scene / Duration calculations
# ---------------------------------------------------------------------------

def calculate_scene_range(duration_sec: int) -> Tuple[int, int]:
    """Return (min_scenes, max_scenes) for a given target duration.

    Heuristic:
        - Each scene is roughly 13–20 seconds long.
        - min_scenes = duration // 20   (longest scenes)
        - max_scenes = duration // 13   (shortest scenes)
    Clamps the minimum to 1 scene.
    """
    min_scenes = max(1, duration_sec // 20)
    max_scenes = max(min_scenes, duration_sec // 13)
    return min_scenes, max_scenes


def format_timestamp(start_sec: float, end_sec: float) -> str:
    """Format a time range as ``MM:SS – MM:SS``."""

    def _fmt(s: float) -> str:
        m = int(s) // 60
        sec = int(s) % 60
        return f"{m:02d}:{sec:02d}"

    return f"{_fmt(start_sec)} – {_fmt(end_sec)}"


# ---------------------------------------------------------------------------
# Demo / placeholder data
# ---------------------------------------------------------------------------

_DEMO_ARTICLE = (
    "WASHINGTON — In a landmark announcement today, NASA confirmed that its "
    "Artemis IV mission has successfully entered lunar orbit, marking the "
    "first crewed visit to the Moon's south pole region. The crew of four "
    "astronauts — Commander Elena Vasquez, Pilot James Okonkwo, and Mission "
    "Specialists Mei-Lin Chen and David Osei — reported nominal systems as "
    "they prepared for a 7-day surface expedition.\n\n"
    "\"This is a defining moment not just for the United States, but for all "
    "of humanity,\" said NASA Administrator Dr. Priya Nair during a live "
    "press briefing from Johnson Space Center. \"The resources locked in the "
    "lunar south pole could power future deep-space missions for decades.\"\n\n"
    "The mission's primary objective is to drill core samples from permanently "
    "shadowed craters believed to contain significant water-ice deposits. If "
    "confirmed in sufficient quantities, the ice could be converted into "
    "hydrogen fuel, dramatically reducing the cost of future Mars missions.\n\n"
    "International partners, including ESA, JAXA, and ISRO, have contributed "
    "key instruments aboard the lander. ESA's PROSPECT drill and JAXA's "
    "micro-rover are expected to begin autonomous operations within 48 hours "
    "of touchdown.\n\n"
    "Markets reacted positively to the news, with aerospace stocks surging "
    "3.2% in midday trading. Analysts say the mission could unlock a new "
    "commercial space economy worth an estimated $1.8 trillion by 2045."
)


def load_demo_article() -> str:
    """Return a realistic sample news article for demonstration purposes."""
    return _DEMO_ARTICLE


def generate_demo_scenes(num_scenes: int, duration_sec: int) -> List[Dict]:
    """Create a list of stub scene dictionaries with placeholder content.

    Each dict contains:
        - scene_number (int)
        - timestamp (str)           e.g. "00:00 – 00:15"
        - voiceover (str)           narration text
        - visual_prompt (str)       image-gen prompt
        - image_bytes (bytes|None)  placeholder PNG
    """
    scene_duration = duration_sec / num_scenes
    scenes: List[Dict] = []

    sample_voiceovers = [
        "NASA's Artemis IV has entered lunar orbit, marking a historic milestone in space exploration.",
        "The crew of four astronauts prepares for a seven-day surface expedition near the Moon's south pole.",
        "Commander Elena Vasquez confirms all systems nominal as the lander begins its descent sequence.",
        "Scientists aim to extract water-ice samples from permanently shadowed craters on the lunar surface.",
        "International partners ESA, JAXA, and ISRO have contributed critical instruments to the mission.",
        "NASA Administrator Dr. Priya Nair calls this a defining moment for humanity's future in space.",
        "Aerospace markets surge as analysts project a trillion-dollar commercial space economy by 2045.",
        "The PROSPECT drill and JAXA micro-rover will begin autonomous surface operations within 48 hours.",
        "If water-ice is confirmed, it could be converted to hydrogen fuel for future Mars-bound missions.",
    ]

    sample_prompts = [
        "Cinematic wide shot of a spacecraft orbiting the Moon, Earth visible in the background, photorealistic, 1080p landscape",
        "Four astronauts inside a modern spacecraft cockpit checking instruments, dramatic lighting, photorealistic",
        "Lunar lander descending toward the Moon's south pole, dust particles floating, cinematic lighting",
        "Close-up of a robotic drill extracting ice core samples from a dark lunar crater, sci-fi realism",
        "International space agency logos displayed on equipment inside a lunar lander, documentary style",
        "NASA press briefing room with a large screen showing the Moon, reporters in foreground, photorealistic",
        "Stock market trading floor with green upward arrows on screens, futuristic UI overlays, cinematic",
        "Small autonomous rover on the grey lunar surface near a crater's edge, long shadows, photorealistic",
        "Hydrogen fuel cell diagram overlaid on a Mars landscape background, educational infographic style",
    ]

    for i in range(num_scenes):
        start = i * scene_duration
        end = start + scene_duration
        scenes.append(
            {
                "scene_number": i + 1,
                "timestamp": format_timestamp(start, end),
                "voiceover": sample_voiceovers[i % len(sample_voiceovers)],
                "visual_prompt": sample_prompts[i % len(sample_prompts)],
                "image_bytes": _create_placeholder_image(i + 1),
            }
        )

    return scenes


# ---------------------------------------------------------------------------
# Placeholder image generation
# ---------------------------------------------------------------------------

def _create_placeholder_image(scene_number: int) -> bytes | None:
    """Generate a simple 768×432 dark placeholder PNG with the scene number."""
    if Image is None:
        return None

    width, height = 768, 432
    bg_color = (18, 24, 36)        # #121824
    accent_color = (245, 166, 35)  # #F5A623

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw a border
    draw.rectangle(
        [2, 2, width - 3, height - 3],
        outline=accent_color,
        width=2,
    )

    # Scene label — use default font (always available)
    label = f"SCENE {scene_number}"
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        ((width - text_w) / 2, (height - text_h) / 2),
        label,
        fill=accent_color,
        font=font,
    )

    # Filmstrip-style decoration (top & bottom bars)
    for y in (0, height - 8):
        draw.rectangle([0, y, width, y + 8], fill=(30, 36, 50))
        for x in range(0, width, 40):
            draw.rectangle([x + 4, y + 1, x + 20, y + 7], fill=(50, 58, 78))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

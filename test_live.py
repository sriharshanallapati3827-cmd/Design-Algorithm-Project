"""Live end-to-end test of generate_storyboard with the confirmed model IDs."""
import sys, os

# Wipe any stale cached import
for key in list(sys.modules.keys()):
    if 'director' in key:
        del sys.modules[key]

from dotenv import load_dotenv
load_dotenv()

from director import generate_storyboard

print("Calling generate_storyboard (live API)...")
scenes = generate_storyboard(
    article_text="India wins the cricket World Cup 2026 in a thrilling final against Australia in Mumbai.",
    duration_sec=60,
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name="Gemini 3.6 Flash",
)
print(f"\nSUCCESS: {len(scenes)} scenes generated")
for s in scenes:
    snum = s["scene_number"]
    narr = s["narration"][:70]
    print(f"  Scene {snum}: {narr}...")

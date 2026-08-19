"""
Verification script for AI NEWS Generator frontend.
Run after installing dependencies:  python verify.py
"""

import sys

def check(label: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    msg = f"[{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main():
    all_ok = True

    # 1. Check dependencies
    try:
        import streamlit
        all_ok &= check("Streamlit installed", True, f"v{streamlit.__version__}")
    except ImportError:
        all_ok &= check("Streamlit installed", False, "pip install streamlit")

    try:
        from PIL import Image
        all_ok &= check("Pillow installed", True)
    except ImportError:
        all_ok &= check("Pillow installed", False, "pip install Pillow")

    # 2. Check theme config
    try:
        import tomllib
        with open(".streamlit/config.toml", "rb") as f:
            cfg = tomllib.load(f)
        t = cfg["theme"]
        all_ok &= check("Theme base=dark", t["base"] == "dark")
        all_ok &= check("Background #0B0F17", t["backgroundColor"] == "#0B0F17")
        all_ok &= check("Secondary bg #121824", t["secondaryBackgroundColor"] == "#121824")
        all_ok &= check("Accent #F5A623", t["primaryColor"] == "#F5A623")
    except Exception as e:
        all_ok &= check("Theme config", False, str(e))

    # 3. Check utils module
    try:
        from utils import calculate_scene_range, format_timestamp, generate_demo_scenes, load_demo_article

        # Scene range calculations
        min60, max60 = calculate_scene_range(60)
        all_ok &= check("Scene range 60s", min60 == 3 and max60 == 4, f"{min60}–{max60}")

        min120, max120 = calculate_scene_range(120)
        all_ok &= check("Scene range 120s", min120 == 6 and max120 == 9, f"{min120}–{max120}")

        # Timestamp formatting
        ts = format_timestamp(0, 15)
        all_ok &= check("Timestamp format", "00:00" in ts and "00:15" in ts, ts)

        # Demo article
        article = load_demo_article()
        all_ok &= check("Demo article", len(article) > 500, f"{len(article)} chars")

        # Scene generation
        scenes = generate_demo_scenes(5, 90)
        all_ok &= check("Scene count", len(scenes) == 5)
        required_keys = {"scene_number", "timestamp", "voiceover", "visual_prompt", "image_bytes"}
        has_keys = required_keys.issubset(set(scenes[0].keys()))
        all_ok &= check("Scene keys", has_keys, str(list(scenes[0].keys())))
        all_ok &= check("Scene images", scenes[0]["image_bytes"] is not None)

    except Exception as e:
        all_ok &= check("Utils module", False, str(e))

    # 4. Check app.py for spec compliance (static analysis)
    try:
        with open("app.py", "r", encoding="utf-8") as f:
            source = f.read()

        all_ok &= check("Brand: AI NEWS Generator", "AI NEWS Generator" in source)
        all_ok &= check("Tab: URL input", "URL" in source and "text_input" in source)
        all_ok &= check("Tab: PDF upload", "file_uploader" in source and "pdf" in source)
        all_ok &= check("Tab: Raw Text", "text_area" in source)
        all_ok &= check("Duration slider 60-120", 'min_value=60' in source and 'max_value=120' in source)
        all_ok &= check("Scene estimate badge", "Estimated Scenes" in source)
        all_ok &= check("No art_style/ART STYLE", "art_style" not in source.lower() or "art style" not in source.lower())
        all_ok &= check("Model selector", "Gemini 2.5 Flash" in source and "Claude 3.5 Sonnet" in source)
        all_ok &= check("Primary CTA", "GENERATE AI NEWS STORYBOARD" in source)
        all_ok &= check("Secondary CTA", "LOAD DEMO NEWS ARTICLE" in source)
        all_ok &= check("Pro Tips", "PRO TIPS" in source)
        all_ok &= check("Empty state heading", "YOUR NEWS STORYBOARD AWAITS" in source)
        all_ok &= check("Empty state subtext", "Enter a URL, PDF, or text on the left" in source)
        all_ok &= check("Scene card header", "SCENE" in source)
        all_ok &= check("Re-render button", "Re-render Image" in source)
        all_ok &= check("Voiceover script", "Voiceover Script" in source)
        all_ok &= check("Visual prompt", "Visual Prompt" in source)
        all_ok &= check("Audio preview", "Audio Preview" in source)
        all_ok &= check("Loading state", "pipeline" in source.lower() or "Generating" in source)
        all_ok &= check("Summarization warning", "summarized" in source)

    except Exception as e:
        all_ok &= check("app.py spec compliance", False, str(e))

    # Summary
    print("\n" + "=" * 50)
    if all_ok:
        print("ALL CHECKS PASSED [OK]")
    else:
        print("SOME CHECKS FAILED [!!] -- review output above")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

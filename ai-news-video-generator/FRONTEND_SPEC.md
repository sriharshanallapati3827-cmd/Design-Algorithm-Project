# TASK SPECIFICATION: AI NEWS Generator - Frontend UI & Architecture

## 1. Executive Summary & Naming
- **Project Name:** `AI NEWS Generator`
- **Goal:** Redesign and implement the frontend interface based on the reference layout while shifting control from manual frame count to video duration (1–2 minutes).

---

## 2. System Architecture & Data Flow
The frontend must integrate seamlessly with the following backend pipeline:
---

## 3. Left Sidebar Configuration (Syncing with Reference Layout)

### A. Brand Header
- Change top logo title to **`AI NEWS Generator`**.

### B. Multi-Source Ingestion (Replacing Raw Text Only)
Provide a tabbed or toggle-based input container with three options:
1. **URL Input:** Input field for news article links (scraped via backend scraper).
2. **PDF Upload:** File upload drag-and-drop zone (`.pdf` format). Displays `Accepted: .pdf up to 5MB (~1–5 pages)`.
3. **Raw Text:** Textarea for pasting manual text directly.

*Constraint Logic:* Include a warning note that long articles will automatically be summarized into a concise news script tailored for a 1-to-2 minute target video.

### C. Duration Control (Replaces "Number of Frames")
- **Control Type:** Slider or Selector for **Target Duration**.
- **Range:** `60 seconds (1 min)` to `120 seconds (2 mins)`.
- **Dynamic Frame Calculation Display:** Below the slider, show a dynamic text badge indicating calculated frame limits:
  - *Example (60s):* `Estimated Scenes: 3 – 5 Frames`
  - *Example (120s):* `Estimated Scenes: 6 – 9 Frames`

### D. Removed Controls
- ❌ **ART STYLE Dropdown:** Remove this component entirely from the sidebar.

### E. Director Model Selector
- Keep the model selection dropdown (e.g., `Gemini 2.5 Flash / Gemini 1.5 Flash` or `Claude 3.5 Sonnet`).

### F. Action Buttons & Pro Tips
- **Primary CTA:** `[ GENERATE AI NEWS STORYBOARD ]` (Triggers backend pipeline).
- **Secondary CTA:** `[ LOAD DEMO NEWS ARTICLE ]`.
- **Pro Tips Box:** Retain the styled dark container at the bottom with tips tailored for news generation (e.g., *"Focus on key headlines"*, *"Use 1080p landscape prompts"*).

---

## 4. Main Workspace (Timeline Display)

### A. Initial Empty State
- Display the centered filmstrip icon and text matching the reference design:
  - **Heading:** `YOUR NEWS STORYBOARD AWAITS`
  - **Subtext:** *"Enter a URL, PDF, or text on the left, pick your duration, and watch your video scenes generate in real-time."*

### B. Generated Scene Grid (Card Component)
When the pipeline finishes, replace the empty state with a horizontal/grid timeline. Each **Scene Card** must display:
1. **Header:** `SCENE [X]` | `Timestamp (e.g., 00:00 - 00:15)`
2. **Visual Frame:** Container rendering the locally generated AI image (`.png`). Include a `[ Re-render Image ]` button.
3. **Voiceover Script:** Editable text block showing the narration for that scene.
4. **Visual Prompt:** Code/text box showing the image prompt sent to the local model.
5. **Audio Preview:** Embedded audio player preview for the scene's TTS output.

---

## 5. Technical Requirements for Implementation
- **Theme:** Dark mode (`#0B0F17` / `#121824` background palette) with gold/yellow accent highlights for controls, mirroring the provided screenshot.
- **State Management:** Enable loading states (progress indicator/step list) while backend scraping, script generation, and local image rendering take place.
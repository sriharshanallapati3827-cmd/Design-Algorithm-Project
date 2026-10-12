# TASK SPECIFICATION: Phase 2 — Ingestion & Director LLM Engine

## 1. Objective
Implement the backend ingestion handlers and Gemini Director LLM integration to generate dynamic news storyboard JSON based on video duration.

---

## 2. Dependencies (`requirements.txt`)
Add the following libraries to `requirements.txt`:
- `google-genai` (Gemini API SDK)
- `pypdf` (PDF text extraction)
- `beautifulsoup4` (Web scraping)
- `requests` (HTTP requests for web scraping)
- `python-dotenv` (For API key management)

---

## 3. Module Breakdown

### A. Ingestion Module (`ingestion.py`)
Create a dedicated module with three helper functions:
1. **`extract_text_from_url(url: str) -> str`**:
   - Fetches the web page using `requests`.
   - Uses `BeautifulSoup` to parse clean body text (paragraphs `<p>`), stripping headers, footers, scripts, and ads.
   - Throws clear error messages if URL is unreachable.
2. **`extract_text_from_pdf(pdf_file, max_pages=5) -> str`**:
   - Reads PDF stream using `pypdf.PdfReader`.
   - Rejects PDFs exceeding 5 pages or 5 MB.
   - Concatenates extracted text across pages.
3. **`clean_text_input(raw_text: str) -> str`**:
   - Normalizes whitespace, removes duplicate newlines, and trims text.

### B. Director LLM Engine (`director.py`)
Integrate Gemini API to analyze the input text and generate structured JSON scenes.

#### WPM & Duration Math:
- Formula: `Target Words = duration_sec * 2.33` (approx. 140 WPM).
- Scene Count Calculation:
  - 60s -> ~140 words, 3 to 4 scenes
  - 100s -> ~230 words, 5 to 7 scenes
  - 120s -> ~280 words, 7 to 9 scenes

#### Structured JSON Output Format required from Gemini:
```json
[
  {
    "scene_number": 1,
    "timestamp": "00:00 - 00:15",
    "narration": "Exact spoken narration text for scene 1.",
    "visual_prompt": "480p landscape cinematic description for image generation."
  }
]
"""
Ingestion Module — AI NEWS Generator
======================================
Provides helpers for extracting article text from URLs, PDFs, and raw text.

Functions:
    extract_text_from_url  — Scrape article body from a web page.
    extract_text_from_pdf  — Read text from an uploaded PDF file.
    clean_text_input       — Normalise raw pasted text.
"""

import re
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_PDF_PAGES = 5
_MAX_PDF_BYTES = 5 * 1024 * 1024  # 5 MB

_REQUEST_TIMEOUT = 15  # seconds
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Tags that typically contain non-article boilerplate
_STRIP_TAGS = ["script", "style", "header", "footer", "nav", "aside",
               "form", "noscript", "iframe", "svg"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_url(url: str) -> str:
    """Fetch a web page and return its main body text (paragraph content).

    Parameters
    ----------
    url : str
        Full URL of the article to scrape.

    Returns
    -------
    str
        Cleaned body text extracted from ``<p>`` tags.

    Raises
    ------
    ValueError
        If the URL is unreachable, returns a non-200 status, or yields
        no readable paragraph text.
    """
    try:
        resp = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
    except requests.exceptions.MissingSchema:
        raise ValueError(
            f"Invalid URL format: '{url}'. Please include http:// or https://."
        )
    except requests.exceptions.ConnectionError:
        raise ValueError(
            f"Could not connect to '{url}'. Check the URL and your internet connection."
        )
    except requests.exceptions.Timeout:
        raise ValueError(
            f"Request to '{url}' timed out after {_REQUEST_TIMEOUT}s."
        )
    except requests.exceptions.HTTPError as exc:
        raise ValueError(
            f"HTTP error {resp.status_code} when fetching '{url}': {exc}"
        )

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove boilerplate elements
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Extract paragraph text
    paragraphs = [p.get_text(separator=" ", strip=True)
                  for p in soup.find_all("p")]
    text = "\n\n".join(p for p in paragraphs if p)

    if not text.strip():
        raise ValueError(
            f"No readable paragraph text found at '{url}'. "
            "The page may be JavaScript-rendered or paywalled."
        )

    return clean_text_input(text)


def extract_text_from_pdf(pdf_file, max_pages: int = _MAX_PDF_PAGES) -> str:
    """Read text from a PDF file object (e.g. a Streamlit ``UploadedFile``).

    Parameters
    ----------
    pdf_file : file-like
        A readable binary stream (must support ``.read()``/``.seek()``).
        Must also expose a ``.size`` attribute **or** be seekable so the
        byte length can be determined.
    max_pages : int, optional
        Reject PDFs with more pages than this (default 5).

    Returns
    -------
    str
        Concatenated and cleaned text from the PDF pages.

    Raises
    ------
    ValueError
        If the file exceeds size or page-count limits, or contains no
        extractable text.
    """
    # --- Size check ---
    size = getattr(pdf_file, "size", None)
    if size is None:
        # Fallback: seek to end to measure
        pos = pdf_file.tell()
        pdf_file.seek(0, 2)
        size = pdf_file.tell()
        pdf_file.seek(pos)

    if size > _MAX_PDF_BYTES:
        raise ValueError(
            f"PDF size ({size / (1024*1024):.1f} MB) exceeds the "
            f"{_MAX_PDF_BYTES / (1024*1024):.0f} MB limit."
        )

    # --- Read pages ---
    pdf_file.seek(0)
    reader = PdfReader(pdf_file)

    if len(reader.pages) > max_pages:
        raise ValueError(
            f"PDF has {len(reader.pages)} pages, which exceeds the "
            f"{max_pages}-page limit."
        )

    page_texts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            page_texts.append(page_text)

    text = "\n\n".join(page_texts)

    if not text.strip():
        raise ValueError(
            "Could not extract any text from the PDF. "
            "It may be image-based or encrypted."
        )

    return clean_text_input(text)


def clean_text_input(raw_text: str) -> str:
    """Normalise whitespace and trim raw text.

    - Collapses multiple consecutive blank lines into a single blank line.
    - Collapses runs of spaces/tabs within a line into a single space.
    - Strips leading/trailing whitespace.
    """
    # Collapse runs of whitespace within lines (preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", raw_text)
    # Collapse 3+ consecutive newlines into 2 (one blank line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

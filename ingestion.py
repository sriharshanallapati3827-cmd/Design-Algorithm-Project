"""
Ingestion Module — AI NEWS Video Generator
===========================================
Provides helpers for extracting article text from URLs, PDFs, and raw text.
Uses resilient HTTP sessions with automatic retries and timeouts.

Functions:
    extract_text_from_url  — Scrape article body from a web page.
    extract_text_from_pdf  — Read text from an uploaded PDF file.
    clean_text_input       — Normalise raw pasted text.
"""

import re
import time
from bs4 import BeautifulSoup
from pypdf import PdfReader

from network_utils import (
    get_resilient_session,
    DEFAULT_TIMEOUT,
    NETWORK_EXCEPTIONS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_PDF_PAGES = 5
_MAX_PDF_BYTES = 5 * 1024 * 1024  # 5 MB

# Tags that typically contain non-article boilerplate
_STRIP_TAGS = [
    "script", "style", "header", "footer", "nav", "aside",
    "form", "noscript", "iframe", "svg"
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_url(url: str, timeout: tuple[int, int] = DEFAULT_TIMEOUT) -> str:
    """Fetch a web page and return its main body text (paragraph content).

    Parameters
    ----------
    url : str
        Full URL of the article to scrape.
    timeout : tuple[int, int], optional
        (connect_timeout, read_timeout) in seconds.

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
    import requests

    session = get_resilient_session()
    last_exc = None

    for attempt in range(1, 3):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            last_exc = None
            break
        except requests.exceptions.MissingSchema:
            raise ValueError(
                f"Invalid URL format: '{url}'. Please include http:// or https://."
            )
        except requests.exceptions.HTTPError as exc:
            raise ValueError(
                f"HTTP error {exc.response.status_code if exc.response else 'unknown'} "
                f"when fetching '{url}': {exc}"
            )
        except NETWORK_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(3.0)
                continue
            raise ValueError(
                f"Connection error or timeout reaching '{url}' after retries: {exc}"
            )
        except Exception as exc:
            raise ValueError(f"Failed to fetch '{url}': {exc}")

    if last_exc:
        raise ValueError(f"Could not connect to '{url}': {last_exc}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove boilerplate elements
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Extract paragraph text
    paragraphs = [
        p.get_text(separator=" ", strip=True)
        for p in soup.find_all("p")
    ]
    text = "\n\n".join(p for p in paragraphs if p)

    if not text.strip():
        raise ValueError(
            f"No readable paragraph text found at '{url}'. "
            "The page may be JavaScript-rendered or paywalled."
        )

    return clean_text_input(text)


def extract_text_from_pdf(pdf_file, max_pages: int = _MAX_PDF_PAGES) -> str:
    """Read text from a PDF file object (e.g. a Streamlit ``UploadedFile``)."""
    # Size check
    size = getattr(pdf_file, "size", None)
    if size is None:
        pos = pdf_file.tell()
        pdf_file.seek(0, 2)
        size = pdf_file.tell()
        pdf_file.seek(pos)

    if size > _MAX_PDF_BYTES:
        raise ValueError(
            f"PDF size ({size / (1024*1024):.1f} MB) exceeds the "
            f"{_MAX_PDF_BYTES / (1024*1024):.0f} MB limit."
        )

    # Read pages
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
    """Normalise whitespace and trim raw text."""
    text = re.sub(r"[^\S\n]+", " ", raw_text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

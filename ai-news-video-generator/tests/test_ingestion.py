"""
Tests for ingestion.py — URL scraping, PDF extraction, text cleaning.
"""

import io
import pytest
from unittest.mock import MagicMock, patch

from ingestion import extract_text_from_url, extract_text_from_pdf, clean_text_input


# ===================================================================
# clean_text_input
# ===================================================================

class TestCleanTextInput:
    """Unit tests for whitespace normalisation."""

    def test_collapses_multiple_spaces(self):
        assert clean_text_input("hello   world") == "hello world"

    def test_collapses_tabs(self):
        assert clean_text_input("hello\t\tworld") == "hello world"

    def test_collapses_excessive_newlines(self):
        result = clean_text_input("line1\n\n\n\nline2")
        assert result == "line1\n\nline2"

    def test_preserves_double_newline(self):
        result = clean_text_input("para1\n\npara2")
        assert result == "para1\n\npara2"

    def test_strips_leading_trailing_whitespace(self):
        assert clean_text_input("  hello  ") == "hello"

    def test_empty_string(self):
        assert clean_text_input("") == ""

    def test_only_whitespace(self):
        assert clean_text_input("   \n\n\t  ") == ""

    def test_mixed_whitespace_and_newlines(self):
        result = clean_text_input("  hello  \t world  \n\n\n\n  foo  ")
        assert "hello world" in result
        assert "foo" in result
        # Should have at most one blank line between paragraphs
        assert "\n\n\n" not in result


# ===================================================================
# extract_text_from_url
# ===================================================================

class TestExtractTextFromUrl:
    """Unit tests for web scraping with mocked HTTP responses."""

    @patch("ingestion.requests.get")
    def test_extracts_paragraph_text(self, mock_get, sample_html):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = sample_html
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = extract_text_from_url("https://example.com/article")

        assert "Artemis IV" in result
        assert "lunar orbit" in result
        assert "astronauts" in result

    @patch("ingestion.requests.get")
    def test_strips_boilerplate_tags(self, mock_get, sample_html):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = sample_html
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = extract_text_from_url("https://example.com/article")

        # Script, style, header, footer content should be stripped
        assert "var x = 1" not in result
        assert "color: red" not in result
        assert "Copyright 2026" not in result
        assert "Menu" not in result

    @patch("ingestion.requests.get")
    def test_raises_on_invalid_url(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.MissingSchema("Invalid URL")

        with pytest.raises(ValueError, match="Invalid URL format"):
            extract_text_from_url("not-a-url")

    @patch("ingestion.requests.get")
    def test_raises_on_connection_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError()

        with pytest.raises(ValueError, match="Could not connect"):
            extract_text_from_url("https://unreachable.example.com")

    @patch("ingestion.requests.get")
    def test_raises_on_timeout(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.Timeout()

        with pytest.raises(ValueError, match="timed out"):
            extract_text_from_url("https://slow.example.com")

    @patch("ingestion.requests.get")
    def test_raises_on_empty_page(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><div>No paragraphs here</div></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="No readable paragraph text"):
            extract_text_from_url("https://example.com/empty")


# ===================================================================
# extract_text_from_pdf
# ===================================================================

class TestExtractTextFromPdf:
    """Unit tests for PDF extraction with mocked pypdf."""

    @patch("ingestion.PdfReader")
    def test_extracts_text_from_pdf(self, mock_reader_cls):
        # Create a fake PDF file object
        pdf_file = io.BytesIO(b"fake pdf content")
        pdf_file.size = 1000  # well under 5 MB

        # Mock the PdfReader
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page one content about Artemis."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        result = extract_text_from_pdf(pdf_file)
        assert "Artemis" in result

    def test_rejects_oversized_pdf(self):
        pdf_file = io.BytesIO(b"x" * 100)
        pdf_file.size = 6 * 1024 * 1024  # 6 MB, exceeds limit

        with pytest.raises(ValueError, match="exceeds the.*MB limit"):
            extract_text_from_pdf(pdf_file)

    @patch("ingestion.PdfReader")
    def test_rejects_too_many_pages(self, mock_reader_cls):
        pdf_file = io.BytesIO(b"fake pdf")
        pdf_file.size = 1000

        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock() for _ in range(10)]  # 10 pages
        mock_reader_cls.return_value = mock_reader

        with pytest.raises(ValueError, match="exceeds the.*page limit"):
            extract_text_from_pdf(pdf_file)

    @patch("ingestion.PdfReader")
    def test_rejects_empty_pdf(self, mock_reader_cls):
        pdf_file = io.BytesIO(b"fake pdf")
        pdf_file.size = 1000

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        with pytest.raises(ValueError, match="Could not extract any text"):
            extract_text_from_pdf(pdf_file)

    @patch("ingestion.PdfReader")
    def test_size_fallback_via_seek(self, mock_reader_cls):
        """When .size attribute is absent, should measure via seek."""
        pdf_file = io.BytesIO(b"fake pdf content here")
        # No .size attribute — force the seek-based fallback

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Seekable PDF content."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        result = extract_text_from_pdf(pdf_file)
        assert "Seekable PDF" in result

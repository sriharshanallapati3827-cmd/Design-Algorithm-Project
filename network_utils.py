"""
network_utils.py — Resilient HTTP Session & Download Utilities
===============================================================
Provides shared requests.Session configured with robust retry logic,
exponential backoff, connection keep-alive, chunked streaming, and
resilient exception handling for cloud and local deployments.
"""

from __future__ import annotations

import http.client
import io
import logging
import time
from typing import Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT: Tuple[int, int] = (10, 90)  # (connect_timeout, read_timeout)
RETRY_STATUS_CODES = [408, 429, 500, 502, 503, 504]
CHUNK_SIZE = 8192

# Exceptions indicating transient socket / connection drop
NETWORK_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    http.client.RemoteDisconnected,
)


# ---------------------------------------------------------------------------
# Timeout Adapter to guarantee connect & read timeouts on all requests
# ---------------------------------------------------------------------------

class TimeoutHTTPAdapter(HTTPAdapter):
    """An HTTPAdapter that enforces a default (connect, read) timeout."""

    def __init__(self, *args, timeout: Tuple[int, int] = DEFAULT_TIMEOUT, **kwargs):
        self.default_timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self.default_timeout
        return super().send(request, **kwargs)


# ---------------------------------------------------------------------------
# Shared Session Singleton
# ---------------------------------------------------------------------------

_SHARED_SESSION: Optional[requests.Session] = None


def get_resilient_session(
    total_retries: int = 5,
    backoff_factor: float = 2.0,
    connect_timeout: int = 10,
    read_timeout: int = 90,
) -> requests.Session:
    """Return a shared requests.Session with exponential backoff and keep-alive.

    Parameters
    ----------
    total_retries : int
        Maximum number of retries for transient status codes.
    backoff_factor : float
        Multiplier for exponential backoff: {backoff factor} * (2 ** (retry - 1))
    connect_timeout : int
        Connection timeout in seconds.
    read_timeout : int
        Read timeout in seconds.

    Returns
    -------
    requests.Session
        Configured session with HTTPAdapter mounted on http:// and https://.
    """
    global _SHARED_SESSION
    if _SHARED_SESSION is not None:
        return _SHARED_SESSION

    session = requests.Session()

    # Enable TCP Keep-Alive headers to keep middle-mile proxies open
    session.headers.update({
        "Connection": "keep-alive",
        "Keep-Alive": "timeout=60, max=100",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36 AI-News-Generator/2.0"
        ),
    })

    retry_strategy = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=RETRY_STATUS_CODES,
        allowed_methods=["GET", "POST", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )

    adapter = TimeoutHTTPAdapter(
        max_retries=retry_strategy,
        timeout=(connect_timeout, read_timeout),
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    _SHARED_SESSION = session
    return _SHARED_SESSION


# ---------------------------------------------------------------------------
# Chunked Streaming Download Helper with Fallback Retry
# ---------------------------------------------------------------------------

def stream_download_bytes(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    session: Optional[requests.Session] = None,
    timeout: Tuple[int, int] = DEFAULT_TIMEOUT,
    chunk_size: int = CHUNK_SIZE,
    max_attempts: int = 2,
    fallback_delay: float = 3.0,
) -> bytes:
    """Download binary content using chunked streaming with transient retry.

    Parameters
    ----------
    url : str
        Target download URL.
    params : dict, optional
        Query parameters.
    headers : dict, optional
        Additional request headers.
    session : requests.Session, optional
        Existing session (defaults to resilient singleton).
    timeout : tuple, optional
        (connect_timeout, read_timeout).
    chunk_size : int, optional
        Chunk size in bytes for streaming (default 8192).
    max_attempts : int, optional
        Outer retry attempts with fallback delay for dropped connections.
    fallback_delay : float, optional
        Sleep duration before retry attempt on connection drop.

    Returns
    -------
    bytes
        Downloaded content bytes.

    Raises
    ------
    requests.exceptions.RequestException
        If the download fails after all attempts and backoffs.
    """
    sess = session or get_resilient_session()
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = sess.get(
                url,
                params=params,
                headers=headers,
                stream=True,
                timeout=timeout,
            )
            resp.raise_for_status()

            buffer = io.BytesIO()
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    buffer.write(chunk)

            data = buffer.getvalue()
            if not data:
                raise ValueError("Server returned 0 bytes content.")
            return data

        except NETWORK_EXCEPTIONS as exc:
            last_exc = exc
            logger.warning(
                "Network drop on %s (attempt %d/%d): %s. Waiting %.1fs...",
                url, attempt, max_attempts, exc, fallback_delay
            )
            if attempt < max_attempts:
                time.sleep(fallback_delay)
                continue
            raise

    if last_exc:
        raise last_exc
    raise RuntimeError(f"Download failed for {url}")

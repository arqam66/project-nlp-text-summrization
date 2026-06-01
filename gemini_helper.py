"""Utility helpers for free‑tier summarization and simple keyword analysis.

The original code expected a Google Gemini implementation, but the project does not ship a
`gemini_helper.py` module.  Importing it caused a runtime error, breaking the Flask
app.  This module provides lightweight, zero‑cost replacements:

* ``generate_gemini_summary`` – wraps the free SummarizeAPI service (no API key
  required).  It maps the requested sentence count to the API’s ``length``
  parameter (short/medium/long).
* ``analyze_text_words`` – performs a tiny keyword‑frequency analysis locally –
  enough to keep the ``/analyze_words`` endpoint functional without external
  credentials.

Both functions raise ``RuntimeError`` with a clear message if the HTTP request
fails, allowing the Flask error‑handling in ``app.py`` to return a JSON error
payload.
"""

from __future__ import annotations

import collections
import json
import logging
import re
from typing import List, Dict, Any

# ``requests`` is a tiny dependency; it is listed in ``requirements.txt``.
try:  # pragma: no‑cover – exercised where ``requests`` is installed.
    import requests
except Exception:  # pragma: no‑cover – fallback to standard library.
    import urllib.request
    import urllib.error
    requests = None  # type: ignore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: free SummarizeAPI wrapper
# ---------------------------------------------------------------------------
_SUMMARIZE_API_URL = "https://summarizeapi.com/"

def _call_summarize_api(text: str, length: str = "short") -> str:
    """Send *text* to the SummarizeAPI free endpoint.

    Parameters
    ----------
    text:
        The article or passage to be summarised.
    length:
        Desired length preset – ``short`` (≈2‑3 sentences), ``medium`` (≈4‑5),
        or ``long`` (≈6‑10).

    Returns
    -------
    str
        The summary string returned by the API.
    """
    payload = {"text": text, "length": length, "style": "professional", "format": "paragraph"}
    headers = {"Content-Type": "application/json"}
    log.debug("Calling SummarizeAPI %s with length=%s", _SUMMARIZE_API_URL, length)

    if requests:
        response = requests.post(_SUMMARIZE_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    else:
        req = urllib.request.Request(
            _SUMMARIZE_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as fh:
                raw = fh.read().decode("utf-8")
            data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"SummarizeAPI HTTP error {exc.code}: {exc.reason}") from exc
        except Exception as exc:  # pragma: no‑cover – generic network failure.
            raise RuntimeError(f"SummarizeAPI request failed: {exc}") from exc

    summary = data.get("summary")
    if not summary:
        raise RuntimeError(f"SummarizeAPI response missing 'summary': {data!r}")
    return summary

# ---------------------------------------------------------------------------
# Public API – generate_gemini_summary
# ---------------------------------------------------------------------------
def generate_gemini_summary(text: str, num_sentences: int = 3, api_key: str | None = None) -> str:
    """Return an abstractive summary using the free SummarizeAPI.

    The original application expected a Google Gemini call that required an API
    key.  This implementation ignores ``api_key`` (it is kept for signature
    compatibility) and maps ``num_sentences`` onto the closest ``length`` preset
    understood by SummarizeAPI.
    """
    if not text:
        raise ValueError("`text` must be a non‑empty string")
    # Map sentence count to length category.
    if num_sentences <= 2:
        length = "short"
    elif num_sentences <= 5:
        length = "medium"
    else:
        length = "long"
    return _call_summarize_api(text, length=length)

# ---------------------------------------------------------------------------
# Simple keyword analysis – a replacement for the Gemini‑based version.
# ---------------------------------------------------------------------------
def analyze_text_words(text: str, api_key: str | None = None) -> List[Dict[str, Any]]:
    """Extract the top‑10 most frequent non‑stop‑word tokens.

    This lightweight implementation is sufficient for the ``/analyze_words``
    endpoint.  It does not contact any external service – the ``api_key``
    argument is accepted for API‑compatibility only.
    """
    if not text:
        raise ValueError("`text` must be a non‑empty string")
    _STOP_WORDS = {
        "the", "and", "or", "a", "an", "in", "on", "of", "to", "is",
        "are", "was", "were", "be", "been", "has", "have", "it", "its",
        "for", "by", "with", "as", "that", "this", "at", "from",
    }
    # Normalise and strip punctuation.
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = [t for t in cleaned.split() if t and t not in _STOP_WORDS]
    freqs = collections.Counter(tokens)
    return [{"word": w, "frequency": c} for w, c in freqs.most_common(10)]

# End of gemini_helper.py

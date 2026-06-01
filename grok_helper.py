from __future__ import annotations

import json
import logging
import os
from typing import List, Dict, Any

try:
    import requests
except Exception:
    import urllib.request
    import urllib.error
    requests = None

log = logging.getLogger(__name__)

_XAI_API_BASE = "https://api.x.ai/v1"

def _call_xai_chat(messages: list, api_key: str, model: str = "grok-2-1212", max_tokens: int = 500, temperature: float = 0.3) -> str:
    if not api_key:
        raise ValueError("xAI API key is required")
    url = f"{_XAI_API_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    log.debug("Calling xAI API with model=%s", model)

    if requests:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as fh:
                raw = fh.read().decode("utf-8")
            data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"xAI API HTTP error {exc.code}: {exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"xAI API request failed: {exc}") from exc

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"xAI API response missing choices: {data!r}")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"xAI API response empty content: {data!r}")
    return content.strip()

def generate_grok_summary(text: str, num_sentences: int = 3, api_key: str | None = None) -> str:
    if not text:
        raise ValueError("`text` must be a non-empty string")
    if not api_key:
        api_key = os.environ.get("XAI_API_KEY", "")
    system_prompt = "You are a helpful assistant that summarizes text accurately and concisely."
    user_prompt = f"Summarize the following text in about {num_sentences} sentences:\n\n{text}"
    return _call_xai_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        api_key=api_key,
        max_tokens=200 * num_sentences,
    )

def analyze_text_grok(text: str, api_key: str | None = None) -> List[Dict[str, Any]]:
    if not text:
        raise ValueError("`text` must be a non-empty string")
    if not api_key:
        api_key = os.environ.get("XAI_API_KEY", "")
    system_prompt = "You are a helpful assistant that analyzes text and extracts key terms with their frequencies."
    user_prompt = (
        f"Analyze the following text and extract the top 10 most important keywords or phrases. "
        f"Return ONLY a JSON array of objects with keys 'word' (string) and 'frequency' (number), "
        f"sorted by frequency descending. No additional text.\n\n{text}"
    )
    result = _call_xai_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        api_key=api_key,
        max_tokens=500,
        temperature=0.1,
    )
    import re
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result)
    if json_match:
        result = json_match.group(1)
    try:
        parsed = json.loads(result)
        if isinstance(parsed, list):
            return parsed
        raise ValueError("Response is not a list")
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("xAI returned non-JSON for word analysis: %s", result)
        raise RuntimeError(f"Failed to parse xAI response as JSON: {e}") from e

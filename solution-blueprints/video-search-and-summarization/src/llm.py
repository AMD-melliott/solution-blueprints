# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""OpenAI-compatible clients for the VLM and LLM served by ROCm vLLM."""

import json
import re
from typing import Dict, Iterator, List, Optional

import httpx
from config import (
    API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    REQUEST_TIMEOUT_S,
    VLM_BASE_URL,
    VLM_MODEL,
)
from media import jpeg_to_data_uri


def _chat(
    base_url: str,
    model: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    extra: Optional[Dict] = None,
) -> str:
    payload: Dict = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if extra:
        payload.update(extra)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        resp = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    content = data["choices"][0]["message"].get("content") or ""
    # Qwen3 hybrid-reasoning models may emit a <think>...</think> block; drop it from the answer.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip()


def vlm_caption(frames: List[bytes], prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    content: List[Dict] = [{"type": "text", "text": prompt}]
    for jpg in frames:
        content.append({"type": "image_url", "image_url": {"url": jpeg_to_data_uri(jpg)}})
    messages: List[Dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return _chat(VLM_BASE_URL, VLM_MODEL, messages, max_tokens=max_tokens, temperature=0.0)


def llm_complete(prompt: str, system: str = "", max_tokens: int = 2048) -> str:
    messages: List[Dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _chat(
        LLM_BASE_URL,
        LLM_MODEL,
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        extra={"chat_template_kwargs": {"enable_thinking": False}},
    )


def llm_stream(prompt: str, system: str = "", max_tokens: int = 2048) -> Iterator[str]:
    """Yield answer text deltas from the LLM via OpenAI-compatible SSE streaming."""
    messages: List[Dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: Dict = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[len("data:") :].strip()
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                piece = (choices[0].get("delta") or {}).get("content") or ""
                if piece:
                    yield piece

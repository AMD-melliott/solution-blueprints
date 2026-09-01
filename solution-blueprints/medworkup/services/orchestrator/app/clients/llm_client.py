# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx
from app.config import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM service fails after all retries."""

    def __init__(self, status: Optional[int], detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"LLM HTTP {status}: {detail}")


class LLMClient:
    """
    Async client for the MedGemma OpenAI-compatible chat endpoint.

    Responsibilities
    ----------------
    - Build and POST /v1/chat/completions payloads.
    - Retry on transient network errors and 5xx responses.
    - Strip markdown fences and JSON-parse structured responses.
    - Never expose raw httpx internals to callers.
    """

    _MAX_RETRIES = 2  # total retry attempts after the first try
    _RETRY_BACKOFF = 1.0  # seconds; doubles each retry

    def __init__(self) -> None:
        self._base_url = settings.llm_base_url.rstrip("/")
        self._timeout = settings.llm_timeout
        self._api_key = settings.llm_api_key

    # ── public API ────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat-completion request and return the assistant's reply as a
        plain string.

        Raises LLMError after all retries are exhausted.
        """
        payload: Dict[str, Any] = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_tokens": max_tokens if max_tokens is not None else settings.llm_max_tokens,
        }

        raw_content = await self._post_with_retry(payload)
        logger.debug("[LLM] Response (%d chars): %.200s…", len(raw_content), raw_content)
        return raw_content

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Same as `chat()` but expects a JSON response.

        Strips surrounding markdown fences (```json … ```) then parses.
        Raises ValueError with the raw content if parsing fails so the
        caller can apply a domain-specific fallback.
        """
        raw = await self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return self._parse_json(raw)

    # ── internals ─────────────────────────────────────────────────────────────

    async def _post_with_retry(self, payload: Dict[str, Any]) -> str:
        """POST the payload, retrying on transient failures. Returns content string."""
        url = f"{self._base_url}/v1/chat/completions"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._MAX_RETRIES + 2):  # attempts: 1, 2, 3
            try:
                content = await self._single_post(url, payload)
                return content

            except LLMError as exc:
                # 4xx errors are not retryable (bad prompt, auth, etc.)
                if exc.status is not None and 400 <= exc.status < 500:
                    raise
                last_exc = exc

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc

            if attempt <= self._MAX_RETRIES:
                wait = self._RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "[LLM] Attempt %d failed (%s), retrying in %.1fs…",
                    attempt,
                    last_exc,
                    wait,
                )
                await asyncio.sleep(wait)

        raise LLMError(None, f"All {self._MAX_RETRIES + 1} attempts failed: {last_exc}")

    async def _single_post(self, url: str, payload: Dict[str, Any]) -> str:
        """
        Execute a single POST and return the assistant content string.

        Keeps `resp` in scope until after `.json()` is called — fixing the
        original bug where `resp.json()` was called outside the `async with`
        block, which is unsafe for streaming responses.
        """
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)

            if resp.status_code >= 400:
                raise LLMError(resp.status_code, resp.text[:300])

            try:
                data = resp.json()
            except Exception as exc:
                raise LLMError(resp.status_code, f"Non-JSON response: {exc}") from exc

        # Parse outside the client context (content already buffered)
        try:
            content: str = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(None, f"Unexpected response schema: {exc} — {str(data)[:300]}") from exc

        return content

    @staticmethod
    def _parse_json(raw: str) -> Any:
        """
        Strip optional markdown fences and parse JSON.

        Handles:
          - ```json\\n{...}\\n```
          - ```\\n{...}\\n```
          - bare {...} or [...]
        """
        cleaned = raw.strip()
        # Remove leading fence (with optional language tag)
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
        # Remove trailing fence
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(
                "[LLM] JSON parse failed: %s\nFirst 500 chars of raw response:\n%s",
                exc,
                raw[:500],
            )
            raise ValueError(f"LLM returned content that could not be parsed as JSON: {exc}") from exc

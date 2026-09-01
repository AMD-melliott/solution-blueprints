# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Base async HTTP client with retry + structured error handling.
All service clients inherit from this.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx
from app.config import settings

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """Raised when a downstream service call fails after retries."""

    def __init__(self, service: str, status: Optional[int], detail: str) -> None:
        self.service = service
        self.status = status
        self.detail = detail
        super().__init__(f"[{service}] HTTP {status}: {detail}")


class BaseClient:
    def __init__(self, base_url: str, service_name: str, timeout: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._service_name = service_name
        self._timeout = timeout
        self._max_retries = settings.http_max_retries

    async def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        url = f"{self._base_url}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 2):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code >= 500:
                        raise ServiceError(self._service_name, resp.status_code, resp.text[:300])
                    resp.raise_for_status()
                    return resp.json()
            except ServiceError:
                raise
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt <= self._max_retries:
                    wait = 0.5 * attempt
                    logger.warning(
                        "[%s] attempt %d failed (%s), retrying in %.1fs",
                        self._service_name,
                        attempt,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)
            except httpx.HTTPStatusError as exc:
                raise ServiceError(self._service_name, exc.response.status_code, exc.response.text[:300]) from exc

        raise ServiceError(self._service_name, None, str(last_exc))

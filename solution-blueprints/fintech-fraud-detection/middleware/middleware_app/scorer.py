# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Scorer: takes a raw transaction row, returns a fraud probability + latency.

If WORKER_URL is set, calls the real worker's POST /score.
If unset, returns a random probability biased by the row's isFraud label
(when available) so the demo is visually interesting.
"""
import asyncio
import logging
import random
import time
from typing import Any

import httpx
from app.config import WORKER_URL

_log = logging.getLogger(__name__)

# Fields the real worker's TransactionIn schema knows about.
# We forward these; extra fields are allowed (extra="allow" in their schema).
_WORKER_FIELDS = {
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "TransactionDT",
    "C1",
    "C2",
    "C14",
    "D1",
    "D4",
    "D10",
    "D15",
    "M4",
    "M6",
}


def _build_worker_payload(row: dict) -> dict:
    """Pick fields the worker schema expects, drop None values."""
    payload = {}
    for k in _WORKER_FIELDS:
        v = row.get(k)
        if v is not None:
            payload[k] = v
    # TransactionAmt is required and must be > 0
    if "TransactionAmt" not in payload or payload["TransactionAmt"] <= 0:
        payload["TransactionAmt"] = 1.0
    return payload


async def score_real(client: httpx.AsyncClient, row: dict) -> tuple[float, int, list[dict]]:
    """Call the real worker's /score endpoint. Returns (probability, latency_ms, rules_matched)."""
    payload = _build_worker_payload(row)
    url = f"{WORKER_URL}/score"
    _log.info("→ POST %s  amt=%.2f  fields=%s", url, payload.get("TransactionAmt", 0), list(payload.keys()))
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            url,
            params={"mode": "xgb_ensemble"},
            json=payload,
            timeout=120.0,
        )
    except Exception as exc:
        _log.error("backend unreachable %s: %s", url, exc)
        raise
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if not resp.is_success:
        _log.error("← backend %s (%dms): %s", resp.status_code, latency_ms, resp.text[:200])
    else:
        _log.info("← backend %s (%dms)  prob=%s", resp.status_code, latency_ms, resp.json().get("fraud_probability"))
    resp.raise_for_status()
    data = resp.json()
    return float(data["fraud_probability"]), latency_ms, data.get("rules_matched", [])


async def score_batch_real(client: httpx.AsyncClient, rows: list[dict], mode: str) -> tuple[list[float], int]:
    """Call backend /score/batch. Returns (list of probabilities, total latency_ms)."""
    payloads = [_build_worker_payload(row) for row in rows]
    url = f"{WORKER_URL}/score/batch"
    _log.info("→ POST %s  n=%d  mode=%s", url, len(payloads), mode)
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json={"transactions": payloads, "mode": mode}, timeout=300.0)
    except Exception as exc:
        _log.error("backend unreachable %s: %s", url, exc)
        raise
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if not resp.is_success:
        _log.error("← backend %s (%dms): %s", resp.status_code, latency_ms, resp.text[:200])
    else:
        _log.info("← backend %s (%dms)  n=%d", resp.status_code, latency_ms, len(rows))
    resp.raise_for_status()
    data = resp.json()
    return [float(r["fraud_probability"]) for r in data["results"]], latency_ms


async def score_fake(row: dict) -> tuple[float, int, list[dict]]:
    """Return a random probability biased by isFraud label, plus fake latency."""
    is_fraud = row.get("isFraud")
    if is_fraud == 1:
        # True fraud — score skewed high but not always above fraud threshold
        # (so we get a mix of true positives, missed fraud, and review)
        prob = random.betavariate(5, 2)  # mean ~0.71
    else:
        # Clean — score skewed low but occasionally elevated (false positives)
        prob = random.betavariate(1.5, 6)  # mean ~0.2
    # Simulate model latency: 20–80ms most of the time, occasional spike
    latency_ms = random.randint(20, 80)
    if random.random() < 0.05:
        latency_ms += random.randint(50, 200)
    # Tiny sleep to make latency feel realistic in the UI
    await asyncio.sleep(latency_ms / 1000)
    return prob, latency_ms, []


async def reset_real(client: httpx.AsyncClient) -> None:
    """Tell the backend to clear stateful inference context (the GNN rolling graph).

    Best-effort: a failure here must not prevent the stream from stopping, so the
    error is logged and swallowed.
    """
    url = f"{WORKER_URL}/reset"
    try:
        resp = await client.post(url, timeout=30.0)
        resp.raise_for_status()
        _log.info("→ POST %s  backend state reset", url)
    except Exception as exc:
        _log.error("backend reset failed %s: %s", url, exc)


class Scorer:
    """Wraps score_real or score_fake depending on config."""

    def __init__(self) -> None:
        self._mode = "real" if WORKER_URL else "fake"
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "Scorer":
        if self._mode == "real":
            self._client = httpx.AsyncClient()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def mode(self) -> str:
        return self._mode

    async def score(self, row: dict) -> tuple[float, int, list[dict]]:
        if self._mode == "real":
            assert self._client is not None
            return await score_real(self._client, row)
        return await score_fake(row)

    async def reset(self) -> None:
        """Clear backend stateful context. No-op in fake mode (no backend, no graph)."""
        if self._mode == "real" and self._client is not None:
            await reset_real(self._client)

    async def score_batch(self, rows: list[dict], mode: str = "xgb_ensemble") -> tuple[list[float], int]:
        if self._mode == "real":
            assert self._client is not None
            return await score_batch_real(self._client, rows, mode)
        probs = []
        total_ms = 0
        for row in rows:
            prob, lat, _ = await score_fake(row)
            probs.append(prob)
            total_ms += lat
        return probs, total_ms

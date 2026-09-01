# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Dict, Generator


class PipelineTimer:
    """
    Accumulates per-stage wall-clock timings.

    Provides both a synchronous and an async context manager so it can be
    used regardless of whether the stage body contains awaits.

    Usage (sync stage):
        with timer.stage("step3_aggregate"):
            result = aggregate_entities(raw)

    Usage (async stage – preferred for I/O-bound steps):
        async with timer.async_stage("step2_ner"):
            result = await extract_entities(preprocessed, ner_client)
    """

    def __init__(self) -> None:
        self._timings: Dict[str, float] = {}
        self._start = time.perf_counter()

    # ── sync ──────────────────────────────────────────────────────────────
    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Sync context manager. Correctly records time even if the body raises."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._timings[name] = round((time.perf_counter() - t0) * 1000, 1)

    # ── async ─────────────────────────────────────────────────────────────
    @asynccontextmanager
    async def async_stage(self, name: str) -> AsyncGenerator[None, None]:
        """
        Async context manager for I/O-bound stages.

        Identical timing semantics to `stage()` but safe to use with
        `await` expressions — the `finally` block is guaranteed to run
        even if the awaited coroutine raises or is cancelled.
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._timings[name] = round((time.perf_counter() - t0) * 1000, 1)

    # ── results ───────────────────────────────────────────────────────────
    def timings(self) -> Dict[str, float]:
        """Return a copy of all recorded stage timings (ms)."""
        return dict(self._timings)

    def total_ms(self) -> float:
        """Wall-clock ms since this timer was created."""
        return round((time.perf_counter() - self._start) * 1000, 1)

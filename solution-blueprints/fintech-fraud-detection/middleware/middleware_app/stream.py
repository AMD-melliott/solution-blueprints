# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Stream engine: iterates the demo dataset, scores each row, applies thresholds
to derive a label, broadcasts the scored transaction to SSE subscribers.

State machine: idle -> running <-> paused -> stopped -> idle (via restart)
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator

from app.config import STREAM_INTERVAL_MS
from app.data_loader import get_dataset
from app.scorer import Scorer

log = logging.getLogger("stream")


# ─── Public types ───────────────────────────────────────────────────────────


@dataclass
class Thresholds:
    review: float = 0.65
    fraud: float = 0.90


@dataclass
class Metrics:
    total: int = 0
    clean: int = 0
    review: int = 0
    fraud: int = 0
    latency_sum_ms: int = 0
    latency_min_ms: int | None = None
    latency_max_ms: int | None = None

    # Quality (only counted when ground truth is available)
    gt_total: int = 0  # rows with known ground truth
    correct: int = 0  # decision agreed with GT
    gt_fraud_total: int = 0  # rows where GT == fraud
    gt_clean_total: int = 0  # rows where GT == clean
    missed_fraud: int = 0  # GT fraud, we said NOT_FRAUD (false negatives)
    false_positive: int = 0  # GT clean, we said FRAUD (annoying for customers)

    # Exposure (USD)
    missed_fraud_usd: float = 0.0
    false_positive_usd: float = 0.0

    def to_dict(self) -> dict:
        avg = (self.latency_sum_ms / self.total) if self.total else 0
        accuracy_pct = (self.correct / self.gt_total * 100) if self.gt_total else None
        missed_pct = self.missed_fraud / self.gt_fraud_total * 100 if self.gt_fraud_total else None
        fp_pct = self.false_positive / self.gt_clean_total * 100 if self.gt_clean_total else None
        return {
            "total": self.total,
            "clean": self.clean,
            "review": self.review,
            "fraud": self.fraud,
            "latency_min_ms": self.latency_min_ms,
            "latency_avg_ms": int(avg) if self.total else None,
            "latency_max_ms": self.latency_max_ms,
            "accuracy_pct": accuracy_pct,
            "missed_fraud_pct": missed_pct,
            "false_positive_pct": fp_pct,
            "missed_fraud_usd": round(self.missed_fraud_usd, 2),
            "false_positive_usd": round(self.false_positive_usd, 2),
        }


def _decide(prob: float, th: Thresholds) -> str:
    if prob >= th.fraud:
        return "FRAUD"
    if prob >= th.review:
        return "REVIEW"
    return "NOT_FRAUD"


def _card_label(row: dict) -> str:
    """Build the card label per spec §5.4.1."""
    card4 = row.get("card4")
    card1 = row.get("card1")
    card2 = row.get("card2")
    network = card4.title() if isinstance(card4, str) else "Unknown"
    parts = [f"{network} · card {int(card1)}"] if card1 is not None else [network]
    if card2 is not None:
        parts.append(str(int(card2)))
    return " · ".join(parts)


# ─── Field prettifiers (IEEE-CIS dataset decoding lives here, spec §5.4) ─────

# ProductCD letter meanings were never officially published by Vesta; these are
# the common community conventions. The raw letter is kept in parens so the
# label is readable without asserting an unconfirmed decoding. Unknown codes
# fall through to the raw value rather than disappearing.
_PRODUCT_MAP = {
    "W": "Web (W)",
    "C": "Card (C)",
    "R": "Recurring (R)",
    "H": "Hospitality (H)",
    "S": "Service (S)",
}

_CARD6_MAP = {
    "credit": "Credit card",
    "debit": "Debit card",
    "charge card": "Charge card",
    "debit or credit": "Debit or credit",
}


def _product_label(row: dict) -> str | None:
    v = row.get("ProductCD")
    if v is None:
        return None
    return _PRODUCT_MAP.get(str(v), str(v))


def _card_type_label(row: dict) -> str | None:
    v = row.get("card6")
    if v is None:
        return None
    return _CARD6_MAP.get(str(v).strip().lower(), str(v))


def _str_or_none(v) -> str | None:
    """Normalize a raw cell to a clean string, or None if empty/NaN-ish."""
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


# ─── Stream state ───────────────────────────────────────────────────────────


class StreamEngine:
    """Single-instance stream coordinator. Survives across HTTP requests."""

    def __init__(self) -> None:
        self.state: str = "idle"
        self.thresholds = Thresholds()
        self.metrics = Metrics()
        self._position: int = 0
        self._scorer: Scorer | None = None
        self._task: asyncio.Task | None = None
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        self._scorer = Scorer()
        await self._scorer.__aenter__()
        log.info("scorer mode: %s", self._scorer.mode)

    async def shutdown(self) -> None:
        await self.stop()
        if self._scorer:
            await self._scorer.__aexit__()

    async def start(self) -> None:
        async with self._lock:
            if self.state == "running":
                return
            if self.state == "paused":
                self.state = "running"
                self._resume_event.set()
                await self._broadcast_state()
                return
            self.metrics = Metrics()
            self.state = "running"
            self._resume_event.set()
            self._task = asyncio.create_task(self._run())
            await self._broadcast_state()

    async def pause(self) -> None:
        async with self._lock:
            if self.state != "running":
                return
            self.state = "paused"
            self._resume_event.clear()
            await self._broadcast_state()

    async def stop(self) -> None:
        async with self._lock:
            if self.state == "idle":
                return
            self.state = "stopped"
            self._resume_event.set()
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    log.debug("stream task cancellation acknowledged during stop()")
            self._task = None
            # Full flush: unlike pause (which keeps the position so it can resume),
            # stop rewinds to the start of the dataset AND clears the backend's
            # stateful inference context (the GNN rolling graph), so the next start
            # replays the entire stream from a clean slate.
            if self._scorer is not None:
                await self._scorer.reset()
            self._position = 0
            self.state = "idle"
            await self._broadcast_state()

    async def set_thresholds(self, review: float, fraud: float) -> None:
        if fraud < review + 0.02:
            fraud = review + 0.02
        review = max(0.0, min(1.0, review))
        fraud = max(0.0, min(1.0, fraud))
        self.thresholds = Thresholds(review=review, fraud=fraud)
        await self._broadcast(
            {
                "event": "thresholds",
                "data": {"review": review, "fraud": fraud},
            }
        )

    def status(self) -> dict:
        return {
            "state": self.state,
            "position": self._position,
            "thresholds": {
                "review": self.thresholds.review,
                "fraud": self.thresholds.fraud,
            },
            "metrics": self.metrics.to_dict(),
        }

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def _broadcast(self, msg: dict) -> None:
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.discard(q)

    async def _broadcast_state(self) -> None:
        await self._broadcast(
            {
                "event": "state",
                "data": {"state": self.state, "metrics": self.metrics.to_dict()},
            }
        )

    async def _run(self) -> None:
        ds = get_dataset()
        rows = list(ds.iter_rows())
        interval = STREAM_INTERVAL_MS / 1000

        try:
            while self._position < len(rows) and self.state != "stopped":
                await self._resume_event.wait()
                if self.state == "stopped":
                    break

                row = rows[self._position]
                self._position += 1

                assert self._scorer is not None
                try:
                    prob, latency_ms, rules_matched = await self._scorer.score(row)
                except Exception as exc:
                    log.exception("scorer error: %s", exc)
                    await asyncio.sleep(interval)
                    continue

                decision = _decide(prob, self.thresholds)
                is_fraud_gt = row.get("isFraud")
                amt = float(row["TransactionAmt"])
                gt_match = None

                m = self.metrics
                m.total += 1
                if decision == "FRAUD":
                    m.fraud += 1
                elif decision == "REVIEW":
                    m.review += 1
                else:
                    m.clean += 1
                m.latency_sum_ms += latency_ms
                m.latency_min_ms = latency_ms if m.latency_min_ms is None else min(m.latency_min_ms, latency_ms)
                m.latency_max_ms = latency_ms if m.latency_max_ms is None else max(m.latency_max_ms, latency_ms)

                # Quality + exposure (only when GT is known)
                if is_fraud_gt is not None:
                    gt_is_fraud = bool(is_fraud_gt)
                    decided_fraud = decision == "FRAUD"
                    gt_match = gt_is_fraud == decided_fraud

                    m.gt_total += 1
                    if gt_match:
                        m.correct += 1

                    if gt_is_fraud:
                        m.gt_fraud_total += 1
                        # Missed = GT fraud but we said NOT_FRAUD
                        # (REVIEW counts as "caught" since a human reviews it)
                        if decision == "NOT_FRAUD":
                            m.missed_fraud += 1
                            m.missed_fraud_usd += amt
                    else:
                        m.gt_clean_total += 1
                        if decision == "FRAUD":
                            m.false_positive += 1
                            m.false_positive_usd += amt

                txn = {
                    "transaction_id": int(row["TransactionID"]),
                    "transaction_dt": row.get("TransactionDT"),
                    "transaction_amt": amt,
                    "card_label": _card_label(row),
                    "score": round(prob, 4),
                    "decision": decision,
                    "latency_ms": latency_ms,
                    "is_fraud_gt": is_fraud_gt,
                    "ground_truth_match": gt_match,
                    "product": _product_label(row),
                    "card_type": _card_type_label(row),
                    "billing_region_code": row.get("addr1"),
                    "billing_country_code": row.get("addr2"),
                    "email_domain": _str_or_none(row.get("P_emaildomain")),
                    "device_type": _str_or_none(row.get("DeviceType")),
                    "device_info": _str_or_none(row.get("DeviceInfo")),
                    "explainability": (
                        {
                            "triggered_rules": rules_matched,
                            "xgboost_top_features": [],
                            "gnn_graph_risk_score": None,
                            "gnn_flagged_neighbors": None,
                        }
                        if rules_matched
                        else None
                    ),
                }

                await self._broadcast({"event": "transaction", "data": txn})
                await self._broadcast({"event": "metrics", "data": m.to_dict()})

                await asyncio.sleep(interval)

            if self._position >= len(rows):
                # Reached the end naturally — treat like a stop: rewind and clear
                # backend state so a replay starts from a clean graph.
                if self._scorer is not None:
                    await self._scorer.reset()
                self._position = 0
                self.state = "idle"
                await self._broadcast_state()
        except asyncio.CancelledError:
            raise


engine = StreamEngine()


async def sse_event_stream(q: asyncio.Queue) -> AsyncIterator[dict]:
    """Format queue messages as SSE events."""
    try:
        yield {
            "event": "state",
            "data": json.dumps(engine.status()),
        }
        while True:
            msg = await q.get()
            yield {
                "event": msg["event"],
                "data": json.dumps(msg["data"]),
            }
    except asyncio.CancelledError:
        return

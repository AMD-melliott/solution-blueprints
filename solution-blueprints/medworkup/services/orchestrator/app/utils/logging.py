# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Optional

# ── request-scoped correlation ID ─────────────────────────────────────────────
# Set this at the start of each /analyze request so every log line emitted
# during that request carries the same ID.
_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(rid: Optional[str] = None) -> str:
    """Set (or generate) a correlation ID for the current async context."""
    rid = rid or uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id.get()


# ── formatter that injects the correlation ID ─────────────────────────────────
class _RequestIdFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.request_id = _request_id.get()
        return super().format(record)


_FORMAT = "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s] – %(message)s"


# ── public helpers ────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger.
    Attaches a stdout handler the first time it is called for a given name.
    Subsequent calls return the cached logger unchanged (idempotent).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_RequestIdFormatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def configure_root(level: str) -> None:
    """
    Configure the root logger once at application startup.
    Should be called before any other logger is created.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_RequestIdFormatter(_FORMAT))
    root = logging.getLogger()
    root.setLevel(level.upper())
    # Remove any handlers added by basicConfig or third-party libs
    root.handlers.clear()
    root.addHandler(handler)

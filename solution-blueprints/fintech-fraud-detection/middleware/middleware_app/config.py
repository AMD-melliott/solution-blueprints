# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Centralized config. Reads from .env once at import time."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the middleware folder root
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

# Paths to the IEEE-CIS CSVs (relative to middleware folder root).
# Either the train_* or test_* split works; the identity file is optional (if the
# configured file is absent the loader simply skips the merge — see data_loader).
DEMO_CSV_PATH: str = os.getenv("DEMO_CSV_PATH", "data/test_transaction.csv")
IDENTITY_CSV_PATH: str = os.getenv("IDENTITY_CSV_PATH", "data/test_identity.csv")

# Backend orchestrator URL; empty = fake/demo scoring mode
WORKER_URL: str = os.getenv("WORKER_URL", "")

# How often to emit a new scored transaction (milliseconds)
STREAM_INTERVAL_MS: int = int(os.getenv("STREAM_INTERVAL_MS", "750"))

# Cap rows loaded into memory; 0 = no limit
MAX_DEMO_ROWS: int = int(os.getenv("MAX_DEMO_ROWS", "20000"))

# CORS: which origins can call this middleware
ALLOWED_ORIGINS: list[str] = [
    "http://localhost:5173",  # Vite dev
    "http://127.0.0.1:5173",
]

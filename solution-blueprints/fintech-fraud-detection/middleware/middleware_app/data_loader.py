# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Loads IEEE-CIS transaction + identity CSVs at startup, left-joins them on
TransactionID, and provides an iterator over the merged rows.
"""
import math
from pathlib import Path
from typing import Iterator

import pandas as pd
from app.config import DEMO_CSV_PATH, IDENTITY_CSV_PATH, MAX_DEMO_ROWS


def _resolve(csv_path: str) -> Path:
    path = Path(csv_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    return path


class DemoDataset:
    """Holds the merged transaction+identity data in memory."""

    def __init__(self, csv_path: str, identity_path: str | None = None) -> None:
        tx_path = _resolve(csv_path)
        if not tx_path.exists():
            raise FileNotFoundError(f"Transaction CSV not found at {tx_path}")
        df = pd.read_csv(tx_path, nrows=MAX_DEMO_ROWS if MAX_DEMO_ROWS > 0 else None)

        if identity_path:
            id_path = _resolve(identity_path)
            if id_path.exists():
                id_df = pd.read_csv(id_path)
                # Left join: every transaction is preserved; identity cols filled where available
                df = df.merge(id_df, on="TransactionID", how="left")
            else:
                # Identity data is optional; proceed with transactions only.
                pass

        # Replace NaN with None so JSON serialization works
        df = df.astype(object).where(pd.notnull(df), None)
        self._rows = df.to_dict(orient="records")
        for row in self._rows:
            for k, v in list(row.items()):
                if isinstance(v, float) and math.isnan(v):
                    row[k] = None

    @property
    def total_rows(self) -> int:
        return len(self._rows)

    def iter_rows(self) -> Iterator[dict]:
        """Yield rows in order. Caller decides when to stop."""
        for row in self._rows:
            yield row


# Module-level instance; populated in main.py lifespan
dataset: DemoDataset | None = None


def get_dataset() -> DemoDataset:
    if dataset is None:
        raise RuntimeError("Dataset not initialized")
    return dataset


def init_dataset() -> DemoDataset:
    global dataset
    dataset = DemoDataset(DEMO_CSV_PATH, IDENTITY_CSV_PATH)
    return dataset

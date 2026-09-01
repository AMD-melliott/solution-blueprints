#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
Train↔serve featurization parity check (the load-bearing invariant).

graph_builder.preprocess (batch/pandas, the FIT path) and RowFeaturizer
(numpy, per-row, the REPLAY path used by every inference service) must produce
identical feature vectors for the same transaction. This script proves it: it
reloads the raw CSVs the same way graph_builder does (load_raw → temporal_split,
which fixes the row order of graph.x), samples rows, runs each raw row through
RowFeaturizer, and compares against the corresponding row of the persisted
graph.x in model.pt.

Usage:
  python -m pipeline.featurize_parity                       # 2000 random rows
  python -m pipeline.featurize_parity -n 5000 --tol 1e-4
  python -m pipeline.featurize_parity -i artifacts/model.pt -c config.yaml

Exit code 0 = parity holds (within tolerance); 1 = drift detected.

Note: a couple of columns can legitimately differ on MISSING inputs only
(documented minor drift — e.g. a numeric freq key absent at serve → 0.0 vs the
median's frequency at train). Present-value rows should match to ~1e-5. The
report lists any offending feature columns so you can tell real drift from the
known-minor kind.
"""

import argparse
import os
import sys

import numpy as np
import torch
from pipeline.featurize import RowFeaturizer
from pipeline.graph_builder import load_config, load_raw, temporal_split


def _row_to_tx(row) -> dict:
    """Raw DataFrame row → transaction dict, NaN → None (as a JSON client sends)."""
    tx = {}
    for k, v in row.items():
        if isinstance(v, float) and not np.isfinite(v):
            continue  # missing → absent key (RowFeaturizer treats as None)
        tx[k] = v
    return tx


def main() -> int:
    ap = argparse.ArgumentParser(description="Train↔serve featurization parity check")
    ap.add_argument("--config", "-c", default="config.yaml")
    ap.add_argument("--input", "-i", default=None, help="model.pt bundle path")
    ap.add_argument("-n", type=int, default=2000, help="number of rows to sample")
    ap.add_argument("--tol", type=float, default=1e-4, help="max abs diff tolerance")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    inp = args.input or os.path.join(cfg["data"]["artifacts_dir"], "model.pt")
    print(f"Reading bundle : {inp}", file=sys.stderr)
    bundle = torch.load(inp, weights_only=False, map_location="cpu")

    feature_cols = bundle["feature_cols"]
    X = bundle["graph"].x.cpu().numpy()  # [n_nodes, n_feats], train-built features
    n_nodes = X.shape[0]

    # Rebuild the SAME row order graph.x has: graph_builder sorts by TransactionDT
    # and resets the index, so graph.x[i] is the i-th row of the sorted frame.
    print("Reloading raw CSVs (load_raw → temporal_split)…", file=sys.stderr)
    df = load_raw(cfg)
    df, _train_idx, _test_idx = temporal_split(df, cfg)
    assert len(df) == n_nodes, f"row count mismatch: df={len(df)} graph.x={n_nodes}"

    feat = RowFeaturizer.from_artifacts(bundle, cfg)

    rng = np.random.RandomState(args.seed)
    sample = rng.choice(n_nodes, size=min(args.n, n_nodes), replace=False)

    max_abs = 0.0
    n_bad_rows = 0
    col_maxdiff = np.zeros(len(feature_cols), dtype=np.float64)
    worst = []  # (diff, row_idx, col_name)

    for i in sample:
        tx = _row_to_tx(df.iloc[int(i)])
        got = feat.transform_row(tx)
        exp = X[int(i)]
        d = np.abs(got - exp)
        col_maxdiff = np.maximum(col_maxdiff, d)
        row_max = float(d.max())
        max_abs = max(max_abs, row_max)
        if row_max > args.tol:
            n_bad_rows += 1
            j = int(d.argmax())
            worst.append((row_max, int(i), feature_cols[j]))

    worst.sort(reverse=True)
    bad_cols = [(feature_cols[j], float(col_maxdiff[j])) for j in np.argsort(-col_maxdiff) if col_maxdiff[j] > args.tol]

    print("\n=== Featurization parity (train graph.x vs RowFeaturizer) ===", file=sys.stderr)
    print(f"rows sampled        : {len(sample):,} / {n_nodes:,}", file=sys.stderr)
    print(f"global max abs diff : {max_abs:.3e}  (tol {args.tol:.0e})", file=sys.stderr)
    print(f"rows over tolerance : {n_bad_rows}", file=sys.stderr)
    if bad_cols:
        print("feature cols over tolerance (col: max abs diff):", file=sys.stderr)
        for name, d in bad_cols[:20]:
            print(f"  {name:<22} {d:.3e}", file=sys.stderr)
    if worst:
        print("worst rows:", file=sys.stderr)
        for d, ridx, col in worst[:10]:
            print(f"  row {ridx:<8} {col:<22} diff={d:.3e}", file=sys.stderr)

    if n_bad_rows == 0:
        print("\nPASS — featurization parity holds.", file=sys.stderr)
        return 0
    print(
        f"\nWARN — {n_bad_rows} row(s) exceed tol. If the only offenders are "
        "numeric freq cols (card*/addr*) on missing inputs, that's the documented "
        "minor drift; anything else is real train/serve divergence to fix.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

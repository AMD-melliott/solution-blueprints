#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
Build a PyG transaction graph from raw IEEE-CIS CSV files.

Performs the full preprocessing pipeline (imputation, encoding, feature
engineering) with all statistics fit on the temporal train split only,
then constructs the graph.

Output bundle keys:
  graph        – torch_geometric.data.Data (x, y, edge_index)
  train_idx    – np.int64 array, first (1-test_frac)% by TransactionDT
  test_idx     – np.int64 array, last test_frac% by TransactionDT
  feature_cols – list[str], node feature column names
  config       – config dict snapshot

Usage:
  python -m pipeline.graph_builder                      # write to artifacts/graph.pt
  python -m pipeline.graph_builder -o path/to/out.pt
  python -m pipeline.graph_builder -o - | python -m pipeline.gnn_train -i -
"""

import argparse
import gc
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _read_csv(path: str, **kwargs) -> pd.DataFrame:
    size = os.path.getsize(path)
    chunks = []
    with open(path, "rb") as f:
        with tqdm(
            total=size, desc=os.path.basename(path), unit="B", unit_scale=True, unit_divisor=1024, file=sys.stderr
        ) as pbar:
            pos = 0
            for chunk in pd.read_csv(f, chunksize=50_000, **kwargs):
                chunks.append(chunk)
                new_pos = f.tell()
                pbar.update(new_pos - pos)
                pos = new_pos
    return pd.concat(chunks, ignore_index=True)


def load_raw(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]["data_dir"]
    df = pd.merge(
        _read_csv(os.path.join(d, "train_transaction.csv")),
        _read_csv(os.path.join(d, "train_identity.csv")),
        on="TransactionID",
        how="left",
    )
    print(f"Loaded  : {df.shape}  fraud={df['isFraud'].mean():.4%}", file=sys.stderr)
    return df


def temporal_split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    sort_col = cfg["split"]["sort_col"]
    test_frac = cfg["split"]["test_fraction"]

    df = df.sort_values(sort_col).reset_index(drop=True)
    n_train = int(len(df) * (1.0 - test_frac))
    train_idx = np.arange(n_train, dtype=np.int64)
    test_idx = np.arange(n_train, len(df), dtype=np.int64)

    cutoff_dt = df.iloc[n_train][sort_col]
    print(
        f"Split   : train={n_train:,}  test={len(df)-n_train:,}" f"  cutoff {sort_col}={cutoff_dt:.0f}",
        file=sys.stderr,
    )
    return df, train_idx, test_idx


def preprocess(df: pd.DataFrame, train_idx: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    TARGET = "isFraud"
    M_COLS = [f"M{i}" for i in range(1, 10)]

    # Classify M columns up front. Most (M1-M3, M5-M9) are binary "T"/"F", but
    # M4 ∈ {M0,M1,M2} is genuinely categorical — force-mapping it through
    # {"T":1,"F":0} turns every value into NaN→-1.0 (a dead constant column).
    # Decide data-drivenly (values ⊆ {T,F} → binary); binary M's get the
    # dedicated T/F→1.0/0.0, missing→-1.0 mapping below and are kept out of
    # cat_cols/num_cols, while categorical M's are LEFT in so they ride the normal
    # label-encoding path like any other categorical.
    m_binary_cols = [c for c in M_COLS if c in df.columns and set(df[c].dropna().unique()) <= {"T", "F"}]
    DROP_COLS = {"TransactionID", TARGET, *m_binary_cols}

    # Snapshot the RAW edge-key values BEFORE any imputation / encoding / dropping,
    # so edges connect transactions sharing the same entity, NaNs drop out (missing
    # key → no edge), and train matches serve. A raw key like card1 is dropped as a
    # feature further down, so it must be captured here. 'uid' is a *composite*
    # identity key (config.graph.uid_cols, e.g. card1_addr1_D1), not a raw column —
    # build it here from raw components (NaN if ANY is missing). Each component is
    # canonicalized via str(float(v)) so the key is dtype-independent: card1 loads
    # as int64 (→ astype(str) would give "13926", not "13926.0") while addr1/D1 are
    # float — float() forces "13926.0" for all of them, byte-identical to the
    # serving side's str(float(p)) (app/predictor.py _edge_key_vals). Keep the two
    # in sync: both MUST use str(float(...)).
    uid_cols = cfg["graph"].get("uid_cols", [])
    _ek: dict = {}
    for c in cfg["graph"].get("edge_cols", []):
        if c == "uid" and uid_cols and all(u in df.columns for u in uid_cols):
            valid = df[uid_cols].notna().all(axis=1)
            s = df[uid_cols[0]].map(lambda v: str(float(v)))
            for p in uid_cols[1:]:
                s = s + "_" + df[p].map(lambda v: str(float(v)))
            _ek[c] = s.where(valid)
        elif c in df.columns:
            _ek[c] = df[c]
    edge_keys_df = pd.DataFrame(_ek, index=df.index).copy()

    cat_cols = df.select_dtypes(include=["object", "string"]).columns.difference(list(DROP_COLS)).tolist()
    num_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns.difference(list(DROP_COLS)).tolist()
        if c != "TransactionDT"
    ]

    train_medians: dict[str, float] = {}
    for c in tqdm(num_cols, desc="Computing medians", unit="col", file=sys.stderr):
        m = df.iloc[train_idx][c].median()
        train_medians[c] = float(m) if pd.notna(m) else 0.0
    for c in tqdm(num_cols, desc="Imputing numerics", unit="col", file=sys.stderr):
        df[c] = df[c].fillna(train_medians[c])
    for c in tqdm(cat_cols, desc="Imputing categoricals", unit="col", file=sys.stderr):
        df[c] = df[c].fillna("Unknown")

    # Binary M columns: "T"→1.0, "F"→0.0, missing→-1.0. Categorical M's (e.g. M4)
    # are left untouched here — they were kept in cat_cols above and get
    # label-encoded with everything else below. `m_cols` is persisted so the
    # serving side applies the identical binary mapping (app/predictor._featurize).
    # TODO: do the same with id34-id38 as they are also binary?
    m_cols = list(m_binary_cols)
    for c in m_cols:
        df[c] = df[c].map({"T": 1.0, "F": 0.0}).fillna(-1.0).astype(np.float32)
    print(f"Processed M cols: binary={m_cols}", file=sys.stderr)

    # Label-encode categoricals on TRAIN rows only — no test leak (mirrors how
    # medians / freq / card-stats are all fit on train_idx). Always reserve an
    # "Unknown" code so categories unseen at train (test rows here, or novel values
    # at serve) map to it instead of colliding with whatever real class got index 0.
    # cat_cols were already imputed NaN→"Unknown" above; the serving side mirrors
    # this via label_map.get(key, label_map["Unknown"]).
    label_maps: dict[str, dict] = {}
    for c in tqdm(cat_cols, desc="Label-encoding", unit="col", file=sys.stderr):
        classes = set(df.iloc[train_idx][c].astype(str).unique())
        classes.add("Unknown")
        label_map = {cls: i for i, cls in enumerate(sorted(classes))}
        label_maps[c] = label_map
        df[c] = df[c].astype(str).map(label_map).fillna(label_map["Unknown"]).astype(np.int64)

    card_amt_stats: dict[int, dict] = {}
    if "card1" in df.columns:
        cs = df.iloc[train_idx].groupby("card1")["TransactionAmt"].agg(["mean", "std"])
        card_amt_stats = {
            int(k): {
                "mean": float(row["mean"]),
                "std": float(row["std"]) if pd.notna(row["std"]) else 0.0,
            }
            for k, row in cs.iterrows()
        }
        cs = cs.rename(columns={"mean": "_cmu", "std": "_csd"})
        df = df.join(cs, on="card1")
        # Standardize per card. Cards with zero/undefined variance (single tx, or
        # all-identical amounts) have NO meaningful z-score → 0.0, NOT (amt-mean)/1e-6
        # which detonates to ~1e6 the moment a live amount differs. Serving mirrors
        # this (predictor: sd>0 ? (amt-mu)/sd : 0.0). The gross-anomaly case is
        # covered by the rules engine (R010), not this feature.
        df["amt_zscore"] = (df["TransactionAmt"] - df["_cmu"]) / df["_csd"]
        df["amt_zscore"] = df["amt_zscore"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df = df.drop(columns=["_cmu", "_csd"])

    prod_risk_map: dict[int, float] = {}
    prod_risk_default: float = 0.0
    if "ProductCD" in df.columns:
        prod_risk = df.iloc[train_idx].groupby("ProductCD")[TARGET].mean().rename("product_risk_score")
        prod_risk_default = float(prod_risk.median()) if len(prod_risk) else 0.0
        prod_risk_map = {int(k): float(v) for k, v in prod_risk.items()}
        df = df.join(prod_risk, on="ProductCD")
        df["product_risk_score"] = df["product_risk_score"].fillna(prod_risk_default)

    dt = df["TransactionDT"]
    df["tx_hour"] = (dt // 3600) % 24
    df["tx_day_week"] = (dt // 86400) % 7
    df["tx_day"] = dt // 86400
    if "D1" in df.columns:
        df["D1_norm"] = df["D1"] - df["tx_day"]

    freq_encode_cols = cfg["graph"].get("freq_encode_cols", [])
    freq_maps: dict[str, dict] = {}
    count_maps: dict[str, dict] = {}
    for c in tqdm(freq_encode_cols, desc="Freq-encoding", unit="col", file=sys.stderr):
        if c not in df.columns:
            continue
        freq = df.iloc[train_idx][c].value_counts(normalize=True)
        count = df.iloc[train_idx][c].value_counts()
        freq_maps[c] = freq.to_dict()
        count_maps[c] = count.to_dict()
        df[f"{c}_freq"] = df[c].map(freq).fillna(0.0).astype(np.float32)
        df[f"{c}_log_freq"] = np.log1p(df[c].map(count).fillna(0)).astype(np.float32)

    drop_raw = [c for c in ["card1", "card2", "card3", "card5", "addr1", "addr2"] if c in df.columns]
    df = df.drop(columns=drop_raw)

    print(f"Prepared: {df.shape}", file=sys.stderr)
    preprocess_artifacts = {
        "label_maps": label_maps,
        "m_cols": m_cols,
        "freq_maps": freq_maps,
        "count_maps": count_maps,
        "train_medians": train_medians,
        "card_stats": card_amt_stats,
        "prod_risk": prod_risk_map,
        "prod_risk_default": prod_risk_default,
    }
    return df, preprocess_artifacts, edge_keys_df


def build_graph(
    df: pd.DataFrame, edge_keys_df: pd.DataFrame, train_idx: np.ndarray, cfg: dict
) -> tuple[Data, list[str], dict]:
    gcfg = cfg["graph"]
    # Edges are built on the raw entity keys captured in preprocess (card1,
    # P_emaildomain, ...), NOT on df's columns — card1 has already been dropped
    # from df and df's email columns are imputed/encoded. edge_keys_df holds the
    # raw pre-imputation values, so grouping matches the serving side.
    edge_cols = list(edge_keys_df.columns)
    max_group = gcfg["max_group_size"]
    max_edges = gcfg["max_edges"]
    exclude = set(gcfg["exclude_node_cols"])

    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feat_cols].values.astype(np.float32)

    mu = np.nanmean(X[train_idx], axis=0)
    sig = np.nanstd(X[train_idx], axis=0)
    # Floor the scale at 1.0 for (near-)constant columns instead of the old
    # `std + 1e-6`: dividing a live deviation by ~1e-6 produced ~1e6 spikes. With a
    # 1.0 floor a constant col stays ~0 on train and degrades gracefully off it.
    sig = np.where(sig > 1e-6, sig, 1.0)
    clip = float(gcfg.get("norm_clip", 10.0))
    norm_stats = {
        "norm_mu": mu.astype(np.float32),
        "norm_sig": sig.astype(np.float32),
        "norm_clip": clip,
    }
    X = np.nan_to_num((X - mu) / sig, nan=0.0, posinf=0.0, neginf=0.0)
    if clip and clip > 0:
        # Final safety net: bound any residual extreme z so no single OOD feature
        # dominates the GNN's first layer. Serving applies the identical clip.
        X = np.clip(X, -clip, clip)

    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(df["isFraud"].values.astype(np.int64), dtype=torch.long)

    fraud_n = int((y == 1).sum())
    print(
        f"Nodes   : {x.shape[0]:,}  features={x.shape[1]}" f"  fraud={fraud_n:,} ({fraud_n/len(y):.2%})",
        file=sys.stderr,
    )

    src: list[int] = []
    dst: list[int] = []
    for col in edge_cols:
        n_before = len(src)
        n_groups = edge_keys_df[col].nunique()
        with tqdm(
            edge_keys_df.groupby(col, sort=False), total=n_groups, desc=f"Edges [{col}]", unit="group", file=sys.stderr
        ) as pbar:
            for _, grp in pbar:
                if len(src) >= max_edges:
                    break
                idxs = grp.index.tolist()
                if len(idxs) < 2:
                    continue
                if "TransactionDT" in df.columns:
                    idxs = df.loc[idxs, "TransactionDT"].sort_values().index.tolist()
                if len(idxs) > max_group:
                    # keep the MOST-RECENT max_group (idxs are sorted ascending by
                    # time): covers the recent/test transactions we actually score,
                    # and bounds noise from junk hub keys (e.g. placeholder card1
                    # values with thousands of txns). Earliest-N would instead drop
                    # exactly the recent rows.
                    idxs = idxs[-max_group:]
                # directed earlier→later: src=a (earlier), dst=b (later), so the
                # later transaction aggregates the earlier one. Message passing
                # never flows backward in time → no future leakage.
                for a, b in zip(idxs[:-1], idxs[1:]):
                    src.append(a)
                    dst.append(b)
                pbar.set_postfix(edges=len(src))
        n_added = len(src) - n_before
        print(f"  {col}: +{n_added:,} edges", file=sys.stderr)
        if len(src) >= max_edges:
            print("  max_edges cap reached", file=sys.stderr)
            break

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    del src, dst
    gc.collect()

    print(
        f"Edges   : {edge_index.shape[1]:,} directed (earlier→later)",
        file=sys.stderr,
    )

    return Data(x=x, y=y, edge_index=edge_index), feat_cols, norm_stats


def write_bundle(bundle: dict, out: str) -> None:
    if out == "-":
        buf = io.BytesIO()
        torch.save(bundle, buf)
        sys.stdout.buffer.write(buf.getvalue())
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        torch.save(bundle, out)
        print(f"Saved   → {out}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build transaction graph")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--output", "-o", default=None, help='output path, or "-" for stdout')
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["training"]["seed"])

    df = load_raw(cfg)
    df, train_idx, test_idx = temporal_split(df, cfg)
    df, preprocess_arts, edge_keys_df = preprocess(df, train_idx, cfg)
    graph_data, feat_cols, norm_arts = build_graph(df, edge_keys_df, train_idx, cfg)

    train_set = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
    train_set[torch.tensor(train_idx, dtype=torch.long)] = True
    ei = graph_data.edge_index  # directed, earlier→later
    s_train = train_set[ei[0]]  # source (earlier) is a train node
    d_train = train_set[ei[1]]  # dest   (later)  is a train node
    # Training graph: train→train only — a train node aggregates earlier TRAIN
    # neighbours (no test features leak in; directedness also blocks future leak).
    graph_data.edge_index = ei[:, s_train & d_train]
    # Inductive graph (eval / embedding export / inductive inference): the FULL
    # time-forward set, so a TEST node aggregates its earlier neighbours —
    # train→test AND test→test. The temporal split guarantees no test→train edge
    # exists, so train embeddings match the training regime and nothing flows
    # from the future.
    inductive_edge_index = ei
    n_tt = int((s_train & d_train).sum())
    n_te = int((s_train & ~d_train).sum())  # train→test
    n_ee = int((~s_train & ~d_train).sum())  # test→test (kept, time-forward)
    n_et = int((~s_train & d_train).sum())  # test→train (should be 0)
    print(
        f"Edges   : directed earlier→later — {n_tt:,} train→train  "
        f"{n_te:,} train→test  {n_ee:,} test→test  (test→train={n_et:,}, expect 0)",
        file=sys.stderr,
    )

    bundle = {
        "graph": graph_data,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "feature_cols": feat_cols,
        "config": cfg,
        "transaction_ids": df["TransactionID"].values.copy() if "TransactionID" in df.columns else None,
        "transaction_dts": df["TransactionDT"].values.copy() if "TransactionDT" in df.columns else None,
        "inductive_edge_index": inductive_edge_index,
        **preprocess_arts,
        **norm_arts,
    }

    out = args.output or os.path.join(cfg["data"]["artifacts_dir"], "graph.pt")
    write_bundle(bundle, out)


if __name__ == "__main__":
    main()

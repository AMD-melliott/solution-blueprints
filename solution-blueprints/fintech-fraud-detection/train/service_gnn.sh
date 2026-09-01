#!/usr/bin/env bash

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
# GNN training pipeline service.
# Runs: graph building → GNN training → embedding extraction → GNN eval → GNN viz
#
# Environment variables:
#   CONFIG        path to config file            (default: config.yaml)
#   LOG_DIR       directory for step logs        (default: logs)
#   SKIP_GRAPH    set to 1 to reuse graph.pt     (default: 0)
#   SKIP_TRAIN    set to 1 to reuse model.pt     (default: 0)
#   SKIP_VIZ      set to 1 to skip plots         (default: 1)

set -euo pipefail

CONFIG="${CONFIG:-config.yaml}"
LOG_DIR="${LOG_DIR:-logs}"
SKIP_GRAPH="${SKIP_GRAPH:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_VIZ="${SKIP_VIZ:-1}"

ARTIFACTS_DIR=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['data']['artifacts_dir'])" 2>/dev/null || echo "artifacts")
GRAPH_PT="$ARTIFACTS_DIR/graph.pt"
MODEL_PT="$ARTIFACTS_DIR/model.pt"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }
hr()  { echo "────────────────────────────────────────"; }

mkdir -p "$LOG_DIR" "$ARTIFACTS_DIR"

log "=== GNN Training Pipeline ==="
log "Config     : $CONFIG"
log "Artifacts  : $ARTIFACTS_DIR"
hr

# Step 1 — Graph
if [[ "$SKIP_GRAPH" == "1" && -f "$GRAPH_PT" ]]; then
    log "SKIP_GRAPH=1 — reusing $GRAPH_PT"
else
    log "Step 1/4 — Building graph..."
    python -m pipeline.graph_builder -c "$CONFIG" 2>&1 | tee "$LOG_DIR/graph_builder.log"
    [[ -f "$GRAPH_PT" ]] || die "graph_builder did not produce $GRAPH_PT"
fi
hr

# Step 2 — GNN training
if [[ "$SKIP_TRAIN" == "1" && -f "$MODEL_PT" ]]; then
    log "SKIP_TRAIN=1 — reusing $MODEL_PT"
else
    log "Step 2/4 — Training GNN (also exports gnn.onnx)..."
    python -m pipeline.gnn_train -c "$CONFIG" -i "$GRAPH_PT" 2>&1 | tee "$LOG_DIR/gnn_train.log"
    [[ -f "$MODEL_PT" ]] || die "gnn_train did not produce $MODEL_PT"
fi
hr

# Step 3 — GNN evaluation
log "Step 3/4 — Evaluating GNN..."
python -m pipeline.gnn_eval -c "$CONFIG" -i "$MODEL_PT" 2>&1 | tee "$LOG_DIR/gnn_eval.log"
hr

# Step 4 — Visualization
if [[ "$SKIP_VIZ" == "1" ]]; then
    log "SKIP_VIZ=1 — skipping GNN plots"
else
    log "Step 4/4 — Generating GNN plots..."
    python -m pipeline.gnn_visualize -c "$CONFIG" -i "$MODEL_PT" 2>&1 | tee "$LOG_DIR/gnn_viz.log"
fi
hr

log "=== GNN pipeline complete. Artifacts ready for XGBoost training. ==="

#!/usr/bin/env bash

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
# XGBoost training pipeline service.
# Requires: artifacts/model.pt already produced by GNN training.
# Runs: XGBoost training → XGBoost eval → XGBoost viz
#
# Environment variables:
#   CONFIG      path to config file        (default: config.yaml)
#   LOG_DIR     directory for step logs    (default: logs)
#   SKIP_VIZ    set to 1 to skip plots     (default: 1)

set -euo pipefail

CONFIG="${CONFIG:-config.yaml}"
LOG_DIR="${LOG_DIR:-logs}"
SKIP_VIZ="${SKIP_VIZ:-1}"

ARTIFACTS_DIR=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['data']['artifacts_dir'])" 2>/dev/null || echo "artifacts")
MODEL_PT="$ARTIFACTS_DIR/model.pt"
XGB_MODEL="$ARTIFACTS_DIR/xgb_model.joblib"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }
hr()  { echo "────────────────────────────────────────"; }

mkdir -p "$LOG_DIR" "$ARTIFACTS_DIR"

log "=== XGBoost Training Pipeline ==="
log "Config     : $CONFIG"
log "Artifacts  : $ARTIFACTS_DIR"
hr

[[ -f "$MODEL_PT" ]] || die "GNN bundle not found at $MODEL_PT — run GNN training first."

# Step 1 — XGBoost training
log "Step 1/3 — Training XGBoost..."
python -m pipeline.xgb_train -c "$CONFIG" -i "$MODEL_PT" 2>&1 | tee "$LOG_DIR/xgb_train.log"
[[ -f "$XGB_MODEL" ]] || die "xgb_train did not produce $XGB_MODEL"
hr

# Step 2 — XGBoost evaluation
log "Step 2/3 — Evaluating XGBoost..."
python -m pipeline.xgb_eval -c "$CONFIG" -i "$MODEL_PT" 2>&1 | tee "$LOG_DIR/xgb_eval.log"
hr

# Step 3 — Visualization
if [[ "$SKIP_VIZ" == "1" ]]; then
    log "SKIP_VIZ=1 — skipping XGBoost plots"
else
    log "Step 3/3 — Generating XGBoost plots..."
    python -m pipeline.xgb_visualize -c "$CONFIG" -i "$MODEL_PT" 2>&1 | tee "$LOG_DIR/xgb_viz.log"
fi
hr

log "=== XGBoost pipeline complete. Artifacts ready for inference. ==="

#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data02/home/scv7eyk/cross-model/VM-DMD}"
# Edit this to the completed experiment directory containing best.pt and config.json.
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/runs/isic18/CHANGE_TO_YOUR_RUN_DIRECTORY}"
SPLIT="${SPLIT:-val}" # val for analysis; test only for final evaluation
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/metrics_${SPLIT}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
cd "$PROJECT_DIR"
python scripts/evaluate_metrics.py --run-dir "$RUN_DIR" --split "$SPLIT" --output-dir "$OUTPUT_DIR"

#!/bin/bash
# run_zpm1_sweep.sh — Local z=±1 6F5 delta proxy sweep
#
# Usage:
#   ./run_zpm1_sweep.sh [TOTAL] [WORKERS]
#   ./run_zpm1_sweep.sh 1000000 8

set -euo pipefail

TOTAL="${1:-1000000}"
WORKERS="${2:-8}"
SEED="${SEED:-20260717}"
THRESH="${THRESH:-0.0}"
OUT="survivors_zpm1_${SEED}.jsonl"

cd "$(dirname "$0")"

echo "=== z=+1/-1 6F5 sweep: ${TOTAL} trajectories, ${WORKERS} workers ==="
python3 sweep_zpm1.py \
    --total "${TOTAL}" \
    --batch 10000 \
    --backend numpy \
    --thresh "${THRESH}" \
    --seed "${SEED}" \
    --workers "${WORKERS}" \
    --out "${OUT}"

echo ""
echo "Survivors:"
wc -l "${OUT}" 2>/dev/null || echo "0"
echo ""
echo "Stage 2: python3 pslq_companion.py --in ${OUT} --dim 6"

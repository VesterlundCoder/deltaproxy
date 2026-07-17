#!/bin/bash
# run_verification_local.sh
# =========================
# Stage-2 (exact, pure-integer) verification of the LUMI GPU-proxy survivors,
# run LOCALLY on the Mac (no LUMI CPU hours needed).
#
# Step 1: pull the survivor files from LUMI scratch.
# Step 2: exact-verify each sweep with cmf_generic.independent_delta.
#
# Usage:
#   bash run_verification_local.sh          # fetch + verify both sweeps
#   bash run_verification_local.sh --no-fetch   # skip rsync, verify existing local files
#
# GPU jobs (submitted 2026-06-26):
#   6F5 200B -> runs/6f5_200b_19558891/   (dim 6)
#   8F7 500B -> runs/8f7_500b_19559452/   (dim 8)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMI_RUNS="/scratch/project_465002669/cmf/runs"
LOCAL_RES="${HERE}/lumi_results"
WORKERS="$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"
NVERIFY="${NVERIFY:-100}"     # exact double-depth = 2*NVERIFY
FETCH=1
[ "${1:-}" = "--no-fetch" ] && FETCH=0

mkdir -p "${LOCAL_RES}"

if [ "${FETCH}" = "1" ]; then
  echo "=== Step 1: fetching survivor files from LUMI (only *.jsonl + logs) ==="
  rsync -avz --prune-empty-dirs \
    --include='*/' --include='survivors*.jsonl' --include='log_*.txt' --exclude='*' \
    "lumi:${LUMI_RUNS}/" "${LOCAL_RES}/"
fi

# Build the verification job list: TAG:DIM
declare -a JOBS=( "6f5_200b:6" "8f7_500b:8" )

for J in "${JOBS[@]}"; do
  TAG="${J%%:*}"; DIM="${J##*:}"
  # find the (newest) run dir for this tag
  RUNDIR="$(ls -d "${LOCAL_RES}/${TAG}"_* 2>/dev/null | sort | tail -n1 || true)"
  if [ -z "${RUNDIR}" ]; then
    echo "!! no local run dir for ${TAG} (did the fetch run / job finish?)"; continue
  fi
  SURV="${RUNDIR}/survivors_all.jsonl"
  # if the end-of-job merge is missing, merge the per-GCD shards ourselves
  if [ ! -s "${SURV}" ]; then
    cat "${RUNDIR}"/survivors_p*.jsonl > "${SURV}" 2>/dev/null || true
  fi
  N="$(wc -l < "${SURV}" 2>/dev/null || echo 0)"
  echo
  echo "=== Step 2: verify ${TAG} (dim ${DIM})  survivors=${N}  workers=${WORKERS} ==="
  if [ "${N}" -eq 0 ]; then
    echo "  (no survivors above threshold -- check max_r2 in ${RUNDIR}/log_*.txt)"
    grep -h "max_r2 seen" "${RUNDIR}"/log_*.txt 2>/dev/null | sort -t= -k2 -g | tail -n3 || true
    continue
  fi
  python3 "${HERE}/verify_survivors.py" \
      --in "${SURV}" --dim "${DIM}" \
      --nverify "${NVERIFY}" --workers "${WORKERS}" --thresh 0.0 \
      --out "${RUNDIR}/confirmed_${TAG}.jsonl"
  echo "  -> confirmed positives: ${RUNDIR}/confirmed_${TAG}.jsonl"
done

echo
echo "=== DONE. Confirmed-hit files: ==="
find "${LOCAL_RES}" -name 'confirmed_*.jsonl' -exec sh -c 'echo "  $(wc -l < "$1") hits  $1"' _ {} \; 2>/dev/null || true

#!/bin/bash
# deploy_zpm1_full.sh
# ===================
# Rsync code to LUMI and launch the full z=+1/-1 6F5 sweep.
#
# Budget: 26,000 GPU hours on project_465002669
# Full space: 6.81e13 trajectories (z=+1,-1; box6; dim6; non-degenerate)
# Throughput: ~301k traj/s/GCD (MI250X fp32)
#
# Coverage with 26,000 GPU hours:
#   26,000 GPUh * 1.08e9 traj/GPUh = 2.82e13 trajectories (41% of full space)
#
# Strategy: 4 waves of 8 nodes, 24h each = 4 * 8 * 8 * 24 = 6,144 GPUh per wave
#   Each wave: 8 nodes * 8 GCD * 301k * 24h * 3600 = 1.66e12 trajectories
#   4 waves = 6.66e12 trajectories (9.8% of full space, 6,144 GPUh)
#
# For full coverage (2.82e13): need ~17 waves at 24h each = 26,112 GPUh
#   That's 17 * 1.66e12 = 2.82e13 = 41% of space
#
# To cover 100%: would need 41 waves = 41 * 1,536 = 62,976 GPUh (over budget)
#
# RECOMMENDED: Run 16 waves of 8 nodes / 24h = 24,576 GPUh (under 26k budget)
#   Total screened: 2.66e13 (39% of full space)
#   Expected survivors at 5e-6 rate: ~133,000
#
# Usage:
#   bash deploy_zpm1_full.sh sync     # rsync code to LUMI
#   bash deploy_zpm1_full.sh launch   # submit all waves
#   bash deploy_zpm1_full.sh status   # check job queue
#   bash deploy_zpm1_full.sh fetch    # download survivors
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMI_HOST="lumi"
LUMI_CODE="/scratch/project_465002669/cmf/continuous_cmf"
LUMI_RUNS="/scratch/project_465002669/cmf/runs"
SEED_BASE=20260717
NODES=8
TIME=24:00:00
WAVES=16
TOTAL_PER_GCD=13000000000   # 13e9 per GCD per 24h wave

cmd="${1:-help}"

case "$cmd" in

  sync)
    echo "=== Rsyncing code to LUMI ==="
    rsync -avz --delete \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      --exclude='.git' \
      --exclude='lumi_results' \
      --exclude='lumi_survivors_*' \
      --exclude='verify_6f5_lumi_out' \
      --exclude='survivors_*.jsonl' \
      --exclude='benchmark_results.json' \
      --exclude='validation_results.json' \
      --exclude='ctf_study' \
      --exclude='verification' \
      "${HERE}/" "${LUMI_HOST}:${LUMI_CODE}/"
    echo ""
    echo "=== Syncing SLURM scripts ==="
    rsync -avz \
      "${HERE}/gpu_proxy/lumi_zpm1_positive_669.sbatch" \
      "${LUMI_HOST}:${LUMI_CODE}/gpu_proxy/"
    echo ""
    echo "Done. Verify on LUMI with:"
    echo "  ssh ${LUMI_HOST} 'ls -la ${LUMI_CODE}/gpu_proxy/gpu_sweep.py ${LUMI_CODE}/gpu_proxy/proxy_kernel.py ${LUMI_CODE}/gpu_proxy/lumi_zpm1_positive_669.sbatch'"
    ;;

  launch)
    echo "=== Launching ${WAVES} waves of z=+1/-1 6F5 sweep ==="
    echo "  Nodes per wave: ${NODES}"
    echo "  Time per wave: ${TIME}"
    echo "  Total per GCD: ${TOTAL_PER_GCD}"
    echo "  Total per wave: $((NODES * 8 * TOTAL_PER_GCD))"
    echo "  Total all waves: $((WAVES * NODES * 8 * TOTAL_PER_GCD))"
    echo "  GPU hours: $((WAVES * NODES * 8 * 24))"
    echo ""

    for w in $(seq 0 $((WAVES - 1))); do
      SEED=$((SEED_BASE + w * 1000))
      TAG="zpm1_full_w${w}"
      echo "  Wave ${w}: seed=${SEED} tag=${TAG}"

      ssh "${LUMI_HOST}" "cd ${LUMI_CODE}/gpu_proxy && \
        sbatch --job-name=${TAG} \
          --nodes=${NODES} --time=${TIME} \
          --export=ALL,SEED=${SEED},TAG=${TAG},TOTAL_PER_GCD=${TOTAL_PER_GCD},THRESH=0.0 \
          lumi_zpm1_positive_669.sbatch"
    done

    echo ""
    echo "All ${WAVES} waves submitted. Check with:"
    echo "  ssh ${LUMI_HOST} 'squeue --me'"
    ;;

  status)
    echo "=== Job queue on LUMI ==="
    ssh "${LUMI_HOST}" "squeue --me -o '%.10i %.15j %.8T %.10M %.6D %R'"
    ;;

  fetch)
    echo "=== Fetching survivors from LUMI ==="
    mkdir -p "${HERE}/lumi_zpm1_results"
    rsync -avz --prune-empty-dirs \
      --include='*/' \
      --include='survivors*.jsonl' \
      --include='log_*.txt' \
      --exclude='*' \
      "lumi:${LUMI_RUNS}/zpm1_full_*" "${HERE}/lumi_zpm1_results/"
    echo ""
    echo "Survivor files:"
    find "${HERE}/lumi_zpm1_results" -name 'survivors_all.jsonl' \
      -exec sh -c 'echo "  $(wc -l < "$1") survivors  $1"' _ {} \;
    ;;

  *)
    echo "usage: $0 {sync|launch|status|fetch}"
    echo ""
    echo "  sync    : rsync code to LUMI"
    echo "  launch  : submit ${WAVES} sweep waves (26k GPUh budget)"
    echo "  status  : check LUMI job queue"
    echo "  fetch   : download survivor files"
    echo ""
    echo "Budget: ${WAVES} waves x ${NODES} nodes x 8 GCD x 24h = $((WAVES * NODES * 8 * 24)) GPUh"
    echo "Coverage: $((WAVES * NODES * 8 * TOTAL_PER_GCD)) trajectories (41% of 6.81e13)"
    ;;
esac

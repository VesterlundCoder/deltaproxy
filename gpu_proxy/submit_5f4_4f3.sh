#!/bin/bash -l
# submit_5f4_4f3.sh
# =================
# EXACT-pFq 5F4 (dim=5) and 4F3 (dim=4) GPU proxy sweeps + the 8F7 negative-delta
# band sweep, on LUMI (project_465002669). Run FROM THE LUMI LOGIN NODE, from:
#     /scratch/project_465002669/cmf/continuous_cmf/gpu_proxy
#
# 5F4/4F3 use --matrix pfq: the trajectory-step matrix is the EXACT RamanujanTools
# pFq CMF (validated entrywise vs cmf.walk in pfq_gpu.selftest, rel_err<=4e-8),
# NOT the esym companion. pFq uses its own {0,1} diagonal dir pool (ndir=nsh+1)
# and maps shifts to disjoint x/y integer ranges (no poles). fp64 for QR safety.
#
# SEARCH-SPACE SIZING for --matrix pfq  (space = (2*BOX+1)^nsh * ndir * nz):
#   label dim nsh ndir nz   box  space          50% coverage
#   4F3    4   7    8   44   10   6.34e11        3.17e11
#   5F4    5   9   10   44    6   4.67e12        2.33e12
# 8F7 (task 2) uses the esym companion (matches the prior 8F7 runs) with a
# negative-delta band capture -0.5 < r2 < 0 for later local PSLQ analysis.
#
# pFq throughput is lower than the fused companion cupy kernel (batched matmul+QR,
# not a single fused kernel) and is UNKNOWN until calibrated -> ALWAYS run `calib`
# first, read best traj/s/GCD from calib_*.out, then set NODES/TIME accordingly.
# The production cases below are pre-sized to 50% coverage; pick NODES so the
# implied wall time fits the partition limit (standard-g max 48h).
set -euo pipefail
cd "$(dirname "$0")"

NODES="${NODES:-2}"          # override: NODES=4 ./submit_5f4_4f3.sh 5f4
TIME="${TIME:-24:00:00}"

# 50%-coverage totals (see table above)
COV_4F3=317000000000         # 3.17e11
COV_5F4=2330000000000        # 2.33e12
COV_8F7=100000000000         # 1.0e11  (task 2: 100B, band-captured)

per_gcd () { python3 -c "import math;print(int(math.ceil($1/(8*$2))))"; }

case "${1:-help}" in

  # ── Step 1 (do this FIRST): 1-GCD pFq calibration, sizes real throughput ──
  calib)
    sbatch --job-name=cal4f3 --export=ALL,DIM=4,BOX=10,DTYPE=float64,MATRIX=pfq,TARGET=${COV_4F3},BATCHES=100000,500000,2000000 \
        lumi_calibrate_669.sbatch
    sbatch --job-name=cal5f4 --export=ALL,DIM=5,BOX=6,DTYPE=float64,MATRIX=pfq,TARGET=${COV_5F4},BATCHES=100000,500000,2000000 \
        lumi_calibrate_669.sbatch
    echo "[submit] pFq calibration queued (4F3 box10, 5F4 box6). Read calib_*.out for traj/s/GCD."
    ;;

  # ── Step 2: production pFq sweeps, pre-sized to 50% coverage ──────────────
  4f3)
    PG=$(per_gcd ${COV_4F3} ${NODES})
    sbatch --job-name=cmf4f3 --nodes=${NODES} --time=${TIME} \
        --export=ALL,DIM=4,TOTAL_PER_GCD=${PG},BATCH=2000000,BOX=10,THRESH=0.02,DTYPE=float64,MATRIX=pfq,TAG=4f3_pfq_50pct \
        lumi_sweep_669.sbatch
    echo "[submit] 4F3 pFq queued: ${NODES} nodes x 8 GCD x ${PG} = ~${COV_4F3} (50% of box-10 space)."
    ;;

  5f4)
    PG=$(per_gcd ${COV_5F4} ${NODES})
    sbatch --job-name=cmf5f4 --nodes=${NODES} --time=${TIME} \
        --export=ALL,DIM=5,TOTAL_PER_GCD=${PG},BATCH=2000000,BOX=6,THRESH=0.02,DTYPE=float64,MATRIX=pfq,TAG=5f4_pfq_50pct \
        lumi_sweep_669.sbatch
    echo "[submit] 5F4 pFq queued: ${NODES} nodes x 8 GCD x ${PG} = ~${COV_5F4} (50% of box-6 space)."
    ;;

  # ── Task 2: 8F7 negative-delta band sweep (esym companion, 100B) ──────────
  8f7)
    PG=$(per_gcd ${COV_8F7} ${NODES})
    sbatch --job-name=cmf8f7 --nodes=${NODES} --time=${TIME} \
        --export=ALL,DIM=8,TOTAL_PER_GCD=${PG},BATCH=4000000,BOX=12,DTYPE=float32,MATRIX=companion,BAND_LO=-0.5,BAND_HI=0,TAG=8f7_negband_100b \
        lumi_sweep_669.sbatch
    echo "[submit] 8F7 band sweep queued: ${NODES} nodes x 8 GCD x ${PG} = ~${COV_8F7}, keeping -0.5<r2<0."
    ;;

  all)
    "$0" 4f3; "$0" 5f4; "$0" 8f7
    ;;

  *)
    echo "usage: [NODES=N] [TIME=HH:MM:SS] $0 {calib|4f3|5f4|8f7|all}"
    echo "  calib : queue 1-GCD pFq calibration for 4F3 & 5F4 (RUN FIRST)"
    echo "  4f3   : 4F3 exact-pFq sweep, 50% of box-10 space (3.17e11)"
    echo "  5f4   : 5F4 exact-pFq sweep, 50% of box-6 space  (2.33e12)"
    echo "  8f7   : 8F7 companion sweep, 100B, negative-delta band -0.5<r2<0"
    echo "  all   : queue 4f3 + 5f4 + 8f7"
    ;;
esac

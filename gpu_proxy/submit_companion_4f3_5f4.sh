#!/bin/bash -l
# submit_companion_4f3_5f4.sh
# ===========================
# CORRECTED large-scale 4F3 (dim=4) and 5F4 (dim=5) GPU proxy sweeps on LUMI
# (project_465002669), using the COMPANION (esym) gauge -- the ONLY gauge whose
# QR/singular-value proxy r2 = -lam2/lam1 tracks the true arithmetic irrationality
# delta. Validated: dim4 sign-agree 398/400 (MAE 0.035), dim5 300/300 (MAE 0.030),
# dim6 (6F5) 0.996. The earlier submit_5f4_4f3.sh used --matrix pfq, which is a
# NON-NORMAL, NON-STATIONARY realization whose proxy does NOT equal delta (100+10
# sampled pFq "survivors" all had NEGATIVE true delta; sign-agree 0/100). DO NOT
# use pfq for 4F3/5F4 delta hunting.
#
# Run FROM THE LUMI LOGIN NODE, from:
#     /scratch/project_465002669/cmf/continuous_cmf/gpu_proxy
#
# Companion uses the dim-generic esym cocycle (cmf_generic.build_M / proxy_batch),
# the SAME construction proven for the 6F5/8F7 sweeps. Backend = torch (batched
# einsum + manual MGS-QR) in the standard LUMI ROCm PyTorch container -- no cupy
# needed. fp64 for QR safety at these small dims.
#
# COMPANION search-space sizing (space = (2*BOX+1)^nsh * ndir * nz ; nz=44):
#   label dim nsh ndir  box  space         50% coverage
#   4F3    4   7   12    10  9.51e11        4.76e11
#   5F4    5   9   16     6  7.47e12        3.73e12
#
# Stage-1 keeps the validated band 0.02 < r2 < 1.0 (positive-delta regime; the
# r2>1 tail is lam1~=0 degeneracy, not real convergents). Survivors are exact-
# verified (Stage 2) and PSLQ-identified (Stage 3) later on CPU.
#
# Torch companion throughput at dim 4/5 is UNKNOWN until calibrated -> ALWAYS run
# `calib` FIRST, read best traj/s/GCD from calib_*.out, then pick NODES so the
# implied wall time fits standard-g (max 48h).
set -euo pipefail
cd "$(dirname "$0")"

NODES="${NODES:-4}"          # override: NODES=8 ./submit_companion_4f3_5f4.sh 5f4
TIME="${TIME:-24:00:00}"

# Coverage totals (COMPANION space, see table above)
COV_4F3=476000000000          # 4.76e11  (box=10, 50% of space)
# 5F4 box=6 50% (3.73e12) would cost ~7960 node-h; capped instead at a 2000
# node-h budget = 2000 * 8 * 58.57e6 traj/GCD-h (calib 19714358) = 9.37e11
# = 12.5% of the box-6 space. Fits one 48h job on ~48 nodes.
COV_5F4=937000000000          # 9.37e11  (box=6, 2000 node-h cap)

per_gcd () { python3 -c "import math;print(int(math.ceil($1/(8*$2))))"; }

case "${1:-help}" in

  # -- Step 1 (do this FIRST): 1-GCD companion calibration, sizes throughput --
  calib)
    sbatch --job-name=cal4f3c --export=ALL,DIM=4,BOX=10,DTYPE=float64,MATRIX=companion,TARGET=${COV_4F3},BATCHES=1000000,4000000,16000000 \
        lumi_calibrate_669.sbatch
    sbatch --job-name=cal5f4c --export=ALL,DIM=5,BOX=6,DTYPE=float64,MATRIX=companion,TARGET=${COV_5F4},BATCHES=1000000,4000000,16000000 \
        lumi_calibrate_669.sbatch
    echo "[submit] companion calibration queued (4F3 box10, 5F4 box6). Read calib_*.out for traj/s/GCD, then set NODES."
    ;;

  # -- Step 2: production companion sweeps, pre-sized to 50% coverage ---------
  4f3)
    PG=$(per_gcd ${COV_4F3} ${NODES})
    sbatch --job-name=cmp4f3 --nodes=${NODES} --time=${TIME} \
        --export=ALL,DIM=4,TOTAL_PER_GCD=${PG},BATCH=4000000,BOX=10,DTYPE=float64,MATRIX=companion,BAND_LO=0.02,BAND_HI=1.0,TAG=4f3_companion_50pct \
        lumi_sweep_669.sbatch
    echo "[submit] 4F3 companion queued: ${NODES} nodes x 8 GCD x ${PG} = ~${COV_4F3} (50% of box-10 space), band 0.02<r2<1.0."
    ;;

  5f4)
    PG=$(per_gcd ${COV_5F4} ${NODES})
    sbatch --job-name=cmp5f4 --nodes=${NODES} --time=${TIME} \
        --export=ALL,DIM=5,TOTAL_PER_GCD=${PG},BATCH=4000000,BOX=6,DTYPE=float64,MATRIX=companion,BAND_LO=0.02,BAND_HI=1.0,TAG=5f4_companion_2knh \
        lumi_sweep_669.sbatch
    echo "[submit] 5F4 companion queued: ${NODES} nodes x 8 GCD x ${PG} = ~${COV_5F4} (12.5% of box-6 space, 2000 node-h cap), band 0.02<r2<1.0."
    ;;

  both)
    "$0" 4f3; "$0" 5f4
    ;;

  *)
    echo "usage: [NODES=N] [TIME=HH:MM:SS] $0 {calib|4f3|5f4|both}"
    echo "  calib : queue 1-GCD companion calibration for 4F3 & 5F4 (RUN FIRST)"
    echo "  4f3   : 4F3 companion sweep, 50% of box-10 space (4.76e11), band 0.02<r2<1.0"
    echo "  5f4   : 5F4 companion sweep, 2000 node-h cap (9.37e11 = 12.5% of box-6), band 0.02<r2<1.0"
    echo "  both  : queue 4f3 + 5f4"
    ;;
esac

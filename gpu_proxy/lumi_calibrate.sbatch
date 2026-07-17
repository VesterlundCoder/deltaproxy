#!/bin/bash -l
#SBATCH --job-name=cmf_calib
#SBATCH --account=project_465002731
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:20:00
#SBATCH --mem=60G
#SBATCH --output=calib_%j.out

# Calibration: ONE GCD, ~minutes. Measures parity vs numpy + real throughput so
# we can size the 200B / 500B sweeps before committing allocation.
set -euo pipefail

PROJECT=project_465002731
CONTAINER="/projappl/${PROJECT}/mevo_train.sif"
PIP_PKG="/projappl/${PROJECT}/mevo_pip_packages"
HOST_LIBS="/projappl/${PROJECT}/mevo_host_libs"
# Deployed code (see README "Run on LUMI"); override with CODE_DIR=... sbatch ...
CODE_DIR="${CODE_DIR:-/scratch/${PROJECT}/cmf/continuous_cmf}"
DIM="${DIM:-6}"
DTYPE="${DTYPE:-float32}"
BOX="${BOX:-6}"
TARGET="${TARGET:-200e9}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export SINGULARITYENV_PYTHONPATH="${PIP_PKG}:${CODE_DIR}"
export SINGULARITYENV_LD_LIBRARY_PATH="${HOST_LIBS}:/opt/rocm/lib"

srun singularity exec \
    -B "/projappl/${PROJECT}" -B "/scratch/${PROJECT}" \
    "${CONTAINER}" \
  bash -c "cd ${CODE_DIR}/gpu_proxy && python3 calibrate.py \
      --dim ${DIM} --backend torch --dtype ${DTYPE} --box ${BOX} --target ${TARGET}"

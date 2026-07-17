#!/bin/bash -l
#SBATCH --job-name=cmf_calib
#SBATCH --account=project_465002669
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

SIF=/appl/local/containers/tested-containers/lumi-pytorch-rocm-6.0.3-python-3.12-pytorch-v2.3.1-dockerhash-2c1c14cafd28.sif
CODE_DIR="${CODE_DIR:-/scratch/project_465002669/cmf/continuous_cmf}"
DIM="${DIM:-6}"
DTYPE="${DTYPE:-float32}"
BOX="${BOX:-6}"
TARGET="${TARGET:-200e9}"
MATRIX="${MATRIX:-companion}"
BATCHES="${BATCHES:-200000,1000000,4000000,16000000}"

export PYTHONUNBUFFERED=1
export SINGULARITY_BIND=/scratch/project_465002669,/projappl/project_465002669

srun singularity exec "$SIF" bash -c '$WITH_CONDA; \
  export PYTHONPATH='"${CODE_DIR}"':$PYTHONPATH; \
  cd '"${CODE_DIR}"'/gpu_proxy && \
  python3 calibrate.py --dim '"${DIM}"' --matrix '"${MATRIX}"' --backend torch --dtype '"${DTYPE}"' \
      --box '"${BOX}"' --target '"${TARGET}"' \
      --batches '"${BATCHES}"''

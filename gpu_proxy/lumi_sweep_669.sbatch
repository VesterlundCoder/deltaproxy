#!/bin/bash -l
#SBATCH --job-name=cmf_sweep
#SBATCH --account=project_465002669
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=24:00:00
#SBATCH --mem=480G
#SBATCH --output=sweep_%x_%j.out

# Production Stage-1 GPU proxy sweep, project_465002669, LUMI ROCm PyTorch.
# Each task = one MI250X GCD, screening a DISJOINT gid range via seed=(SEED+PROCID).
# Survivors only are written per-GCD; exact verify (Stage 2) runs later on CPU.
#
# Scale by setting --nodes: total screened = TOTAL_PER_GCD * 8 * NODES, because
# SLURM_PROCID is global across all tasks. Examples (after calibration sizes them):
#   sbatch --job-name=cmf6f5 --nodes=N --time=HH:00:00 \
#       --export=ALL,DIM=6,TOTAL_PER_GCD=...,BOX=6,THRESH=0.02,TAG=6f5_200b lumi_sweep_669.sbatch
#   sbatch --job-name=cmf8f7 --nodes=N --time=HH:00:00 \
#       --export=ALL,DIM=8,TOTAL_PER_GCD=...,BOX=12,THRESH=0.01,TAG=8f7_500b lumi_sweep_669.sbatch
set -euo pipefail

SIF=/appl/local/containers/tested-containers/lumi-pytorch-rocm-6.0.3-python-3.12-pytorch-v2.3.1-dockerhash-2c1c14cafd28.sif
CODE_DIR="${CODE_DIR:-/scratch/project_465002669/cmf/continuous_cmf}"

DIM="${DIM:-6}"
TOTAL_PER_GCD="${TOTAL_PER_GCD:-25000000000}"
BATCH="${BATCH:-4000000}"
BOX="${BOX:-6}"
THRESH="${THRESH:-0.02}"
DTYPE="${DTYPE:-float32}"
ZMAX="${ZMAX:-1.0}"
MATRIX="${MATRIX:-companion}"     # companion (esym) | pfq (exact RamanujanTools)
BAND_LO="${BAND_LO:-}"            # optional: keep BAND_LO < r2 < BAND_HI (else r2>THRESH)
BAND_HI="${BAND_HI:-}"
SEED="${SEED:-20260626}"
TAG="${TAG:-sweep}"
OUT_DIR="${OUT_DIR:-/scratch/project_465002669/cmf/runs/${TAG}_${SLURM_JOB_ID}}"
mkdir -p "${OUT_DIR}"

export PYTHONUNBUFFERED=1
export SINGULARITY_BIND=/scratch/project_465002669,/projappl/project_465002669

cat << 'EOF' > "${OUT_DIR}/select_gpu"
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
EOF
chmod +x "${OUT_DIR}/select_gpu"

BAND_ARG=""
[ -n "${BAND_LO}" ] && BAND_ARG="--band ${BAND_LO} ${BAND_HI}"

echo "[lumi_sweep] TAG=${TAG} DIM=${DIM} MATRIX=${MATRIX} nodes=${SLURM_NNODES} "\
"TOTAL_PER_GCD=${TOTAL_PER_GCD} BOX=+-${BOX} THRESH=${THRESH} BAND='${BAND_LO} ${BAND_HI}' DTYPE=${DTYPE} -> ${OUT_DIR}"

srun "${OUT_DIR}/select_gpu" \
  singularity exec "$SIF" bash -c '$WITH_CONDA; \
    export PYTHONPATH='"${CODE_DIR}"':$PYTHONPATH; \
    cd '"${CODE_DIR}"'/gpu_proxy && \
    python3 gpu_sweep.py \
      --dim '"${DIM}"' --matrix '"${MATRIX}"' --backend torch --dtype '"${DTYPE}"' --device cuda \
      --total '"${TOTAL_PER_GCD}"' --batch '"${BATCH}"' --box '"${BOX}"' \
      --z-max '"${ZMAX}"' --thresh '"${THRESH}"' '"${BAND_ARG}"' \
      --seed $(( '"${SEED}"' + SLURM_PROCID )) \
      --out '"${OUT_DIR}"'/survivors_p${SLURM_PROCID}.jsonl \
      > '"${OUT_DIR}"'/log_p${SLURM_PROCID}.txt 2>&1'

rm -f "${OUT_DIR}/select_gpu"
cat "${OUT_DIR}"/survivors_p*.jsonl > "${OUT_DIR}/survivors_all.jsonl" 2>/dev/null || true
echo "[lumi_sweep] DONE survivors=$(wc -l < ${OUT_DIR}/survivors_all.jsonl) -> ${OUT_DIR}/survivors_all.jsonl"

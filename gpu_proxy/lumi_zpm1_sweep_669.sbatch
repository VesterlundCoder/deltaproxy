#!/bin/bash -l
#SBATCH --job-name=zpm1_6f5
#SBATCH --account=project_465002669
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=12:00:00
#SBATCH --mem=480G
#SBATCH --output=zpm1_%x_%j.out

# z=±1 CRITICAL-REGIME SWEEP for 6F5 companion CMFs.
#
# Total space (z=+1,-1; box6; dim6) = 6.81e13 trajectories.
# 10% = 6.81e12.  On 8 nodes × 8 GCDs = 64 GCDs, each scans ~1.06e11.
#
# Funnel: keep -0.1 < r2 < 0.1 (band around 0 — the z=±1 boundary is
# polynomial-convergence, so δ≈0; standard thresh=0.02 would reject everything).
#
# fp32 screen for max throughput; survivors re-verified exactly in Stage 2.
set -euo pipefail

SIF=/appl/local/containers/tested-containers/lumi-pytorch-rocm-6.0.3-python-3.12-pytorch-v2.3.1-dockerhash-2c1c14cafd28.sif
CODE_DIR="${CODE_DIR:-/scratch/project_465002669/cmf/continuous_cmf}"

DIM="${DIM:-6}"
BOX="${BOX:-6}"
DTYPE="${DTYPE:-float32}"
NSTEPS="${NSTEPS:-120}"
BATCH="${BATCH:-4000000}"
SEED="${SEED:-20260706}"
TAG="${TAG:-zpm1_10pct}"
OUT_DIR="${OUT_DIR:-/scratch/project_465002669/cmf/runs/${TAG}_${SLURM_JOB_ID}}"
mkdir -p "${OUT_DIR}"

# 10% of 6.81e13 / 64 GCDs = 1.064e11 per GCD
TOTAL_PER_GCD="${TOTAL_PER_GCD:-106400000000}"

export PYTHONUNBUFFERED=1
export SINGULARITY_BIND=/scratch/project_465002669,/projappl/project_465002669

cat << 'EOF' > "${OUT_DIR}/select_gpu"
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
EOF
chmod +x "${OUT_DIR}/select_gpu"

echo "[zpm1] TAG=${TAG} DIM=${DIM} nodes=${SLURM_NNODES} GCDs=${SLURM_NTASKS} "\
"TOTAL_PER_GCD=${TOTAL_PER_GCD} BOX=+-${BOX} band=[-0.1,0.1] DTYPE=${DTYPE} -> ${OUT_DIR}"

srun "${OUT_DIR}/select_gpu" \
  singularity exec "$SIF" bash -c '$WITH_CONDA; \
    export PYTHONPATH='"${CODE_DIR}"':$PYTHONPATH; \
    cd '"${CODE_DIR}"'/gpu_proxy && \
    python3 gpu_sweep.py \
      --dim '"${DIM}"' --matrix companion --backend torch --dtype '"${DTYPE}"' --device cuda \
      --total '"${TOTAL_PER_GCD}"' --batch '"${BATCH}"' --box '"${BOX}"' \
      --z-fixed "1,1;-1,1" --band -0.1 0.1 \
      --nrank '"${NSTEPS}"' \
      --seed $(( '"${SEED}"' + SLURM_PROCID )) \
      --out '"${OUT_DIR}"'/survivors_p${SLURM_PROCID}.jsonl \
      > '"${OUT_DIR}"'/log_p${SLURM_PROCID}.txt 2>&1'

rm -f "${OUT_DIR}/select_gpu"
cat "${OUT_DIR}"/survivors_p*.jsonl > "${OUT_DIR}/survivors_all.jsonl" 2>/dev/null || true
N=$(wc -l < "${OUT_DIR}/survivors_all.jsonl" 2>/dev/null || echo 0)
echo "[zpm1] DONE survivors=${N} -> ${OUT_DIR}/survivors_all.jsonl"
echo "[zpm1] Stage 2: download + python3 pslq_companion.py --in survivors_all.jsonl --dim 6 --min-delta -0.1"

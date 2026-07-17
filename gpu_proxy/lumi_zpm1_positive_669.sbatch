#!/bin/bash -l
#SBATCH --job-name=zpm1_pos_6f5
#SBATCH --account=project_465002669
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=12:00:00
#SBATCH --mem=480G
#SBATCH --output=zpm1_pos_%x_%j.out

# z=+1/-1 POSITIVE-DELTA SWEEP for 6F5 companion CMFs.
#
# Total space (z=+1,-1; box6; dim6, non-degenerate) = 6.81e13 trajectories.
# At ~301k traj/s/GCD (MI250X fp32), 64 GCDs screen:
#   12h job:  8.3e11 trajectories  (1.2% of total)
#   24h job:  1.66e12              (2.4%)
#   Full space needs ~41 days or multiple submissions.
#
# Local CPU test (200k trajectories, seed=20260717):
#   - Non-degenerate, lambda_1 > 0 filter active
#   - 1 survivor with r2 = +0.001 (exact delta = +0.004 at N=400, -> 0)
#   - Positive rate ~ 5e-6 (weak positives, delta -> 0)
#   - All strong r2 values (up to +1.33) are degenerate false positives
#
# Filters: --require-pos-lam1 (reject contracting denominators)
#          --no-degenerate (reject shift[i]==-1 for i in range(dim))
#
# Usage:
#   sbatch lumi_zpm1_positive_669.sbatch
#   # 12h default: 1.3e10 per GCD = 8.3e11 total
#   TOTAL_PER_GCD=13000000000 sbatch lumi_zpm1_positive_669.sbatch
#   # 24h: 2.6e10 per GCD = 1.66e12 total
#   TOTAL_PER_GCD=26000000000 sbatch --time=24:00:00 lumi_zpm1_positive_669.sbatch

set -euo pipefail

SIF=/appl/local/containers/tested-containers/lumi-pytorch-rocm-6.0.3-python-3.12-pytorch-v2.3.1-dockerhash-2c1c14cafd28.sif
CODE_DIR="${CODE_DIR:-/scratch/project_465002669/cmf/continuous_cmf}"

DIM="${DIM:-6}"
BOX="${BOX:-6}"
DTYPE="${DTYPE:-float32}"
NSTEPS="${NSTEPS:-120}"
BATCH="${BATCH:-4000000}"
SEED="${SEED:-20260717}"
THRESH="${THRESH:-0.0}"
BACKEND="${BACKEND:-torch}"
TAG="${TAG:-zpm1_pos}"
OUT_DIR="${OUT_DIR:-/scratch/project_465002669/cmf/runs/${TAG}_${SLURM_JOB_ID}}"
mkdir -p "${OUT_DIR}"

# 12h at 301k/s/GCD = 1.3e10 per GCD = 8.3e11 total (64 GCDs)
TOTAL_PER_GCD="${TOTAL_PER_GCD:-13000000000}"

export PYTHONUNBUFFERED=1
export SINGULARITY_BIND=/scratch/project_465002669,/projappl/project_465002669

cat << 'EOF' > "${OUT_DIR}/select_gpu"
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
EOF
chmod +x "${OUT_DIR}/select_gpu"

echo "[zpm1_pos] TAG=${TAG} DIM=${DIM} nodes=${SLURM_NNODES} GCDs=${SLURM_NTASKS} "\
"TOTAL_PER_GCD=${TOTAL_PER_GCD} BOX=+-${BOX} thresh=${THRESH} DTYPE=${DTYPE} -> ${OUT_DIR}"
echo "[zpm1_pos] z=+1,-1 only  backend=${BACKEND}  seed_base=${SEED}"

srun "${OUT_DIR}/select_gpu" \
  singularity exec "$SIF" bash -c '$WITH_CONDA; \
    export PYTHONPATH='"${CODE_DIR}"':$PYTHONPATH; \
    cd '"${CODE_DIR}"'/gpu_proxy && \
    python3 gpu_sweep.py \
      --dim '"${DIM}"' --matrix companion --backend '"${BACKEND}"' --dtype '"${DTYPE}"' --device cuda \
      --total '"${TOTAL_PER_GCD}"' --batch '"${BATCH}"' --box '"${BOX}"' \
      --z-fixed "1,1;-1,1" --thresh '"${THRESH}"' --require-pos-lam1 --no-degenerate \
      --nrank '"${NSTEPS}"' \
      --seed $(( '"${SEED}"' + SLURM_PROCID )) \
      --out '"${OUT_DIR}"'/survivors_p${SLURM_PROCID}.jsonl \
      > '"${OUT_DIR}"'/log_p${SLURM_PROCID}.txt 2>&1'

rm -f "${OUT_DIR}/select_gpu"
cat "${OUT_DIR}"/survivors_p*.jsonl > "${OUT_DIR}/survivors_all.jsonl" 2>/dev/null || true
N=$(wc -l < "${OUT_DIR}/survivors_all.jsonl" 2>/dev/null || echo 0)
echo "[zpm1_pos] DONE survivors=${N} -> ${OUT_DIR}/survivors_all.jsonl"
echo "[zpm1_pos] Stage 2: download + python3 pslq_companion.py --in survivors_all.jsonl --dim 6"

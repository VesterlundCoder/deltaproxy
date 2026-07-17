#!/bin/bash -l
#SBATCH --job-name=cmf_cluster
#SBATCH --account=project_465002669
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=06:00:00
#SBATCH --mem=480G
#SBATCH --output=cluster_%x_%j.out

# Anchored neighborhood ("cluster") sweep on LUMI, CONSTANT-AGNOSTIC.
# Each of the 8 GCDs perturbs the SAME top-delta anchor set with a DIFFERENT seed
# (SEED + SLURM_PROCID) -> complementary neighborhood coverage, zero overlap of
# random draws. Survivors (delta>=thresh) are tiny; PSLQ (20-constant library)
# is done LATER, locally. cluster_sweep.py --backend torch runs in the standard
# LUMI ROCm PyTorch container (validated vs numpy: median 7e-16, sign 1.0000).
#
# Which constant a survivor hits is decided at PSLQ, so ONE cluster serves all
# 20 constants (zeta2/3/5/7, pi^k, Catalan, ln2/3/5, e, gamma, phi, sqrt.., ...).
#
# Sizing: candidates = NODES * 8 * N_ANCHORS * PER_ANCHOR.
#   1 node, 5000 anchors, per_anchor=2500 -> 1.0e8 candidates.
#   survivor rate ~0.2-0.5% at thresh=0.02 -> ~2-5e5 survivors (downloadable).
set -euo pipefail

SIF=/appl/local/containers/tested-containers/lumi-pytorch-rocm-6.0.3-python-3.12-pytorch-v2.3.1-dockerhash-2c1c14cafd28.sif
CODE_DIR="${CODE_DIR:-/scratch/project_465002669/cmf/continuous_cmf}"

DIM="${DIM:-6}"
ANCHORS="${ANCHORS:-${CODE_DIR}/cluster_out/anchors_top5000.jsonl}"
N_ANCHORS="${N_ANCHORS:-5000}"
PER_ANCHOR="${PER_ANCHOR:-2500}"
RADIUS="${RADIUS:-2}"
DIR_SWITCH="${DIR_SWITCH:-0.15}"
ZMODE="${ZMODE:-extended}"
ZMIN="${ZMIN:-0.02}"
ZMAX="${ZMAX:-0.999}"
NSTEPS="${NSTEPS:-120}"
THRESH="${THRESH:-0.02}"
BATCH="${BATCH:-500000}"
DTYPE="${DTYPE:-float64}"
SEED="${SEED:-20260706}"
TAG="${TAG:-cluster6f5}"
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

echo "[cluster] TAG=${TAG} DIM=${DIM} anchors=${N_ANCHORS} per_anchor=${PER_ANCHOR} "\
"radius=+-${RADIUS} zmode=${ZMODE} z=[${ZMIN},${ZMAX}] thresh=${THRESH} -> ${OUT_DIR}"

srun "${OUT_DIR}/select_gpu" \
  singularity exec "$SIF" bash -c '$WITH_CONDA; \
    export PYTHONPATH='"${CODE_DIR}"':$PYTHONPATH; \
    cd '"${CODE_DIR}"' && \
    python3 cluster_sweep.py \
      --anchors '"${ANCHORS}"' --dim '"${DIM}"' \
      --n-anchors '"${N_ANCHORS}"' --per-anchor '"${PER_ANCHOR}"' \
      --radius '"${RADIUS}"' --dir-switch '"${DIR_SWITCH}"' \
      --z-mode '"${ZMODE}"' --z-min '"${ZMIN}"' --z-max '"${ZMAX}"' \
      --nsteps '"${NSTEPS}"' --thresh '"${THRESH}"' --batch '"${BATCH}"' \
      --backend torch --device cuda --dtype '"${DTYPE}"' \
      --seed $(( '"${SEED}"' + SLURM_PROCID )) \
      --out '"${OUT_DIR}"'/survivors_p${SLURM_PROCID}.jsonl \
      > '"${OUT_DIR}"'/log_p${SLURM_PROCID}.txt 2>&1'

rm -f "${OUT_DIR}/select_gpu"
cat "${OUT_DIR}"/survivors_p*.jsonl > "${OUT_DIR}/survivors_all.jsonl" 2>/dev/null || true
echo "[cluster] DONE survivors=$(wc -l < ${OUT_DIR}/survivors_all.jsonl) -> ${OUT_DIR}/survivors_all.jsonl"

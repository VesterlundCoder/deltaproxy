#!/bin/bash -l
#SBATCH --job-name=cmf_sweep
#SBATCH --account=project_465002731
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=24:00:00
#SBATCH --mem=480G
#SBATCH --output=sweep_%x_%j.out

# Production Stage-1 GPU proxy sweep on 1 node = 8 MI250X GCDs.
# Each task (GCD) screens a DISJOINT gid range via seed=(SEED+PROCID); survivors
# only are written per-GCD. Exact verification (Stage 2) is run separately on
# CPU afterwards so no GPU walltime is spent on it.
#
# Parameterize via env, e.g. the 6F5 and 8F7 launches:
#   sbatch --job-name=cmf6f5 --export=ALL,DIM=6,TOTAL_PER_GCD=25000000000,BOX=6,THRESH=0.02,TAG=6f5_200b lumi_sweep.sbatch
#   sbatch --job-name=cmf8f7 --export=ALL,DIM=8,TOTAL_PER_GCD=62500000000,BOX=12,THRESH=0.01,TAG=8f7_500b lumi_sweep.sbatch
set -euo pipefail

PROJECT=project_465002731
CONTAINER="/projappl/${PROJECT}/mevo_train.sif"
PIP_PKG="/projappl/${PROJECT}/mevo_pip_packages"
HOST_LIBS="/projappl/${PROJECT}/mevo_host_libs"
CODE_DIR="${CODE_DIR:-/scratch/${PROJECT}/cmf/continuous_cmf}"

DIM="${DIM:-6}"
TOTAL_PER_GCD="${TOTAL_PER_GCD:-25000000000}"   # 8 GCDs * 25e9 = 200e9
BATCH="${BATCH:-4000000}"
BOX="${BOX:-6}"
THRESH="${THRESH:-0.02}"
DTYPE="${DTYPE:-float32}"
ZMAX="${ZMAX:-1.0}"
SEED="${SEED:-20260626}"
TAG="${TAG:-sweep}"
OUT_DIR="${OUT_DIR:-/scratch/${PROJECT}/cmf/runs/${TAG}_${SLURM_JOB_ID}}"
mkdir -p "${OUT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export SINGULARITYENV_PYTHONPATH="${PIP_PKG}:${CODE_DIR}"
export SINGULARITYENV_LD_LIBRARY_PATH="${HOST_LIBS}:/opt/rocm/lib"

# one GCD per task
cat << 'EOF' > "${OUT_DIR}/select_gpu"
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec $*
EOF
chmod +x "${OUT_DIR}/select_gpu"

echo "[lumi_sweep] TAG=${TAG} DIM=${DIM} TOTAL_PER_GCD=${TOTAL_PER_GCD} "\
"x8 = $(python3 -c "print(f'{${TOTAL_PER_GCD}*8:.3g}')") total  BOX=+-${BOX} "\
"THRESH=${THRESH} DTYPE=${DTYPE} -> ${OUT_DIR}"

srun "${OUT_DIR}/select_gpu" \
  singularity exec \
    -B "/projappl/${PROJECT}" -B "/scratch/${PROJECT}" \
    "${CONTAINER}" \
  bash -c 'cd '"${CODE_DIR}"'/gpu_proxy && \
    python3 gpu_sweep.py \
      --dim '"${DIM}"' --backend torch --dtype '"${DTYPE}"' --device cuda \
      --total '"${TOTAL_PER_GCD}"' --batch '"${BATCH}"' --box '"${BOX}"' \
      --z-max '"${ZMAX}"' --thresh '"${THRESH}"' \
      --seed $(( '"${SEED}"' + SLURM_PROCID )) \
      --out '"${OUT_DIR}"'/survivors_gcd${SLURM_PROCID}.jsonl \
      > '"${OUT_DIR}"'/log_gcd${SLURM_PROCID}.txt 2>&1'

rm -f "${OUT_DIR}/select_gpu"
cat "${OUT_DIR}"/survivors_gcd*.jsonl > "${OUT_DIR}/survivors_all.jsonl" 2>/dev/null || true
echo "[lumi_sweep] DONE. survivors: $(wc -l < ${OUT_DIR}/survivors_all.jsonl) -> ${OUT_DIR}/survivors_all.jsonl"
echo "  Stage 2 (CPU): python3 verify_survivors.py --in ${OUT_DIR}/survivors_all.jsonl --dim ${DIM} --workers 32"

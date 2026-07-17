#!/bin/bash
#SBATCH --job-name=cmf_gpu_proxy
#SBATCH --account=project_XXXXXXXXX     # <-- set your MEVO/LUMI project
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8               # MI250X = 8 GCDs visible
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=02:00:00
#SBATCH --output=gpu_proxy_%j.out

# Stage-1 GPU proxy sweep on LUMI (AMD MI250X, ROCm). One task per GCD; each
# task screens a disjoint gid range so there is zero overlap and full coverage.
# Survivors (tiny) are written per-task, then merged + exact-verified on CPU.

module load LUMI/23.09 partition/G
module load rocm
# cupy-rocm in your venv (see README). Adjust to your environment:
source "$HOME/cmf-rocm-venv/bin/activate"

DIM=6
TOTAL_PER_GCD=2000000000      # 2e9 per GCD -> 1.6e10 per node / 2h
BATCH=4000000
THRESH=0.02
SEED=20260626
DTYPE=float32                 # fp32 screen; survivors re-checked in fp64 + exact

srun --gpu-bind=closest bash -c '
  GCD=$SLURM_PROCID
  python3 gpu_sweep.py \
    --dim '"$DIM"' --backend cupy --dtype '"$DTYPE"' \
    --total '"$TOTAL_PER_GCD"' --batch '"$BATCH"' \
    --thresh '"$THRESH"' --seed $(( '"$SEED"' + GCD )) \
    --out survivors_${SLURM_JOB_ID}_gcd${GCD}.jsonl
'

# merge + exact verify (CPU, after the GPU stage)
cat survivors_${SLURM_JOB_ID}_gcd*.jsonl > survivors_${SLURM_JOB_ID}_all.jsonl
python3 verify_survivors.py \
    --in survivors_${SLURM_JOB_ID}_all.jsonl --dim $DIM \
    --nverify 100 --workers $SLURM_CPUS_PER_TASK --thresh 0.0 \
    --out confirmed_${SLURM_JOB_ID}.jsonl

#!/bin/bash -l
#SBATCH --job-name=guided-e-search
#SBATCH --account=project_465002952
#SBATCH --output=/scratch/project_465002952/guided_e_search/logs/stdout.log
#SBATCH --error=/scratch/project_465002952/guided_e_search/logs/stderr.log
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --mem=64G

set -euo pipefail

cd /scratch/project_465002952/guided_e_search

module load LUMI/24.03
module load cray-python

pip install --user mpmath numpy 2>/dev/null || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python3 guided_e_search.py

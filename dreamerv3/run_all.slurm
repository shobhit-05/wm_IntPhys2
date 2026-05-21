#!/bin/bash
#SBATCH --job-name=dv3_rq3_full
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/projects/infoseeking/maanyacb/rq3_outputs/slurm-%j.out

module purge
module load conda/Miniforge3-25.3.1-3
module load gcc/13.4.0
conda activate /gpfs/projects/infoseeking/sgupta01/conda_envs/dreamerv3

mkdir -p /gpfs/projects/infoseeking/maanyacb/pip_cache
mkdir -p /gpfs/projects/infoseeking/maanyacb/pip_tmp

export PIP_CACHE_DIR=/gpfs/projects/infoseeking/maanyacb/pip_cache
export TMPDIR=/gpfs/projects/infoseeking/maanyacb/pip_tmp
export XDG_CACHE_HOME=/gpfs/projects/infoseeking/maanyacb/pip_cache
export JAX_COMPILATION_CACHE_DIR=/gpfs/projects/infoseeking/maanyacb/pip_tmp

cd /gpfs/projects/infoseeking/maanyacb
python3 run_all_metrics.py

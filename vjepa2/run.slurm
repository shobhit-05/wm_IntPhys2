#!/bin/bash
#SBATCH --job-name=vjepa_extract
#SBATCH --partition=ckpt-all
#SBATCH --account=astro-ckpt
#SBATCH --qos=ckpt
#SBATCH --array=0-3
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/vjepa2/slurm-extract-%A_%a.out

set -e

WORK=/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/vjepa2
cd "$WORK"

module purge
module load conda/Miniforge3-25.9.1-0
module load gcc/13.2.0

conda activate /mmfs1/gscratch/astro/klinjin/conda_envs/vjepa2

echo "=== Rank ${SLURM_ARRAY_TASK_ID}/4 — extracting embeddings ==="
python vjepa_extract_embeddings.py \
    --rank ${SLURM_ARRAY_TASK_ID} \
    --world-size 4

echo "=== Rank ${SLURM_ARRAY_TASK_ID} done ==="

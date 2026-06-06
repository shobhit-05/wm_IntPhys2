#!/bin/bash
#SBATCH --job-name=vjepa_metrics
#SBATCH --partition=ckpt-all
#SBATCH --account=astro-ckpt
#SBATCH --qos=ckpt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/vjepa2/slurm-metrics-%j.out

set -e

WORK=/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/vjepa2
cd "$WORK"

module purge
module load conda/Miniforge3-25.9.1-0
module load gcc/13.2.0

conda activate /mmfs1/gscratch/astro/klinjin/conda_envs/vjepa2

echo "=== Merging 4 shards ==="
python vjepa_merge_shards.py --world-size 4

echo ""
echo "=== Computing metrics ==="
python vjepa_metrics.py

echo ""
echo "=== Done. Results in $WORK/outputs/ ==="

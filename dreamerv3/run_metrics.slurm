#!/bin/bash
#SBATCH --job-name=dv3_metrics_only
#SBATCH --partition=ckpt-all
#SBATCH --account=astro-ckpt
#SBATCH --qos=ckpt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/dreamerv3/slurm-metrics-%j.out

# Runs dreamerv3_compute_metrics.py from pre-extracted latent npy files.
# Requires outputs/id_latents.npy and outputs/ood_latents.npy to exist.
# Copy them from tillicum if needed:
#   scp tillicum:/gpfs/projects/infoseeking/maanyacb/rq3_outputs/dreamerv3/{id,ood}_latents.npy \
#       /mmfs1/gscratch/astro/klinjin/wm_IntPhys2/dreamerv3/outputs/

set -e

WORK=/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/dreamerv3

for f in id_latents.npy ood_latents.npy id_metadata.json ood_metadata.json; do
    if [ ! -f "$WORK/outputs/$f" ]; then
        echo "ERROR: missing $WORK/outputs/$f"
        exit 1
    fi
done

cd "$WORK"

module purge
module load conda/Miniforge3-25.9.1-0
module load gcc/13.2.0

conda activate /mmfs1/gscratch/astro/klinjin/conda_envs/vjepa2

echo "=== DreamerV3 metrics from cached latents ==="
python dreamerv3_compute_metrics.py

echo ""
echo "=== Done. Results in $WORK/outputs/ ==="

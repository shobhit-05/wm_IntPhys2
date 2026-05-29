#!/bin/bash
#SBATCH --job-name=dv3_rq3_full
#SBATCH --partition=ckpt-all
#SBATCH --account=astro-ckpt
#SBATCH --qos=ckpt
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/dreamerv3/slurm-%j.out

set -e

WORK=/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/dreamerv3

# Checkpoint: copy from tillicum before submitting this job:
#   scp tillicum:/gpfs/projects/infoseeking/sgupta01/dreamerv3_logs/latest_atari100k_pong/ckpt/20260125T200341F803200/agent.pkl \
#       $WORK/checkpoints/agent.pkl
export DREAMERV3_CKPT=$WORK/checkpoints/agent.pkl
export DREAMERV3_REPO=$WORK/dreamerv3_repo

if [ ! -f "$DREAMERV3_CKPT" ]; then
    echo "ERROR: checkpoint not found at $DREAMERV3_CKPT"
    echo "Run: scp tillicum:/gpfs/projects/infoseeking/sgupta01/dreamerv3_logs/latest_atari100k_pong/ckpt/20260125T200341F803200/agent.pkl $WORK/checkpoints/agent.pkl"
    exit 1
fi

cd "$WORK"

module purge
module load conda/Miniforge3-25.9.1-0
module load gcc/13.2.0

conda activate /mmfs1/gscratch/astro/klinjin/conda_envs/dreamerv3

mkdir -p outputs

export JAX_COMPILATION_CACHE_DIR=$WORK/.jax_cache
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "=== DreamerV3 full pipeline: latent extraction + RQ3 metrics ==="
python run_all_metrics.py

echo ""
echo "=== Done. Results in $WORK/outputs/ ==="

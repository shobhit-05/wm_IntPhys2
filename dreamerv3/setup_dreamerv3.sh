#!/bin/bash
# Create conda env and clone dreamerv3 for Hyak.
# Run once from a login node (no GPU needed).
#
# After this runs, copy the checkpoint:
#   scp tillicum:/gpfs/projects/infoseeking/sgupta01/dreamerv3_logs/latest_atari100k_pong/ckpt/20260125T200341F803200/agent.pkl \
#       checkpoints/agent.pkl

set -e

WORK=/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/dreamerv3
ENV_DIR=/mmfs1/gscratch/astro/klinjin/conda_envs/dreamerv3

module purge
module load conda/Miniforge3-25.9.1-0
module load gcc/13.2.0

# ── Create environment ─────────────────────────────────────────────────
if [ ! -d "$ENV_DIR" ]; then
    echo "=== Creating conda env: $ENV_DIR ==="
    conda create -y -p "$ENV_DIR" python=3.11
else
    echo "=== Conda env already exists: $ENV_DIR ==="
fi

conda activate "$ENV_DIR"

# ── Clone dreamerv3 repo ───────────────────────────────────────────────
REPO_DIR=$WORK/dreamerv3_repo
if [ ! -d "$REPO_DIR" ]; then
    echo "=== Cloning dreamerv3 ==="
    git clone https://github.com/danijar/dreamerv3.git "$REPO_DIR"
else
    echo "=== dreamerv3 repo already exists; pulling latest ==="
    git -C "$REPO_DIR" pull --ff-only
fi

# ── Install dependencies ───────────────────────────────────────────────
echo "=== Installing core deps ==="
pip install --upgrade pip

# JAX with CUDA 12 pip wheels (compatible with CUDA 13 on this cluster)
pip install "jax[cuda12]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# dreamerv3 requirements (skip nvidia-cuda-nvcc-cu12 to avoid CUDA version conflict)
pip install \
    "ninjax>=3.5.1" \
    "elements>=3.19.1" \
    "numpy<2" \
    "granular>=0.20.3" \
    "portal>=3.5.0" \
    "scope>=0.4.4" \
    optax \
    chex \
    einops \
    jaxtyping \
    imageio \
    Pillow \
    scikit-learn

echo ""
echo "=== Setup complete ==="
echo "Conda env: $ENV_DIR"
echo "Repo:      $REPO_DIR"
echo ""
echo "Next: copy the checkpoint to $WORK/checkpoints/agent.pkl"
echo "Then: sbatch $WORK/run_all.slurm"

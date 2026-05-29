#!/bin/bash
# Setup script for vjepa-h-rope IntPhys2 evaluation
# Run once from /mmfs1/home/lindajin/wm_IntPhys2/vjepa2/

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== [1/3] Cloning IntPhys2 repository ==="
if [ ! -d "IntPhys2" ]; then
    git clone https://github.com/facebookresearch/IntPhys2.git
    echo "Cloned to $DIR/IntPhys2"
else
    echo "IntPhys2 already exists, pulling latest..."
    cd IntPhys2 && git pull && cd ..
fi

echo ""
echo "=== [2/3] Downloading V-JEPA ViT-H checkpoint ==="
mkdir -p checkpoints
CKPT="checkpoints/vith16.pth.tar"
if [ ! -f "$CKPT" ]; then
    wget https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar \
         -O "$CKPT" --progress=bar:force
    echo "Saved to $DIR/$CKPT"
else
    echo "$CKPT already exists, skipping download."
fi

echo ""
echo "=== [3/3] Checking Python dependencies ==="
python3 -c "import torch; print(f'torch {torch.__version__}')"
python3 -c "import decord; print('decord OK')" || {
    echo "decord not found, installing..."
    pip install decord
}
python3 -c "import sklearn; print(f'sklearn {sklearn.__version__}')"
python3 -c "import numpy; print(f'numpy {numpy.__version__}')"

echo ""
echo "=== [4/4] Verifying model import ==="
python3 -c "
import sys
sys.path.insert(0, '$DIR/IntPhys2/prediction_evals')
import app.vjepa.models.vision_transformer as vit_module
m = vit_module.vit_huge_rope(
    img_size=224, num_frames=16, patch_size=16,
    tubelet_size=2, uniform_power=True, use_sdpa=False,
    use_SiLU=True, wide_SiLU=False, is_causal=False,
)
import torch
x = torch.zeros(1, 3, 16, 224, 224)
with torch.no_grad():
    out = m(x)
print(f'Model OK — output shape: {out.shape}')  # expect [1, 1568, 1280]
"

echo ""
echo "Setup complete. Next: sbatch run.slurm  (or: python vjepa_extract_embeddings.py)"

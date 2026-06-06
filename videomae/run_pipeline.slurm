#!/bin/bash
#SBATCH --job-name=videomae_metrics
#SBATCH --partition=ckpt-all
#SBATCH --account=astro-ckpt
#SBATCH --qos=ckpt
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/videomae/slurm-pipeline-%j.out

set -e

WORK=/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/videomae
VIDEOS_DIR=/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/IntPhys2/Main/Videos
METADATA_CSV=/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/IntPhys2/Main/metadata.csv
CKPT_DIR=$WORK/checkpoints
MANIFEST=$WORK/manifests/intphys2_main1012.txt
INFER_OUT=$WORK/outputs/per_video_embeddings
BUNDLE_DIR=$WORK/outputs

cd "$WORK"

module purge
module load conda/Miniforge3-25.9.1-0
module load gcc/13.2.0

conda activate /mmfs1/gscratch/astro/klinjin/conda_envs/vjepa2

export HUGGING_FACE_HUB_TOKEN=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
export HF_TOKEN=$HUGGING_FACE_HUB_TOKEN

# ── Step 1: Download VideoMAEv2-Base checkpoint ───────────────────────────
mkdir -p "$CKPT_DIR"
CKPT_PATH=$CKPT_DIR/model.safetensors

if [ ! -f "$CKPT_PATH" ]; then
    echo "=== Downloading VideoMAEv2-Base checkpoint ==="
    python - <<'EOF'
import os
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="OpenGVLab/VideoMAEv2-Base",
    filename="model.safetensors",
    local_dir=os.environ["CKPT_DIR"],
    token=os.environ["HF_TOKEN"],
)
print(f"Downloaded to: {path}")
EOF
else
    echo "=== Checkpoint already exists: $CKPT_PATH ==="
fi

# ── Step 2: Build manifest of all 1012 videos ────────────────────────────
mkdir -p "$(dirname $MANIFEST)"
echo "=== Building manifest ==="
python - <<'EOF'
import csv, os
from pathlib import Path

metadata_csv = os.environ["METADATA_CSV"]
videos_dir   = Path(os.environ["VIDEOS_DIR"])
manifest     = os.environ["MANIFEST"]

with open(metadata_csv) as f:
    rows = list(csv.DictReader(f))

paths = []
for r in rows:
    p = videos_dir / Path(r["file_name"]).name
    if p.exists():
        paths.append(str(p))
    else:
        print(f"WARNING: missing {p}")

with open(manifest, "w") as f:
    f.write("\n".join(paths) + "\n")

print(f"Manifest written: {len(paths)} videos -> {manifest}")
EOF

# ── Step 3: Run VideoMAE inference on all videos ──────────────────────────
mkdir -p "$INFER_OUT"
echo ""
echo "=== Running VideoMAE inference ==="
python infer_intphys2_subset.py \
    --manifest "$MANIFEST" \
    --output_dir "$INFER_OUT" \
    --ckpt_path "$CKPT_PATH" \
    --model_name vit_base_patch16_224 \
    --clip_len 16 \
    --input_size 224 \
    --max_videos 2000 \
    --per_video_subdir ""

# ── Step 4: Aggregate per-video embeddings into bundle ────────────────────
echo ""
echo "=== Aggregating embeddings into bundle ==="
python aggregate_intphys2_embeddings.py \
    --summary_jsonl "$INFER_OUT/summary.jsonl" \
    --metadata_csv "$METADATA_CSV" \
    --output_dir "$BUNDLE_DIR" \
    --output_prefix intphys2_main1012

# ── Step 5: Compute all RQ3 metrics ──────────────────────────────────────
echo ""
echo "=== Computing RQ3 metrics ==="
python compute_intphys2_main1012_dci_cv.py \
    --bundle "$BUNDLE_DIR/intphys2_main1012_embeddings_bundle.pt" \
    --output_dir "$BUNDLE_DIR"

echo ""
echo "=== VideoMAE pipeline complete ==="
echo "Results: $BUNDLE_DIR/videomae_rq3_metrics.{json,csv,txt}"

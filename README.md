# IntPhys2 Model Setup: V-JEPA, DreamerV3, and VideoMAEv2

## 1. Overview

The comparison has four repeated stages:

1. Set up model code and environment.
2. Load pretrained weights or trained checkpoints.
3. Extract IntPhys2 embeddings or latents.
4. Run unified metrics and UMAP plots.

Outputs are expected as JSON/CSV/TXT metrics files and PNG UMAP plots.

---

## 2. Shared Data and Output Layout

### IntPhys2 Main Data

The shared IntPhys2 Main split contains 1012 videos:
- 506 Possible / ID videos
- 506 Impossible / OOD videos

DreamerV3 and VideoMAEv2 run on the `/gpfs/projects/infoseeking` cluster paths:
```bash
/gpfs/projects/infoseeking/sgupta01/VideoMAEv2/data/IntPhys2/Main/metadata.csv
/gpfs/projects/infoseeking/sgupta01/VideoMAEv2/data/IntPhys2/Main/Videos/
```

V-JEPA runs on the `/mmfs1` cluster paths:
```python
METADATA_CSV = '/mmfs1/home/lindajin/wm_IntPhys2/videomae/metadata.csv'
VIDEOS_BASE  = '/mmfs1/home/lindajin/wm_IntPhys2/videomae/'
```

### Canonical Joined Labels

DreamerV3 and VideoMAEv2 use the VideoMAE joined label file for consistent `condition`, `type`, `difficulty`, and `camera` labels:

```bash
/gpfs/projects/infoseeking/sgupta01/VideoMAEv2/outputs/intphys2_main1012_pipeline/aggregated/intphys2_main1012_joined.csv
```

### DreamerV3 and VideoMAEv2 Output Directory

```bash
/gpfs/projects/infoseeking/sgupta01/outputs/
```

Expected files:
```bash
dreamerv3_metrics.json
dreamerv3_metrics.csv
dreamerv3_metrics.txt
dreamerv3_umap.png

videomaev2_metrics.json
videomaev2_metrics.csv
videomaev2_metrics.txt
videomaev2_umap.png
```

V-JEPA outputs follow the V-JEPA workspace naming from `vjepa_plan.md`:
```bash
/mmfs1/home/lindajin/wm_IntPhys2/vjepa2/outputs/vjepa_rq3_metrics.json
/mmfs1/home/lindajin/wm_IntPhys2/vjepa2/outputs/vjepa_rq3_metrics.csv
/mmfs1/home/lindajin/wm_IntPhys2/vjepa2/outputs/vjepa_rq3_metrics.txt
```

---

## 3. Shared Python Packages

All model environments need the usual scientific stack:

```bash
numpy
torch
scikit-learn
matplotlib
umap-learn
decord
```

Install evaluation dependencies in each conda env:
```bash
conda install -y -c conda-forge scikit-learn umap-learn matplotlib
```

For video decoding and checkpoint loading:
```bash
python3 -m pip install decord safetensors imageio pillow
```

---

## 4. V-JEPA Setup and Run

### Working Directory

```bash
cd /mmfs1/home/lindajin/wm_IntPhys2/vjepa2
```

Expected layout:
```bash
/mmfs1/home/lindajin/wm_IntPhys2/vjepa2/
├── setup.sh
├── vjepa_extract_embeddings.py
├── vjepa_rq3_metrics.py
├── run.slurm
├── IntPhys2/
├── checkpoints/
│   └── vith16.pth.tar
└── outputs/
```

### Clone IntPhys2

```bash
cd /mmfs1/home/lindajin/wm_IntPhys2/vjepa2
git clone https://github.com/facebookresearch/IntPhys2.git
```

This provides:
```bash
IntPhys2/prediction_evals/app/vjepa/models/vision_transformer.py
IntPhys2/prediction_evals/src/models/utils/rope.py
IntPhys2/prediction_evals/app/vjepa/transforms.py
```

### Download V-JEPA ViT-H Weights

```bash
mkdir -p /mmfs1/home/lindajin/wm_IntPhys2/vjepa2/checkpoints
wget https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar \
  -O /mmfs1/home/lindajin/wm_IntPhys2/vjepa2/checkpoints/vith16.pth.tar
```

### Environment

Use an env with:
```bash
torch torchvision decord scikit-learn numpy matplotlib umap-learn
```

Create or activate an environment on the `/mmfs1` cluster with those packages:
```bash
conda activate /mmfs1/home/lindajin/wm_IntPhys2/vjepa2/conda_env
python3 -m pip install decord
```

### Extract V-JEPA Embeddings

```bash
cd /mmfs1/home/lindajin/wm_IntPhys2/vjepa2
sbatch run.slurm
```

Interactive GPU run:
```bash
cd /mmfs1/home/lindajin/wm_IntPhys2/vjepa2
python vjepa_extract_embeddings.py
```

Expected bundle:
```bash
outputs/intphys2_main1012_embeddings_bundle.pt
```

Expected bundle keys:
```python
embeddings
frame_embeddings
video_ids
conditions
types
difficulties
cameras
```

### Run V-JEPA Metrics

```bash
cd /mmfs1/home/lindajin/wm_IntPhys2/vjepa2
python vjepa_rq3_metrics.py
```

Expected outputs:
```bash
outputs/vjepa_rq3_metrics.json
outputs/vjepa_rq3_metrics.csv
outputs/vjepa_rq3_metrics.txt
```

---

## 5. DreamerV3 Setup and Run

### Clone DreamerV3

```bash
cd /gpfs/projects/infoseeking
git clone https://github.com/danijar/dreamerv3.git sgupta01/dreamerv3
```

Refer to:
```bash
/gpfs/projects/infoseeking/sgupta01/dreamerv3/README.md
```

### Create Environment

```bash
cd /gpfs/projects/infoseeking
conda create -y -p /gpfs/projects/infoseeking/sgupta01/conda_envs/dreamerv3 python=3.11
conda activate /gpfs/projects/infoseeking/sgupta01/conda_envs/dreamerv3
```

Install DreamerV3:
```bash
cd /gpfs/projects/infoseeking/sgupta01/dreamerv3
pip install -U pip wheel
pip install -U -r requirements.txt
```

Install evaluation dependencies:
```bash
conda install -y -c conda-forge scikit-learn umap-learn matplotlib
python3 -m pip install imageio pillow
```

### Prepare DreamerV3 Checkpoint

Existing checkpoint used in our run:
```bash
/gpfs/projects/infoseeking/sgupta01/dreamerv3_logs/atari100k_pong_50870_20260125_193301/ckpt/20260125T200341F803200/agent.pkl
```

To train a checkpoint yourself:
```bash
cd /gpfs/projects/infoseeking/sgupta01/dreamerv3
python -u -m dreamerv3.main \
  --logdir /gpfs/projects/infoseeking/sgupta01/dreamerv3_logs/my_atari100k_pong \
  --configs atari100k \
  --task atari100k_pong
```

Locate checkpoints:
```bash
find /gpfs/projects/infoseeking/sgupta01/dreamerv3_logs -path '*/ckpt/*/agent.pkl' | head
```

### Extract DreamerV3 Latents

Script:
```bash
/gpfs/projects/infoseeking/maanyacb/extract_latents.py
```

Before running, set constants in the script:
```python
CKPT = '/gpfs/projects/infoseeking/sgupta01/dreamerv3_logs/atari100k_pong_50870_20260125_193301/ckpt/20260125T200341F803200/agent.pkl'
META = '/gpfs/projects/infoseeking/sgupta01/VideoMAEv2/data/IntPhys2/Main/metadata.csv'
VID_DIR = '/gpfs/projects/infoseeking/sgupta01/VideoMAEv2/data/IntPhys2/Main/Videos/'
OUT_DIR = '/gpfs/projects/infoseeking/maanyacb/rq3_outputs/dreamerv3/'
MAX_PER_SPLIT = 506
```

Run extraction:
```bash
cd /gpfs/projects/infoseeking
conda activate /gpfs/projects/infoseeking/sgupta01/conda_envs/dreamerv3
python3 maanyacb/extract_latents.py
```

Expected outputs:
```bash
/gpfs/projects/infoseeking/maanyacb/rq3_outputs/dreamerv3/id_latents.npy
/gpfs/projects/infoseeking/maanyacb/rq3_outputs/dreamerv3/ood_latents.npy
/gpfs/projects/infoseeking/maanyacb/rq3_outputs/dreamerv3/id_metadata.json
/gpfs/projects/infoseeking/maanyacb/rq3_outputs/dreamerv3/ood_metadata.json
```

---

## 6. VideoMAEv2 Setup and Run

### Clone VideoMAEv2

```bash
cd /gpfs/projects/infoseeking
git clone https://github.com/OpenGVLab/VideoMAEv2.git sgupta01/VideoMAEv2
```

Refer to:
```bash
/gpfs/projects/infoseeking/sgupta01/VideoMAEv2/README.md
/gpfs/projects/infoseeking/sgupta01/VideoMAEv2/docs/INSTALL.md
```

### Create Environment

```bash
cd /gpfs/projects/infoseeking
conda create -y -p /gpfs/projects/infoseeking/sgupta01/conda_envs/videomaev2 python=3.10
conda activate /gpfs/projects/infoseeking/sgupta01/conda_envs/videomaev2
```

Install model dependencies:
```bash
cd /gpfs/projects/infoseeking/sgupta01/VideoMAEv2
pip install -U pip wheel
pip install -U -r requirements.txt
```

Install runtime and evaluation dependencies:
```bash
python3 -m pip install decord safetensors huggingface_hub
conda install -y -c conda-forge scikit-learn umap-learn matplotlib
```

### Download VideoMAEv2-Base Weights

```bash
cd /gpfs/projects/infoseeking
mkdir -p /gpfs/projects/infoseeking/sgupta01/hf_cache/models--OpenGVLab--VideoMAEv2-Base

huggingface-cli download OpenGVLab/VideoMAEv2-Base model.safetensors \
  --local-dir /gpfs/projects/infoseeking/sgupta01/hf_cache/models--OpenGVLab--VideoMAEv2-Base/snapshots/78c337a418cc4adaf7f1ff6c7ba343418393966b
```

Expected checkpoint:
```bash
/gpfs/projects/infoseeking/sgupta01/hf_cache/models--OpenGVLab--VideoMAEv2-Base/snapshots/78c337a418cc4adaf7f1ff6c7ba343418393966b/model.safetensors
```

### Prepare IntPhys2 Manifest

Expected data:
```bash
/gpfs/projects/infoseeking/sgupta01/VideoMAEv2/data/IntPhys2/Main/metadata.csv
/gpfs/projects/infoseeking/sgupta01/VideoMAEv2/data/IntPhys2/Main/Videos/
```

Generate manifest if missing:
```bash
cd /gpfs/projects/infoseeking/sgupta01/VideoMAEv2
python3 - <<'PY'
import csv
from pathlib import Path
meta = Path('data/IntPhys2/Main/metadata.csv')
out = Path('manifests/intphys2_main1012.txt')
base = Path('data/IntPhys2/Main')
out.parent.mkdir(parents=True, exist_ok=True)
with meta.open() as f, out.open('w') as g:
    for r in csv.DictReader(f):
        g.write(str((base / r['file_name']).resolve()) + '\n')
print('wrote', out)
PY
```

### Run VideoMAEv2 Inference

```bash
cd /gpfs/projects/infoseeking/sgupta01/VideoMAEv2
conda activate /gpfs/projects/infoseeking/sgupta01/conda_envs/videomaev2
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MANIFEST=/gpfs/projects/infoseeking/sgupta01/VideoMAEv2/manifests/intphys2_main1012.txt
CKPT=/gpfs/projects/infoseeking/sgupta01/hf_cache/models--OpenGVLab--VideoMAEv2-Base/snapshots/78c337a418cc4adaf7f1ff6c7ba343418393966b/model.safetensors
OUT_DIR=/gpfs/projects/infoseeking/sgupta01/VideoMAEv2/outputs/intphys2_main1012_pipeline

python -u scripts/infer_intphys2_subset.py \
  --manifest "$MANIFEST" \
  --output_dir "$OUT_DIR" \
  --per_video_subdir per_video_embeddings \
  --ckpt_path "$CKPT" \
  --model_name vit_base_patch16_224 \
  --clip_len 16 \
  --input_size 224 \
  --max_videos 1012
```

Expected per-video files include:
```bash
embedding
frame_embeddings
```

### Aggregate VideoMAEv2 Bundle

```bash
cd /gpfs/projects/infoseeking/sgupta01/VideoMAEv2
python -u scripts/aggregate_intphys2_embeddings.py \
  --summary_jsonl outputs/intphys2_main1012_pipeline/summary.jsonl \
  --output_dir outputs/intphys2_main1012_pipeline/aggregated \
  --output_prefix intphys2_main1012
```

### Join Labels

```bash
cd /gpfs/projects/infoseeking/sgupta01/VideoMAEv2
python -u scripts/join_intphys2_labels.py \
  --intphys_metadata_csv data/IntPhys2/Main/metadata.csv \
  --subset_metadata_csv outputs/intphys2_main1012_pipeline/aggregated/intphys2_main1012_metadata.csv \
  --output_dir outputs/intphys2_main1012_pipeline/aggregated \
  --output_prefix intphys2_main1012_joined
```

Expected label file:
```bash
/gpfs/projects/infoseeking/sgupta01/VideoMAEv2/outputs/intphys2_main1012_pipeline/aggregated/intphys2_main1012_joined.csv
```

---

## 7. Run Metrics and UMAP

### DreamerV3

```bash
cd /gpfs/projects/infoseeking
conda activate /gpfs/projects/infoseeking/sgupta01/conda_envs/dreamerv3
python3 sgupta01/dreamerv3_metrics.py
python3 sgupta01/dreamerv3_umap.py
```

### VideoMAEv2

```bash
cd /gpfs/projects/infoseeking
conda activate /gpfs/projects/infoseeking/sgupta01/conda_envs/videomaev2
python3 sgupta01/videomaev2_metrics.py
python3 sgupta01/videomaev2_umap.py
```

### V-JEPA

```bash
cd /mmfs1/home/lindajin/wm_IntPhys2/vjepa2
python vjepa_rq3_metrics.py
```

If a V-JEPA UMAP script exists in that workspace:
```bash
cd /mmfs1/home/lindajin/wm_IntPhys2/vjepa2
python vjepa_umap.py
```

---

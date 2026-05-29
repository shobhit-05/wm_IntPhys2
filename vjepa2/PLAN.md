# V-JEPA: Representational Metrics on IntPhys2 Main
## Complete Execution Plan

---

## 0. What This Plan Does

Extracts latent embeddings from the pretrained V-JEPA ViT-H/RoPE encoder on all 1012 videos of
IntPhys2 Main, then computes a unified battery of representational geometry metrics that are
**directly comparable** to the DreamerV3 and VideoMAE baselines.

Uses the official V-JEPA wrapper from the IntPhys2 evaluation codebase
(`app.vjepa.models.vision_transformer.vit_huge_rope`) with the public V-JEPA ViT-H checkpoint
from `https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar`.

```
Stage 1  vjepa_extract_embeddings.py
         → Encodes all 1012 videos (506 Possible + 506 Impossible)
         → Saves outputs/intphys2_main1012_embeddings_bundle.pt
         → Saves outputs/intphys2_main1012_joined.csv

Stage 2  vjepa__metrics.py
         → Loads bundle, splits by type
         → Computes all metrics (geometry + DCI)
         → Saves outputs/vjepa_rq3_metrics.{json,csv,txt}
```

---

## 1. Working Directory & Paths

All work lives under:
```
/mmfs1/home/lindajin/wm_IntPhys2/vjepa2/
├── PLAN.md
├── setup.sh                              ← clone IntPhys2, download checkpoint
├── vjepa_extract_embeddings.py           ← Stage 1
├── vjepa_rq3_metrics.py                  ← Stage 2
├── run.slurm                             ← SLURM job
├── IntPhys2/                             ← cloned from facebookresearch/IntPhys2
│   └── prediction_evals/                 ← added to sys.path
├── checkpoints/
│   └── vith16.pth.tar                    ← V-JEPA ViT-H public checkpoint
└── outputs/
    ├── intphys2_main1012_embeddings_bundle.pt
    ├── intphys2_main1012_joined.csv
    ├── vjepa_rq3_metrics.json
    ├── vjepa_rq3_metrics.csv
    └── vjepa_rq3_metrics.txt
```

### Fixed Paths (shared data)
```python
METADATA_CSV = '/mmfs1/home/lindajin/wm_IntPhys2/videomae/metadata.csv'
VIDEOS_BASE  = '/gpfs/projects/infoseeking/preiyalt/Main/'
# file_name column in metadata.csv is e.g. "Videos/abc123.mp4"
# full path = VIDEOS_BASE + row['file_name']
```

---

## 2. Downloads Required

### 2a. Clone IntPhys2 Repository
```bash
cd /mmfs1/home/lindajin/wm_IntPhys2/vjepa2
git clone https://github.com/facebookresearch/IntPhys2.git
# Provides: IntPhys2/prediction_evals/app/vjepa/models/vision_transformer.py
#           IntPhys2/prediction_evals/src/models/utils/rope.py
#           IntPhys2/prediction_evals/app/vjepa/transforms.py
```

### 2b. V-JEPA ViT-H Checkpoint
```bash
mkdir -p /mmfs1/home/lindajin/wm_IntPhys2/vjepa2/checkpoints
wget https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar \
     -O /mmfs1/home/lindajin/wm_IntPhys2/vjepa2/checkpoints/vith16.pth.tar
# ~2.5 GB — ViT-H/16, 224x224, trained on VideoMix2M
# Checkpoint keys: encoder, target_encoder, predictor, opt, scaler, epoch
```

### 2c. Python Environment
```bash
# Activate whatever env has: torch, torchvision, decord, scikit-learn, numpy
# (same env used for VideoMAE baseline is fine)
pip install decord  # if not already installed
```

---

## 3. Model Details

### Architecture
- **Model:** `vit_huge_rope` from `IntPhys2/prediction_evals/app/vjepa/models/vision_transformer.py`
- **embed_dim:** 1280 (ViT-H)
- **depth:** 32 layers, 16 heads
- **Positional encoding:** RoPE (rotary positional embeddings, no learnable pos_embed)
- **Checkpoint key:** `encoder` (strips `module.` prefix; `pos_embed` key in checkpoint is ignored since RoPE model has none)
- **Input:** `[B, C, T, H, W]` → `[B, N_patches, 1280]`
  - `N = (T/tubelet) × (H/patch) × (W/patch) = 8 × 14 × 14 = 1568`

### Config (from `prediction_evals/evals/intphys2/configs/vjepa_rope.yaml`)
```yaml
resolution: 224
frames_per_clip: 16
patch_size: 16
tubelet_size: 2
uniform_power: true
use_sdpa: true
use_SiLU: true
wide_SiLU: false
is_causal: false
```

---

## 4. Stage 1 — Embedding Extraction (`vjepa_extract_embeddings.py`)

### Preprocessing (ImageNet stats, matches VideoMAE baseline)
```python
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
# Pipeline: decord → uint8 [T,H,W,C] → resize short-side 224 bilinear
#           → center crop 224×224 → float32/255 → normalize
```

### Frame Sampling
```python
from decord import VideoReader, cpu
vr = VideoReader(path, ctx=cpu(0))
indices = np.linspace(0, len(vr)-1, 16, dtype=int)
frames = vr.get_batch(indices).asnumpy()  # [16, H, W, 3] uint8
```

### Embedding
```python
# model output: [1, 1568, 1280]
# mean-pool all patches → [1280] per video
embedding = tokens.mean(dim=1).squeeze(0)

# also save temporal embeddings for gen_gap:
# reshape [1568, 1280] → [8 tubes, 196 spatial, 1280] → mean spatial → [8, 1280]
frame_emb = tokens.squeeze(0).reshape(8, 196, 1280).mean(dim=1)
```

### Output Bundle Format (compatible with `compute_intphys2_main1012_dci_cv.py`)
```python
torch.save({
    'embeddings':       torch.stack(all_embeddings),        # [1012, 1280]
    'frame_embeddings': torch.stack(all_frame_embeddings),  # [1012, 8, 1280]
    'video_ids':        all_video_ids,                       # list[str] (name column)
    'conditions':       all_conditions,                      # list[str]
    'types':            all_types,                           # list[str]
    'difficulties':     all_difficulties,                    # list[str]
    'cameras':          all_cameras,                         # list[str]
}, bundle_path)
```

Also saves `intphys2_main1012_joined.csv` with columns:
`video_id, condition, type, difficulty, camera`
(lowercase column names to match `compute_intphys2_main1012_dci_cv.py`).

---

## 5. Stage 2 — Metrics (`vjepa_rq3_metrics.py`)

Computes all metrics from the DreamerV3 and VideoMAE baselines in a single unified script.

### Data Split
```python
possible_mask   = [t == '1_Possible'   for t in bundle['types']]
impossible_mask = [t == '2_Impossible' for t in bundle['types']]
emb_pos = emb[possible_mask]   # [506, 1280] — geometry metrics scope
emb_imp = emb[impossible_mask] # [506, 1280] — invariance + gen_gap only
```

### Metrics Computed

| # | Metric | Reference Script |
|---|--------|-----------------|
| 1 | Silhouette (GT condition labels, cosine) | dreamerv3 |
| 2 | Silhouette (KMeans k=4, cosine) | dreamerv3 |
| 3 | Centroid gap (inter − intra) | dreamerv3 |
| 4 | NN gap (diff − same cosine dist) | dreamerv3 |
| 5 | KNN-5 accuracy | dreamerv3 |
| 6 | Intervention invariance (random 500 pairs) | dreamerv3 (fixed) |
| 7 | Latent temporal MSE gap (OOD − ID) | dreamerv3 (adapted) |
| 8 | DCI informativeness (LogReg CV per factor) | videomae |
| 9 | DCI disentanglement (full-fit linear probe) | videomae |
| 10 | DCI completeness (full-fit linear probe) | videomae |

### Key Consistency Fixes vs Baselines

| Issue | DreamerV3 | VideoMAE DCI | This script |
|---|---|---|---|
| Silhouette scope | Possible only ✓ | All 1012 ✗ | Possible only ✓ |
| Invariance sampling | First-found, order-dep ✗ | N/A | Random 500 pairs ✓ |
| DCI classifier | GBT 300-subsample ✗ | LogReg StratKFold ✓ | LogReg StratKFold ✓ |
| Gen gap signal | Frame-MSE in RSSM space | N/A | Temporal tube MSE ✓ |

---

## 6. Output Schema (`outputs/vjepa_rq3_metrics.json`)

```json
{
  "model": "vjepa_vith_rope",
  "checkpoint": "vith16.pth.tar",
  "embed_dim": 1280,
  "n_possible": 506,
  "n_impossible": 506,
  "metrics": {
    "silhouette_gt":             <float>,
    "silhouette_kmeans":         <float>,
    "centroid_gap":              <float>,
    "nn_gap":                    <float>,
    "knn5_accuracy":             <float>,
    "invariance_score":          <float>,
    "equiv_cosine_mean":         <float>,
    "diff_cosine_mean":          <float>,
    "latent_temporal_mse_gap":   <float>,
    "id_mean_frame_mse":         <float>,
    "ood_mean_frame_mse":        <float>
  },
  "dci": {
    "disentanglement_fullfit":   <float>,
    "completeness_fullfit":      <float>,
    "informativeness_cv":        <float>,
    "factor_reports": { ... }
  }
}
```

---

## 7. Execution Order

```bash
cd /mmfs1/home/lindajin/wm_IntPhys2/vjepa2

# Step 0: Setup (once)
bash setup.sh

# Step 1: Extract embeddings (GPU required, ~30-60min for 1012 videos)
sbatch run.slurm
# or interactively on GPU node:
python vjepa_extract_embeddings.py

# Step 2: Compute metrics (CPU, ~5-10min)
python vjepa_rq3_metrics.py

# Results at:
# outputs/vjepa_rq3_metrics.json
# outputs/vjepa_rq3_metrics.csv
# outputs/vjepa_rq3_metrics.txt
```

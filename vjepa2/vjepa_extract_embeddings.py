#!/usr/bin/env python3
"""
Stage 1: Extract V-JEPA ViT-H/RoPE embeddings for all IntPhys2 Main videos.

Uses vit_huge_rope from the IntPhys2 prediction_evals wrapper.
Produces an embeddings bundle compatible with vjepa_rq3_metrics.py and
compute_intphys2_main1012_dci_cv.py.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── paths ──────────────────────────────────────────────────────────────────
VJEPA2_DIR   = Path(__file__).resolve().parent
INTPHYS2_REPO = VJEPA2_DIR / 'IntPhys2' / 'prediction_evals'
CHECKPOINT    = VJEPA2_DIR / 'checkpoints' / 'vith16.pth.tar'
METADATA_CSV  = Path('/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/videomae/metadata.csv')
VIDEOS_BASE   = Path('/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/IntPhys2/Main')
OUTPUT_DIR    = VJEPA2_DIR / 'outputs'

NUM_FRAMES   = 16
RESOLUTION   = 224
TUBELET_SIZE = 2
PATCH_SIZE   = 16
# ViT-H spatial patches per temporal tube: (224/16)^2 = 196
# Temporal tubes: 16/2 = 8  →  total patches: 1568
N_TUBES      = NUM_FRAMES // TUBELET_SIZE    # 8
N_SPATIAL    = (RESOLUTION // PATCH_SIZE) ** 2  # 196
EMBED_DIM    = 1280

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

# ── sharding args ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--rank',       type=int, default=0)
parser.add_argument('--world-size', type=int, default=1)
args = parser.parse_args()
RANK       = args.rank
WORLD_SIZE = args.world_size

# ── imports ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(INTPHYS2_REPO))
import app.vjepa.models.vision_transformer as vit_module

try:
    from decord import VideoReader, cpu as decord_cpu
    USE_DECORD = True
except ImportError:
    import cv2
    USE_DECORD = False
    print("WARNING: decord not found, falling back to OpenCV")

# ── device ─────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── build model ────────────────────────────────────────────────────────────
print("Building vit_huge_rope...")
model = vit_module.vit_huge_rope(
    img_size=RESOLUTION,
    num_frames=NUM_FRAMES,
    patch_size=PATCH_SIZE,
    tubelet_size=TUBELET_SIZE,
    uniform_power=True,
    use_sdpa=True,
    use_SiLU=True,
    wide_SiLU=False,
    is_causal=False,
)

print(f"Loading checkpoint: {CHECKPOINT}")
ckpt = torch.load(str(CHECKPOINT), map_location='cpu', weights_only=False)
state = ckpt['encoder']
state = {k.replace('module.', ''): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
if unexpected:
    # pos_embed from non-RoPE checkpoint will appear here — expected
    print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
model.eval().to(device)
print("Model ready.")

# ── transforms ─────────────────────────────────────────────────────────────
def preprocess_frames(frames_np: np.ndarray) -> torch.Tensor:
    """
    frames_np: [T, H, W, 3] uint8 numpy
    returns:   [1, 3, T, H, W] float32 tensor, normalized
    """
    t = torch.from_numpy(frames_np).float() / 255.0  # [T, H, W, 3]
    t = t.permute(0, 3, 1, 2)  # [T, 3, H, W]

    # Resize: shorter side to RESOLUTION
    _, _, h, w = t.shape
    scale = RESOLUTION / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    t = F.interpolate(t, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # Center crop
    ch = (new_h - RESOLUTION) // 2
    cw = (new_w - RESOLUTION) // 2
    t = t[:, :, ch:ch + RESOLUTION, cw:cw + RESOLUTION]

    # Normalize: ImageNet stats
    mean = IMAGENET_MEAN.view(1, 3, 1, 1)
    std  = IMAGENET_STD.view(1, 3, 1, 1)
    t = (t - mean) / std  # [T, 3, H, W]

    # Rearrange to model input: [1, C, T, H, W]
    return t.permute(1, 0, 2, 3).unsqueeze(0)


def load_frames(path: str) -> np.ndarray | None:
    """Load NUM_FRAMES uniformly sampled frames; returns [T,H,W,3] uint8 or None."""
    try:
        if USE_DECORD:
            vr = VideoReader(path, ctx=decord_cpu(0))
            total = len(vr)
            if total < 1:
                return None
            indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
        else:
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 1:
                cap.release()
                return None
            indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if len(frames) != NUM_FRAMES:
                return None
            frames = np.stack(frames)
        return frames.astype(np.uint8)
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
        return None


@torch.no_grad()
def encode_video(frames_np: np.ndarray):
    """
    Returns:
        embedding:      [1280] float32 — mean-pooled over all patch tokens
        frame_embedding:[8, 1280] float32 — mean-pooled over spatial per temporal tube
    """
    x = preprocess_frames(frames_np).to(device)  # [1, 3, 16, 224, 224]
    tokens = model(x)                              # [1, 1568, 1280]
    embedding = tokens.mean(dim=1).squeeze(0).cpu().float()
    # temporal: reshape [1568] → [8 tubes, 196 spatial] → mean spatial
    frame_emb = tokens.squeeze(0).reshape(N_TUBES, N_SPATIAL, EMBED_DIM).mean(dim=1).cpu().float()
    return embedding, frame_emb


# ── load metadata ──────────────────────────────────────────────────────────
print(f"\nLoading metadata from {METADATA_CSV}")
with open(METADATA_CSV, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

# Lowercase the metadata column names for consistency
rows = [{k.lower(): v for k, v in r.items()} for r in rows]

# Normalise: metadata.csv uses 'difficulty' (or 'Difficulty') and 'camera' (or 'Camera')
# after lowercasing: 'difficulty', 'camera', 'condition', 'type', 'name', 'file_name'
print(f"Total rows: {len(rows)} | columns: {list(rows[0].keys())}")

# ── shard rows for this rank ───────────────────────────────────────────────
rows = [r for i, r in enumerate(rows) if i % WORLD_SIZE == RANK]
print(f"Rank {RANK}/{WORLD_SIZE}: processing {len(rows)} videos")

# ── extract embeddings ─────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
shard_suffix = f'_shard{RANK}of{WORLD_SIZE}' if WORLD_SIZE > 1 else ''
BUNDLE_PATH  = OUTPUT_DIR / f'intphys2_main1012_embeddings_bundle{shard_suffix}.pt'
JOINED_CSV   = OUTPUT_DIR / f'intphys2_main1012_joined{shard_suffix}.csv'
SUMMARY_PATH = OUTPUT_DIR / f'extraction_summary{shard_suffix}.json'

all_embeddings    = []
all_frame_embeddings = []
all_video_ids     = []
all_conditions    = []
all_types         = []
all_difficulties  = []
all_cameras       = []
errors            = []

t0 = time.time()
for i, row in enumerate(rows):
    vid_path = str(VIDEOS_BASE / row['file_name'])
    frames = load_frames(vid_path)
    if frames is None:
        errors.append({'row': i, 'file': row['file_name'], 'reason': 'load_failed'})
        continue

    emb, frame_emb = encode_video(frames)
    all_embeddings.append(emb)
    all_frame_embeddings.append(frame_emb)
    all_video_ids.append(row['name'])
    all_conditions.append(row['condition'])
    all_types.append(row['type'])
    all_difficulties.append(row.get('difficulty', ''))
    all_cameras.append(row.get('camera', ''))

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(rows) - i - 1)
        print(f"  rank{RANK} {i+1}/{len(rows)} | elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

print(f"\nProcessed: {len(all_embeddings)} / {len(rows)} videos ({len(errors)} errors)")

# ── save bundle ────────────────────────────────────────────────────────────
bundle = {
    'embeddings':        torch.stack(all_embeddings),         # [N, 1280]
    'frame_embeddings':  torch.stack(all_frame_embeddings),   # [N, 8, 1280]
    'video_ids':         all_video_ids,
    'conditions':        all_conditions,
    'types':             all_types,
    'difficulties':      all_difficulties,
    'cameras':           all_cameras,
}
torch.save(bundle, str(BUNDLE_PATH))
print(f"Bundle saved: {BUNDLE_PATH}  shape={bundle['embeddings'].shape}")

# ── save joined CSV (for compute_intphys2_main1012_dci_cv.py) ──────────────
with open(str(JOINED_CSV), 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['video_id', 'condition', 'type', 'difficulty', 'camera'])
    writer.writeheader()
    for vid_id, cond, typ, diff, cam in zip(
        all_video_ids, all_conditions, all_types, all_difficulties, all_cameras
    ):
        writer.writerow({
            'video_id':   vid_id,
            'condition':  cond,
            'type':       typ,
            'difficulty': diff,
            'camera':     cam,
        })
print(f"Joined CSV saved: {JOINED_CSV}")

# ── save summary ───────────────────────────────────────────────────────────
summary = {
    'n_processed':  len(all_embeddings),
    'n_total':      len(rows),
    'n_errors':     len(errors),
    'embed_dim':    EMBED_DIM,
    'n_tubes':      N_TUBES,
    'n_spatial':    N_SPATIAL,
    'model':        'vjepa_vith_rope',
    'checkpoint':   str(CHECKPOINT),
    'elapsed_s':    round(time.time() - t0, 1),
    'errors':       errors[:20],
}
with open(str(SUMMARY_PATH), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {SUMMARY_PATH}")
print(f"\nTotal elapsed: {time.time()-t0:.0f}s")

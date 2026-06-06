#!/usr/bin/env python3
"""UMAP visualization of V-JEPA embeddings, colored by ID/OOD, condition, difficulty, camera."""

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
import numpy as np
import torch
import umap

ROOT_DIR    = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT_DIR / "VideoMAEv2" / "outputs" / "intphys2_main1012_pipeline" / "aggregated" / "intphys2_main1012_embeddings_bundle.pt"
JOINED_CSV  = ROOT_DIR / "VideoMAEv2" / "outputs" / "intphys2_main1012_pipeline" / "aggregated" / "intphys2_main1012_joined.csv"
OUT_PATH    = ROOT_DIR / "outputs" / "videomaev2_umap.png"

BASE_FONT = 10
SCALE     = 2.5
matplotlib.rcParams.update({"font.size": BASE_FONT * SCALE})

print(f"Loading bundle: {BUNDLE_PATH}")
bundle = torch.load(str(BUNDLE_PATH), map_location="cpu", weights_only=False)
print(f"Loading joined metadata: {JOINED_CSV}")

if "embeddings" in bundle:
    emb = bundle["embeddings"]
    emb = emb.float().numpy() if torch.is_tensor(emb) else np.asarray(emb, dtype=np.float32)
elif "frame_embeddings" in bundle:
    frame_emb = bundle["frame_embeddings"]
    frame_emb = frame_emb.float().numpy() if torch.is_tensor(frame_emb) else np.asarray(frame_emb, dtype=np.float32)
    emb = frame_emb.mean(axis=1)
else:
    raise RuntimeError("Bundle must contain 'embeddings' or 'frame_embeddings'.")

video_ids = list(bundle["video_ids"])
with open(str(JOINED_CSV), "r", encoding="utf-8", newline="") as f:
    joined_rows = list(csv.DictReader(f))
joined_by_id = {r["video_id"]: r for r in joined_rows if r.get("video_id")}

aligned_idx = []
aligned_rows = []
for i, vid in enumerate(video_ids):
    row = joined_by_id.get(vid)
    if row is None:
        continue
    aligned_idx.append(i)
    aligned_rows.append(row)

if not aligned_rows:
    raise RuntimeError("No aligned rows between bundle video_ids and joined CSV.")

emb = emb[np.asarray(aligned_idx, dtype=np.int64)]
types = [
    (r.get("type_raw") or r.get("possible_or_impossible") or "").strip()
    for r in aligned_rows
]
conditions = [(r.get("condition") or "").strip() for r in aligned_rows]
difficulties = [(r.get("difficulty") or "").strip() for r in aligned_rows]
cameras = [(r.get("camera") or "").strip() for r in aligned_rows]

print(f"Embeddings shape: {emb.shape}")

# ── UMAP reduction ────────────────────────────────────────────────────────────
print("Running UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
xy = reducer.fit_transform(emb)
print("UMAP done.")

# ── helper: encode labels → int indices, build discrete colormap ──────────────
def encode(labels, cmap_name="tab10"):
    unique  = sorted(set(labels))
    n       = len(unique)
    lut     = {v: i for i, v in enumerate(unique)}
    indices = np.array([lut[l] for l in labels])
    cmap    = plt.get_cmap(cmap_name, n)
    norm    = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n), ncolors=n)
    return indices, cmap, norm, unique

def encode_custom(labels, color_list):
    unique  = sorted(set(labels))
    lut     = {v: i for i, v in enumerate(unique)}
    indices = np.array([lut[l] for l in labels])
    cmap    = mcolors.ListedColormap(color_list[: len(unique)])
    norm    = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, len(unique)), ncolors=len(unique))
    return indices, cmap, norm, unique

def add_colorbar(fig, ax, sc, unique, n):
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_ticks(np.arange(n))
    cb.set_ticklabels(unique)
    cb.ax.tick_params(labelsize=BASE_FONT * SCALE)

# ── derive labels ─────────────────────────────────────────────────────────────
id_ood = ["ID (Possible)" if "Possible" in t else "OOD (Impossible)" for t in types]

# High-contrast blue vs crimson for ID/OOD
ID_OOD_COLORS = ["#1f77b4", "#d62728"]   # muted blue, vivid red

configs = [
    ("ID / OOD",          id_ood,       None,    ID_OOD_COLORS),
    ("Physics condition", conditions,   "tab10", None),
    ("Difficulty",        difficulties, "Set2",  None),
    ("Camera",            cameras,      "Set1",  None),
]

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(26, 20))
fig.suptitle("VideoMAEv2 — UMAP embeddings", fontsize=BASE_FONT * SCALE * 1.2, y=1.01)

fs = BASE_FONT * SCALE

for ax, (title, labels, cmap_name, custom_colors) in zip(axes.flat, configs):
    if custom_colors is not None:
        indices, cmap, norm, unique = encode_custom(labels, custom_colors)
    else:
        indices, cmap, norm, unique = encode(labels, cmap_name)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=indices, cmap=cmap, norm=norm,
                    s=40, alpha=0.7, linewidths=0)
    ax.set_title(title, fontsize=fs * 1.1)
    ax.set_xlabel("UMAP 1", fontsize=fs)
    ax.set_ylabel("UMAP 2", fontsize=fs)
    ax.tick_params(labelsize=fs * 0.85)
    add_colorbar(fig, ax, sc, unique, len(unique))

plt.tight_layout(h_pad=2.0, w_pad=1.5)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")

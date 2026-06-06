#!/usr/bin/env python3
"""UMAP visualization of V-JEPA embeddings, colored by ID/OOD, condition, difficulty, camera."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
import numpy as np
import torch
import umap

BUNDLE_PATH = Path(__file__).resolve().parent / "outputs" / "intphys2_main1012_embeddings_bundle.pt"
OUT_PATH    = Path(__file__).resolve().parent / "outputs" / "vjepa_umap.png"

BASE_FONT = 10
SCALE     = 2.5
matplotlib.rcParams.update({"font.size": BASE_FONT * SCALE})

print(f"Loading bundle: {BUNDLE_PATH}")
bundle = torch.load(str(BUNDLE_PATH), map_location="cpu", weights_only=False)

emb = bundle["embeddings"]
emb = emb.float().numpy() if torch.is_tensor(emb) else np.asarray(emb, dtype=np.float32)

video_ids    = list(bundle["video_ids"])
types        = list(bundle["types"])
conditions   = list(bundle["conditions"])
difficulties = list(bundle.get("difficulties", [""] * len(video_ids)))
cameras      = list(bundle.get("cameras",      [""] * len(video_ids)))

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
fig.suptitle("V-JEPA ViT-H/RoPE — UMAP embeddings", fontsize=BASE_FONT * SCALE * 1.2, y=1.01)

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

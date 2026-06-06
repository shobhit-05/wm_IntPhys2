#!/usr/bin/env python3
"""
Stage 2: Compute all representational metrics from V-JEPA ViT-H/RoPE embeddings.

Metrics match the DreamerV3 baseline (run_all_metrics.py) and VideoMAE baseline
(compute_intphys2_main1012_dci_cv.py) for direct comparison.

Consistency fixes applied vs baselines:
  - Silhouette: possible-only (not all 1012 as in VideoMAE DCI script)
  - Invariance: random 500 pairs (not first-found order-dependent)
  - DCI classifier: LogReg + StratifiedKFold (not GBT 300-subsample as in DreamerV3)
"""

import csv
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
VJEPA2_DIR  = Path(__file__).resolve().parent
BUNDLE_PATH = VJEPA2_DIR / 'outputs' / 'intphys2_main1012_embeddings_bundle.pt'
OUTPUT_DIR  = VJEPA2_DIR / 'outputs'

# ── load bundle ────────────────────────────────────────────────────────────
print(f"Loading bundle: {BUNDLE_PATH}")
bundle = torch.load(str(BUNDLE_PATH), map_location='cpu', weights_only=False)

emb  = bundle['embeddings']
if torch.is_tensor(emb):
    emb = emb.float().numpy()
else:
    emb = np.asarray(emb, dtype=np.float32)

frame_emb = bundle.get('frame_embeddings')
if frame_emb is not None:
    if torch.is_tensor(frame_emb):
        frame_emb = frame_emb.float().numpy()
    else:
        frame_emb = np.asarray(frame_emb, dtype=np.float32)

video_ids    = list(bundle['video_ids'])
conditions   = list(bundle['conditions'])
types        = list(bundle['types'])
difficulties = list(bundle.get('difficulties', [''] * len(video_ids)))
cameras      = list(bundle.get('cameras',      [''] * len(video_ids)))

n_total, embed_dim = emb.shape
print(f"Embeddings: {emb.shape}")

# ── split possible / impossible ────────────────────────────────────────────
possible_mask   = np.array(['Possible'   in t for t in types])
impossible_mask = np.array(['Impossible' in t for t in types])

emb_pos  = emb[possible_mask]
emb_imp  = emb[impossible_mask]
conds_pos  = [c for c, m in zip(conditions, possible_mask) if m]
conds_imp  = [c for c, m in zip(conditions, impossible_mask) if m]
diffs_pos  = [d for d, m in zip(difficulties, possible_mask) if m]
cams_pos   = [c for c, m in zip(cameras, possible_mask) if m]

n_pos, n_imp = len(emb_pos), len(emb_imp)
print(f"Possible: {n_pos} | Impossible: {n_imp}")

# ── sklearn imports ────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── scale (fit on possible only) ───────────────────────────────────────────
scaler   = StandardScaler()
emb_sc   = scaler.fit_transform(emb_pos)    # [n_pos, 1280] standardized
emb_sc_imp = scaler.transform(emb_imp)

le = LabelEncoder()
y_pos = le.fit_transform(conds_pos)

metrics = {}

# ── 1. Silhouette (GT condition labels) ────────────────────────────────────
print("\n[1/7] Silhouette (GT labels, cosine)...")
sil_gt = silhouette_score(emb_sc, y_pos, metric='cosine')
metrics['silhouette_gt'] = float(sil_gt)
print(f"  silhouette_gt = {sil_gt:.4f}")

# ── 2. Silhouette (KMeans k=4) ─────────────────────────────────────────────
print("[2/7] KMeans silhouette (k=4, cosine)...")
km = KMeans(n_clusters=4, n_init=10, random_state=42)
km_labels = km.fit_predict(emb_sc)
sil_km = silhouette_score(emb_sc, km_labels, metric='cosine')
metrics['silhouette_kmeans'] = float(sil_km)
print(f"  silhouette_kmeans = {sil_km:.4f}")

# ── 3. Centroid gap ────────────────────────────────────────────────────────
print("[3/7] Centroid gap...")
unique_conds = le.classes_
centroids = {c: emb_sc[y_pos == i].mean(axis=0) for i, c in enumerate(unique_conds)}
intra, inter = [], []
for i, c in enumerate(unique_conds):
    pts = emb_sc[y_pos == i]
    if len(pts) > 0:
        intra.append(np.linalg.norm(pts - centroids[c], axis=1).mean())
    for j, c2 in enumerate(unique_conds):
        if c2 != c:
            inter.append(np.linalg.norm(centroids[c] - centroids[c2]))
centroid_gap = float(np.mean(inter) - np.mean(intra))
metrics['centroid_gap']  = centroid_gap
metrics['inter_centroid'] = float(np.mean(inter))
metrics['intra_centroid'] = float(np.mean(intra))
print(f"  centroid_gap = {centroid_gap:.4f}  (inter={np.mean(inter):.4f}  intra={np.mean(intra):.4f})")

# ── 4. NN gap (nearest-neighbour purity) ───────────────────────────────────
print("[4/7] NN gap (cosine, k=1)...")
nbrs = NearestNeighbors(n_neighbors=2, metric='cosine').fit(emb_sc)
dists, idxs = nbrs.kneighbors(emb_sc)
nn_same, nn_diff = [], []
for i in range(len(emb_sc)):
    nn_idx = idxs[i, 1]
    d = dists[i, 1]
    (nn_same if y_pos[nn_idx] == y_pos[i] else nn_diff).append(d)
nn_gap = float(np.mean(nn_diff) - np.mean(nn_same)) if nn_diff and nn_same else 0.0
metrics['nn_gap']      = nn_gap
metrics['nn_same_mean'] = float(np.mean(nn_same)) if nn_same else float('nan')
metrics['nn_diff_mean'] = float(np.mean(nn_diff)) if nn_diff else float('nan')
print(f"  nn_gap = {nn_gap:.4f}  (same={metrics['nn_same_mean']:.4f}  diff={metrics['nn_diff_mean']:.4f})")

# ── 5. KNN-5 accuracy ─────────────────────────────────────────────────────
print("[5/7] KNN-5 accuracy (cosine)...")
nbrs5 = NearestNeighbors(n_neighbors=6, metric='cosine').fit(emb_sc)
_, idxs5 = nbrs5.kneighbors(emb_sc)
correct = sum(
    np.bincount(y_pos[idxs5[i, 1:]]).argmax() == y_pos[i]
    for i in range(len(emb_sc))
)
knn5_acc = float(correct / len(emb_sc))
metrics['knn5_accuracy'] = knn5_acc
print(f"  knn5_accuracy = {knn5_acc:.4f}")

# ── 6. Intervention invariance ─────────────────────────────────────────────
print("[6/7] Intervention invariance (random 500 pairs)...")
rng = np.random.default_rng(42)
pos_by_cond = defaultdict(list)
imp_by_cond = defaultdict(list)
for i, c in enumerate(conds_pos): pos_by_cond[c].append(i)
for i, c in enumerate(conds_imp): imp_by_cond[c].append(i)
all_conds_list = [c for c in pos_by_cond if c in imp_by_cond]

equiv_pairs, diff_pairs = [], []
for _ in range(500):
    c = rng.choice(all_conds_list)
    i = rng.choice(pos_by_cond[c])
    j = rng.choice(imp_by_cond[c])
    equiv_pairs.append((i, j))

for _ in range(500):
    c1, c2 = rng.choice(all_conds_list, size=2, replace=False)
    i = rng.choice(pos_by_cond[c1])
    j = rng.choice(imp_by_cond[c2])
    diff_pairs.append((i, j))

def cos_sim_pairs(pairs, A, B):
    ai = np.stack([A[i] for i, _ in pairs])
    bj = np.stack([B[j] for _, j in pairs])
    an = ai / (np.linalg.norm(ai, axis=1, keepdims=True) + 1e-10)
    bn = bj / (np.linalg.norm(bj, axis=1, keepdims=True) + 1e-10)
    return (an * bn).sum(axis=1)

equiv_sims = cos_sim_pairs(equiv_pairs, emb_pos, emb_imp)
diff_sims  = cos_sim_pairs(diff_pairs,  emb_pos, emb_imp)
invariance = float(equiv_sims.mean() - diff_sims.mean())
metrics['invariance_score']  = invariance
metrics['equiv_cosine_mean'] = float(equiv_sims.mean())
metrics['diff_cosine_mean']  = float(diff_sims.mean())
print(f"  invariance_score = {invariance:.4f}  (equiv={equiv_sims.mean():.4f}  diff={diff_sims.mean():.4f})")

# ── 7. Latent temporal MSE gap (generalization gap) ────────────────────────
print("[7/7] Latent temporal MSE gap...")
if frame_emb is not None:
    fe_pos = frame_emb[possible_mask]    # [n_pos, 8, 1280]
    fe_imp = frame_emb[impossible_mask]  # [n_imp, 8, 1280]
    id_mse  = float(np.mean((fe_pos[:, 1:] - fe_pos[:, :-1]) ** 2))
    ood_mse = float(np.mean((fe_imp[:, 1:] - fe_imp[:, :-1]) ** 2))
    gen_gap = ood_mse - id_mse
    metrics['latent_temporal_mse_gap'] = gen_gap
    metrics['id_mean_frame_mse']       = id_mse
    metrics['ood_mean_frame_mse']      = ood_mse
    print(f"  latent_temporal_mse_gap = {gen_gap:.5f}  (id={id_mse:.5f}  ood={ood_mse:.5f})")
else:
    metrics['latent_temporal_mse_gap'] = None
    metrics['id_mean_frame_mse']       = None
    metrics['ood_mean_frame_mse']      = None
    print("  WARNING: frame_embeddings not in bundle — skipping gen_gap")

# ── DCI: disentanglement / completeness / informativeness ─────────────────
print("\n[DCI] Cross-validated DCI (LogReg, possible only)...")

candidate_factors = [
    ('condition',  'condition',  conds_pos),
    ('type_intra', 'type_intra', None),       # skip — all possible here
    ('difficulty', 'difficulty', diffs_pos),
    ('camera',     'camera',     cams_pos),
]

CV_FOLDS = 5
SEED     = 42

used_factors     = []
skipped_factors  = []
factor_reports   = {}
importance_vecs  = []

for source_col, out_name, values in candidate_factors:
    if values is None:
        skipped_factors.append({'factor': out_name, 'reason': 'not_applicable'})
        continue
    values = [v.strip() if v else '__MISSING__' for v in values]
    counts = Counter(values)
    if len(counts) < 2:
        skipped_factors.append({'factor': out_name, 'reason': 'fewer_than_2_classes', 'counts': dict(counts)})
        continue
    min_count = min(counts.values())
    folds = min(CV_FOLDS, min_count)
    if folds < 2:
        skipped_factors.append({'factor': out_name, 'reason': 'insufficient_samples_for_cv', 'counts': dict(counts)})
        continue

    _le = LabelEncoder()
    y   = _le.fit_transform(values)

    clf_cv = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=SEED)
    cv     = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(clf_cv, emb_sc, y, cv=cv, scoring='accuracy')

    clf_full = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=SEED)
    clf_full.fit(emb_sc, y)
    imp = np.mean(np.abs(clf_full.coef_), axis=0)

    used_factors.append(out_name)
    importance_vecs.append(imp)
    factor_reports[out_name] = {
        'source_column':    source_col,
        'folds_used':       int(folds),
        'class_counts':     dict(counts),
        'n_classes':        int(len(counts)),
        'cv_mean_accuracy': float(np.mean(cv_scores)),
        'cv_std_accuracy':  float(np.std(cv_scores)),
    }
    print(f"  {out_name}: cv_acc={np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}  [{folds} folds]")

# ── DCI aggregate scores ───────────────────────────────────────────────────
def entropy_normalized(probs, axis, log_base):
    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.where(probs > 0, np.log(probs), 0.0)
    ent = -np.sum(probs * logp, axis=axis)
    return ent / np.log(log_base) if log_base > 1 else np.zeros_like(ent)

if used_factors:
    R           = np.stack(importance_vecs, axis=1).astype(np.float64)
    total_imp   = float(R.sum())
    if total_imp > 0:
        row_sums       = R.sum(axis=1, keepdims=True)
        p_f_given_d    = np.divide(R, row_sums, out=np.zeros_like(R), where=row_sums > 0)
        ent_d          = entropy_normalized(p_f_given_d, axis=1, log_base=R.shape[1])
        disent_per_dim = 1.0 - ent_d
        w_d            = row_sums[:, 0] / total_imp
        disentanglement = float(np.sum(w_d * disent_per_dim))

        col_sums       = R.sum(axis=0, keepdims=True)
        p_d_given_f    = np.divide(R, col_sums, out=np.zeros_like(R), where=col_sums > 0)
        ent_f          = entropy_normalized(p_d_given_f, axis=0, log_base=R.shape[0])
        comp_per_factor = 1.0 - ent_f
        w_f             = col_sums[0] / total_imp
        completeness    = float(np.sum(w_f * comp_per_factor))
    else:
        disentanglement = float('nan')
        completeness    = float('nan')

    informativeness = float(np.mean([factor_reports[f]['cv_mean_accuracy'] for f in used_factors]))
else:
    disentanglement = float('nan')
    completeness    = float('nan')
    informativeness = float('nan')

dci = {
    'disentanglement_fullfit': disentanglement,
    'completeness_fullfit':    completeness,
    'informativeness_cv':      informativeness,
    'factors_used':            used_factors,
    'factors_skipped':         skipped_factors,
    'factor_reports':          factor_reports,
}
print(f"\n  DCI disentanglement = {disentanglement:.4f}")
print(f"  DCI completeness    = {completeness:.4f}")
print(f"  DCI informativeness = {informativeness:.4f}")

# ── assemble final results ─────────────────────────────────────────────────
results = {
    'model':       'vjepa_vith_rope',
    'embed_dim':   int(embed_dim),
    'n_possible':  int(n_pos),
    'n_impossible': int(n_imp),
    'metrics':     metrics,
    'dci':         dci,
}

# ── save outputs ───────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
json_out = OUTPUT_DIR / 'vjepa_metrics.json'
csv_out  = OUTPUT_DIR / 'vjepa_metrics.csv'
txt_out  = OUTPUT_DIR / 'vjepa_metrics.txt'

with open(str(json_out), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

flat = {
    'model':             'vjepa_vith_rope',
    'embed_dim':         embed_dim,
    'n_possible':        n_pos,
    'n_impossible':      n_imp,
    **{k: v for k, v in metrics.items()},
    'dci_disentanglement': disentanglement,
    'dci_completeness':    completeness,
    'dci_informativeness': informativeness,
    'factors_used':        '|'.join(used_factors),
}
for f_name in used_factors:
    flat[f'cv_acc_{f_name}'] = factor_reports[f_name]['cv_mean_accuracy']
    flat[f'cv_std_{f_name}'] = factor_reports[f_name]['cv_std_accuracy']

with open(str(csv_out), 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
    writer.writeheader()
    writer.writerow(flat)

with open(str(txt_out), 'w', encoding='utf-8') as f:
    f.write(f"Model: vjepa_vith_rope  |  embed_dim={embed_dim}\n")
    f.write(f"n_possible={n_pos}  n_impossible={n_imp}\n\n")
    f.write("=== Geometry Metrics (possible split) ===\n")
    for k, v in metrics.items():
        f.write(f"  {k:<35s} {v}\n")
    f.write("\n=== DCI Scores ===\n")
    f.write(f"  {'disentanglement_fullfit':<35s} {disentanglement:.6f}\n")
    f.write(f"  {'completeness_fullfit':<35s} {completeness:.6f}\n")
    f.write(f"  {'informativeness_cv':<35s} {informativeness:.6f}\n")
    f.write(f"\n  factors_used: {used_factors}\n")
    f.write("  per-factor CV accuracy:\n")
    for fname in used_factors:
        rp = factor_reports[fname]
        f.write(f"    {fname:<20s} mean={rp['cv_mean_accuracy']:.4f}  std={rp['cv_std_accuracy']:.4f}  folds={rp['folds_used']}\n")
    if skipped_factors:
        f.write(f"\n  skipped: {json.dumps(skipped_factors)}\n")

print(f"\nSaved: {json_out}")
print(f"Saved: {csv_out}")
print(f"Saved: {txt_out}")

print("\n" + "=" * 55)
print("ALL METRICS COMPLETE")
print("=" * 55)
print(json.dumps(results['metrics'], indent=2))
print(json.dumps({'dci_disentanglement': disentanglement,
                  'dci_completeness': completeness,
                  'dci_informativeness': informativeness}, indent=2))

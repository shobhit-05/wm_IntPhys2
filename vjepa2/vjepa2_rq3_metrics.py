import numpy as np
import pandas as pd
import json
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

OUT_DIR  = '/gpfs/projects/infoseeking/preiyalt/rq3_outputs/vjepa2/'
METADATA = '/gpfs/projects/infoseeking/preiyalt/Main/metadata.csv'

print("Loading saved latents...")
id_latents_arr  = np.load(OUT_DIR + 'id_latents.npy')
ood_latents_arr = np.load(OUT_DIR + 'ood_latents.npy')
id_errors_arr   = np.load(OUT_DIR + 'id_errors.npy')
ood_errors_arr  = np.load(OUT_DIR + 'ood_errors.npy')
print(f"ID latents: {id_latents_arr.shape}, OOD latents: {ood_latents_arr.shape}")

# flatten to 2D for sklearn
id_flat  = id_latents_arr.reshape(len(id_latents_arr), -1)
ood_flat = ood_latents_arr.reshape(len(ood_latents_arr), -1)
all_latents = np.concatenate([id_flat, ood_flat], axis=0)

# get condition labels from metadata
df = pd.read_csv(METADATA)
id_rows  = df[df['type'] == '1_Possible'].head(len(id_latents_arr))
ood_rows = df[df['type'] == '2_Impossible'].head(len(ood_latents_arr))
all_conditions = list(id_rows['condition']) + list(ood_rows['condition'])
unique_conds = list(set(all_conditions))
cond_to_int  = {c: i for i, c in enumerate(unique_conds)}
gt_labels    = np.array([cond_to_int[c] for c in all_conditions])

# --- METRIC 1: Generalization gap ---
gen_gap = float(ood_errors_arr.mean() - id_errors_arr.mean())
print(f"\nGeneralization Gap: {gen_gap:.6f}")
print(f"  ID mean latent change:  {id_errors_arr.mean():.6f}")
print(f"  OOD mean latent change: {ood_errors_arr.mean():.6f}")

# --- METRIC 2: Silhouette GT ---
print("\nComputing silhouette (GT conditions)...")
sil_gt = float(silhouette_score(all_latents, gt_labels))
print(f"Silhouette Score (GT physics conditions): {sil_gt:.4f}")

# --- METRIC 3: Silhouette KMeans ---
print("Computing silhouette (KMeans k=4)...")
kmeans    = KMeans(n_clusters=4, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(all_latents)
sil_km    = float(silhouette_score(all_latents, km_labels))
print(f"Silhouette Score (KMeans k=4): {sil_km:.4f}")

# --- METRIC 4: Intervention invariance ---
print("Computing intervention invariance...")
equiv_sims, diff_sims = [], []
id_conditions  = list(id_rows['condition'])
ood_conditions = list(ood_rows['condition'])
for i in range(len(id_flat)):
    for j in range(len(ood_flat)):
        sim = float(cosine_similarity(
            id_flat[i].reshape(1,-1),
            ood_flat[j].reshape(1,-1)
        )[0][0])
        if id_conditions[i] == ood_conditions[j]:
            equiv_sims.append(sim)
        else:
            diff_sims.append(sim)
        if len(equiv_sims) >= 5000 and len(diff_sims) >= 5000:
            break
    if len(equiv_sims) >= 5000 and len(diff_sims) >= 5000:
        break

invariance_score = float(np.mean(equiv_sims) - np.mean(diff_sims))
print(f"Intervention Invariance Score: {invariance_score:.4f}")
print(f"  Physically-equivalent cosine sim: {np.mean(equiv_sims):.4f}")
print(f"  Physically-different cosine sim:  {np.mean(diff_sims):.4f}")

# --- Save summary ---
summary = {
    'generalization_gap':       gen_gap,
    'id_mean_latent_change':    float(id_errors_arr.mean()),
    'ood_mean_latent_change':   float(ood_errors_arr.mean()),
    'silhouette_gt_conditions': sil_gt,
    'silhouette_kmeans_k4':     sil_km,
    'invariance_score':         invariance_score,
    'equiv_cosine_sim':         float(np.mean(equiv_sims)),
    'diff_cosine_sim':          float(np.mean(diff_sims)),
    'n_id':                     len(id_latents_arr),
    'n_ood':                    len(ood_latents_arr),
}
with open(OUT_DIR + 'vjepa2_rq3_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSummary saved.")
print(json.dumps(summary, indent=2))

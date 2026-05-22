import torch
import cv2
import numpy as np
import pandas as pd
import os
import json
import sys
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
sys.path.append('/gpfs/home/preiyalt/vjepa2')

from src.models.vision_transformer import vit_large

CHECKPOINT = '/gpfs/projects/infoseeking/preiyalt/checkpoints/vitl.pt'
VIDEOS_BASE = '/gpfs/projects/infoseeking/preiyalt/Main/'
METADATA    = '/gpfs/projects/infoseeking/preiyalt/Main/metadata.csv'
OUT_DIR     = '/gpfs/projects/infoseeking/preiyalt/rq3_outputs/vjepa2/'
SEQ_LEN     = 16
RESOLUTION  = 224

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading V-JEPA 2 model...")
model = vit_large(
    img_size=224,
    patch_size=16,
    tubelet_size=2,
    uniform_power=True,
    use_rope=True,
)
checkpoint = torch.load(CHECKPOINT, map_location=device)
state_dict = checkpoint.get('target_encoder', checkpoint)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()
print("Model loaded.")

def load_frames(video_path, num_frames=SEQ_LEN, resolution=RESOLUTION):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (resolution, resolution))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    if len(frames) < num_frames:
        return None
    return frames

def get_latent(frames):
    tensor = torch.tensor(np.stack(frames).astype(np.float32) / 255.0)
    tensor = tensor.permute(0, 3, 1, 2).to(device)
    all_features = []
    with torch.no_grad():
        for frame in tensor:
            feat = model(frame.unsqueeze(0))
            all_features.append(feat.cpu())
    return torch.stack(all_features).mean(0).squeeze().numpy()

def latent_change(frames):
    tensor = torch.tensor(np.stack(frames).astype(np.float32) / 255.0)
    tensor = tensor.permute(0, 3, 1, 2).to(device)
    feats = []
    with torch.no_grad():
        for frame in tensor:
            feat = model(frame.unsqueeze(0))
            feats.append(feat.cpu().squeeze().numpy())
    feats = np.stack(feats)
    return float(np.mean((feats[1:] - feats[:-1])**2))

df = pd.read_csv(METADATA)
id_rows  = df[df['type'] == '1_Possible'].head(506)
ood_rows = df[df['type'] == '2_Impossible'].head(506)

print(f"ID videos: {len(id_rows)}, OOD videos: {len(ood_rows)}")

id_latents, id_errors, id_meta   = [], [], []
ood_latents, ood_errors, ood_meta = [], [], []

for split_name, rows, latents_list, errors_list, meta_list in [
    ('ID',  id_rows,  id_latents,  id_errors,  id_meta),
    ('OOD', ood_rows, ood_latents, ood_errors, ood_meta),
]:
    print(f"\nProcessing {split_name} ({len(rows)} videos)...")
    for i, (_, row) in enumerate(rows.iterrows()):
        vid_path = os.path.join(VIDEOS_BASE, row['file_name'])
        if not os.path.exists(vid_path):
            continue
        frames = load_frames(vid_path)
        if frames is None:
            continue
        latent = get_latent(frames)
        error  = latent_change(frames)
        latents_list.append(latent)
        errors_list.append(error)
        meta_list.append({
            'condition': row.get('condition', 'unknown'),
            'type':      row['type'],
        })
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(rows)} done")

id_latents_arr  = np.stack(id_latents)
ood_latents_arr = np.stack(ood_latents)
id_errors_arr   = np.array(id_errors)
ood_errors_arr  = np.array(ood_errors)

np.save(OUT_DIR + 'id_latents.npy',  id_latents_arr)
np.save(OUT_DIR + 'ood_latents.npy', ood_latents_arr)
np.save(OUT_DIR + 'id_errors.npy',   id_errors_arr)
np.save(OUT_DIR + 'ood_errors.npy',  ood_errors_arr)
print(f"\nLatents saved. ID: {id_latents_arr.shape}, OOD: {ood_latents_arr.shape}")

# --- METRIC 1: Generalization gap ---
gen_gap = float(ood_errors_arr.mean() - id_errors_arr.mean())
print(f"\nGeneralization Gap: {gen_gap:.6f}")
print(f"  ID mean latent change:  {id_errors_arr.mean():.6f}")
print(f"  OOD mean latent change: {ood_errors_arr.mean():.6f}")

# --- METRIC 2 & 3: Silhouette scores ---
all_latents_3d = np.concatenate([id_latents_arr, ood_latents_arr], axis=0)
all_latents = all_latents_3d.reshape(len(all_latents_3d), -1)

# GT condition labels
id_conditions  = [m['condition'] for m in id_meta]
ood_conditions = [m['condition'] for m in ood_meta]
all_conditions = id_conditions + ood_conditions
unique_conds   = list(set(all_conditions))
cond_to_int    = {c: i for i, c in enumerate(unique_conds)}
gt_labels      = np.array([cond_to_int[c] for c in all_conditions])

sil_gt = float(silhouette_score(all_latents, gt_labels))
print(f"\nSilhouette Score (GT physics conditions): {sil_gt:.4f}")

# KMeans k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(all_latents)
sil_km = float(silhouette_score(all_latents, km_labels))
print(f"Silhouette Score (KMeans k=4): {sil_km:.4f}")

# --- METRIC 4: Intervention invariance ---
# cosine similarity between physically-equivalent pairs (same condition, ID vs OOD)
# vs physically-different pairs (different conditions)
from sklearn.metrics.pairwise import cosine_similarity

equiv_sims, diff_sims = [], []
for i, id_m in enumerate(id_meta):
    for j, ood_m in enumerate(ood_meta):
        sim = float(cosine_similarity(
            id_latents_arr[i].reshape(1,-1),
            ood_latents_arr[j].reshape(1,-1)
        )[0][0])
        if id_m['condition'] == ood_m['condition']:
            equiv_sims.append(sim)
        else:
            diff_sims.append(sim)

# sample to avoid memory issues
equiv_sims = equiv_sims[:5000]
diff_sims  = diff_sims[:5000]

invariance_score = float(np.mean(equiv_sims) - np.mean(diff_sims))
print(f"\nIntervention Invariance Score: {invariance_score:.4f}")
print(f"  Physically-equivalent cosine sim: {np.mean(equiv_sims):.4f}")
print(f"  Physically-different cosine sim:  {np.mean(diff_sims):.4f}")

# --- Save summary ---
summary = {
    'generalization_gap':         gen_gap,
    'id_mean_latent_change':      float(id_errors_arr.mean()),
    'ood_mean_latent_change':     float(ood_errors_arr.mean()),
    'silhouette_gt_conditions':   sil_gt,
    'silhouette_kmeans_k4':       sil_km,
    'invariance_score':           invariance_score,
    'equiv_cosine_sim':           float(np.mean(equiv_sims)),
    'diff_cosine_sim':            float(np.mean(diff_sims)),
    'n_id':                       len(id_latents),
    'n_ood':                      len(ood_latents),
}
with open(OUT_DIR + 'vjepa2_rq3_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSummary saved.")
print(json.dumps(summary, indent=2))

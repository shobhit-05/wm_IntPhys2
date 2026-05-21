import sys, pickle, csv, json, time
sys.path.insert(0, '/gpfs/projects/infoseeking/sgupta01/dreamerv3')
import numpy as np
import jax.numpy as jnp
import ninjax as nj
import elements
import imageio
from PIL import Image
from dreamerv3.rssm import RSSM, Encoder

# ── paths ──────────────────────────────────────────────────────────────
CKPT    = '/gpfs/projects/infoseeking/sgupta01/dreamerv3_logs/latest_atari100k_pong/ckpt/20260125T200341F803200/agent.pkl'
META    = '/gpfs/projects/infoseeking/preiyalt/Main/metadata.csv'
VID_DIR = '/gpfs/projects/infoseeking/preiyalt/Main/Videos/'
OUT     = '/gpfs/projects/infoseeking/maanyacb/rq3_outputs/dreamerv3/'
SEQ_LEN = 16
IMG_SIZE = 64
MAX_PER_SPLIT = 506   # full dataset

# ── load model ─────────────────────────────────────────────────────────
print("Loading checkpoint...")
with open(CKPT, 'rb') as f:
    ckpt = pickle.load(f)
pretrained = ckpt['params']

act_space = {'action': elements.Space(np.int32, (), 0, 6)}
obs_space = {'image': elements.Space(np.uint8, (IMG_SIZE, IMG_SIZE, 3), 0, 255)}
encoder = Encoder(obs_space, depth=64, mults=(2,3,4,4), act='silu', norm='rms', kernel=5, name='enc')
rssm    = RSSM(act_space, deter=8192, hidden=1024, stoch=32, classes=64, blocks=8, act='silu', norm='rms', name='dyn')

dummy_imgs  = jnp.zeros((1, SEQ_LEN, IMG_SIZE, IMG_SIZE, 3), dtype=jnp.uint8)
dummy_obs   = {'image': dummy_imgs}
dummy_acts  = {'action': jnp.zeros((1, SEQ_LEN), dtype=jnp.int32)}
dummy_reset = jnp.zeros((1, SEQ_LEN), dtype=bool).at[:, 0].set(True)

enc_state = nj.init(encoder.__call__)({}, {}, dummy_obs, dummy_reset, training=False, seed=0)
rssm_state, carry0 = nj.pure(rssm.initial)({}, 1)
rssm_state, _ = nj.pure(rssm.observe)(rssm_state, carry0, jnp.zeros((1,SEQ_LEN,4096)), dummy_acts, dummy_reset, training=False, seed=0, create=True)
enc_state.update({k:v for k,v in pretrained.items() if k.startswith('enc/')})
rssm_state.update({k:v for k,v in pretrained.items() if k.startswith('dyn/')})
print("Weights loaded.")

# ── helpers ────────────────────────────────────────────────────────────
def load_frames(path):
    reader = imageio.get_reader(path)
    frames = []
    for i, frame in enumerate(reader):
        img = np.array(Image.fromarray(frame).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        frames.append(img)
        if i >= SEQ_LEN - 1:
            break
    reader.close()
    return np.stack(frames).astype(np.uint8) if len(frames) == SEQ_LEN else None

def run_model(frames):
    imgs  = jnp.array(frames)[None]
    obs   = {'image': imgs}
    acts  = {'action': jnp.zeros((1, SEQ_LEN), dtype=jnp.int32)}
    reset = jnp.zeros((1, SEQ_LEN), dtype=bool).at[:, 0].set(True)
    es, (_, _, tokens) = nj.pure(encoder.__call__)(enc_state, {}, obs, reset, training=False, seed=0)
    rs, carry0 = nj.pure(rssm.initial)(rssm_state, 1)
    rs, (_, _, feat) = nj.pure(rssm.observe)(rs, carry0, tokens, acts, reset, training=False, seed=0, create=True)
    deter = np.array(feat['deter'][0])
    stoch = np.array(feat['stoch'][0]).reshape(SEQ_LEN, -1)
    return np.concatenate([deter, stoch], axis=-1).astype(np.float32)  # [T, 10240]

# ── extract latents ────────────────────────────────────────────────────
with open(META) as f:
    all_rows = list(csv.DictReader(f))

id_rows  = [r for r in all_rows if 'Possible'   in r['type']][:MAX_PER_SPLIT]
ood_rows = [r for r in all_rows if 'Impossible' in r['type']][:MAX_PER_SPLIT]

results = {}
for split, rows in [('id', id_rows), ('ood', ood_rows)]:
    print(f"\nExtracting {split.upper()} ({len(rows)} videos)...")
    latents, metadata, errors = [], [], []
    t0 = time.time()
    for i, row in enumerate(rows):
        frames = load_frames(VID_DIR + row['file_name'].split('/')[-1])
        if frames is None:
            continue
        lat = run_model(frames)
        err = float(np.mean((lat[1:] - lat[:-1])**2))
        latents.append(lat)
        errors.append(err)
        metadata.append({k: row[k] for k in ['name','game_name','condition','type','Difficulty']})
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(rows)} | elapsed: {time.time()-t0:.0f}s")
    latents_arr = np.stack(latents)
    errors_arr  = np.array(errors, dtype=np.float32)
    np.save(OUT + f'{split}_latents.npy', latents_arr)
    np.save(OUT + f'{split}_errors.npy',  errors_arr)
    with open(OUT + f'{split}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    results[split] = {'latents': latents_arr, 'errors': errors_arr, 'metadata': metadata}
    print(f"[{split.upper()}] shape={latents_arr.shape} mean_err={errors_arr.mean():.5f}")

# ── load flat latents ──────────────────────────────────────────────────
id_flat   = results['id']['latents'].mean(axis=1)    # [N, 10240]
ood_flat  = results['ood']['latents'].mean(axis=1)
id_meta   = results['id']['metadata']
ood_meta  = results['ood']['metadata']
id_conds  = [m['condition'] for m in id_meta]
ood_conds = [m['condition'] for m in ood_meta]
label_map = {'solidity':0,'permanence':1,'continuity':2,'immutability':3}
y_id  = np.array([label_map[c] for c in id_conds])
y_ood = np.array([label_map[c] for c in ood_conds])

print(f"\nID flat: {id_flat.shape} | OOD flat: {ood_flat.shape}")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

scaler   = StandardScaler()
id_sc    = scaler.fit_transform(id_flat)
ood_sc   = scaler.transform(ood_flat)
all_flat = np.concatenate([id_flat, ood_flat], axis=0)
all_sc   = scaler.transform(all_flat)
all_y    = np.concatenate([y_id, y_ood])

metrics = {}

# ── 1. Silhouette score (GT labels) ───────────────────────────────────
print("\n[1/6] Silhouette (GT labels)...")
sil_gt = silhouette_score(id_sc, y_id, metric='cosine')
metrics['silhouette_gt'] = float(sil_gt)
print(f"  silhouette_gt = {sil_gt:.4f}")

# ── 2. K-Means silhouette ─────────────────────────────────────────────
print("[2/6] KMeans silhouette...")
km = KMeans(n_clusters=4, n_init=10, random_state=42)
km_labels = km.fit_predict(id_sc)
sil_km = silhouette_score(id_sc, km_labels, metric='cosine')
metrics['silhouette_kmeans'] = float(sil_km)
print(f"  silhouette_kmeans = {sil_km:.4f}")

# ── 3. Centroid gap ───────────────────────────────────────────────────
print("[3/6] Centroid gap...")
centroids = {}
for cond, idx in label_map.items():
    mask = y_id == idx
    if mask.sum() > 0:
        centroids[cond] = id_sc[mask].mean(axis=0)
intra, inter = [], []
for cond, idx in label_map.items():
    if cond not in centroids:
        continue
    mask = y_id == idx
    if mask.sum() < 2:
        continue
    pts = id_sc[mask]
    intra.append(np.mean(np.linalg.norm(pts - centroids[cond], axis=1)))
    for cond2, idx2 in label_map.items():
        if cond2 != cond and cond2 in centroids:
            inter.append(np.linalg.norm(centroids[cond] - centroids[cond2]))
centroid_gap = float(np.mean(inter) - np.mean(intra))
metrics['centroid_gap'] = centroid_gap
print(f"  centroid_gap = {centroid_gap:.4f} (inter={np.mean(inter):.4f} intra={np.mean(intra):.4f})")

# ── 4. NN gap (nearest neighbour purity) ──────────────────────────────
print("[4/6] NN gap...")
nbrs = NearestNeighbors(n_neighbors=2, metric='cosine').fit(id_sc)
dists, idxs = nbrs.kneighbors(id_sc)
nn_same, nn_diff = [], []
for i in range(len(id_sc)):
    nn_idx = idxs[i, 1]  # nearest neighbour (skip self)
    d = dists[i, 1]
    if y_id[nn_idx] == y_id[i]:
        nn_same.append(d)
    else:
        nn_diff.append(d)
nn_gap = float(np.mean(nn_diff) - np.mean(nn_same))
metrics['nn_gap'] = nn_gap
print(f"  nn_gap = {nn_gap:.4f} (same={np.mean(nn_same):.4f} diff={np.mean(nn_diff):.4f})")

# ── 5. K=5 nearest neighbours accuracy ───────────────────────────────
print("[5/6] K=5 NN accuracy...")
nbrs5 = NearestNeighbors(n_neighbors=6, metric='cosine').fit(id_sc)
dists5, idxs5 = nbrs5.kneighbors(id_sc)
correct = 0
for i in range(len(id_sc)):
    nn_labels = y_id[idxs5[i, 1:]]  # skip self
    pred = np.bincount(nn_labels).argmax()
    if pred == y_id[i]:
        correct += 1
knn5_acc = float(correct / len(id_sc))
metrics['knn5_accuracy'] = knn5_acc
print(f"  knn5_accuracy = {knn5_acc:.4f}")

# ── 6. Intervention invariance proxy ─────────────────────────────────
print("[6/6] Invariance proxy...")
def cos_sim_paired(A, B):
    A = np.array(A); B = np.array(B)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return (An * Bn).sum(axis=1)

equiv_id, equiv_ood, diff_id, diff_ood = [], [], [], []
for i, ic in enumerate(id_conds):
    for j, oc in enumerate(ood_conds):
        if ic == oc and len(equiv_id) < 500:
            equiv_id.append(id_flat[i]); equiv_ood.append(ood_flat[j])
        elif ic != oc and len(diff_id) < 500:
            diff_id.append(id_flat[i]); diff_ood.append(ood_flat[j])

equiv_sims = cos_sim_paired(equiv_id, equiv_ood)
diff_sims  = cos_sim_paired(diff_id,  diff_ood)
invariance = float(equiv_sims.mean() - diff_sims.mean())
metrics['invariance_score']   = invariance
metrics['equiv_cosine_mean']  = float(equiv_sims.mean())
metrics['diff_cosine_mean']   = float(diff_sims.mean())
print(f"  invariance_score = {invariance:.4f}")
print(f"  equiv_sim={equiv_sims.mean():.4f}  diff_sim={diff_sims.mean():.4f}")

# ── 7. Cross-validated DCI proxy ──────────────────────────────────────
print("\n[7/7] Cross-validated DCI (GBT predictor)...")
# DCI: how well can we predict each factor from latents?
# Use cross-validated accuracy of GBT predicting physics condition
# as a proxy for informativeness (key DCI component)
# Subsample for speed on login node
max_dci = min(300, len(all_sc))
idx_sub = np.random.choice(len(all_sc), max_dci, replace=False)
X_sub = all_sc[idx_sub]
y_sub = all_y[idx_sub]

clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
cv_scores = cross_val_score(clf, X_sub, y_sub, cv=5, scoring='accuracy')
dci_proxy = float(cv_scores.mean())
metrics['dci_cv_accuracy']    = dci_proxy
metrics['dci_cv_std']         = float(cv_scores.std())
print(f"  dci_cv_accuracy = {dci_proxy:.4f} ± {cv_scores.std():.4f}")

# ── generalization gap ────────────────────────────────────────────────
gen_gap = float(results['ood']['errors'].mean() - results['id']['errors'].mean())
metrics['generalization_gap'] = gen_gap
metrics['id_mean_err']        = float(results['id']['errors'].mean())
metrics['ood_mean_err']       = float(results['ood']['errors'].mean())
metrics['n_id']               = len(id_meta)
metrics['n_ood']              = len(ood_meta)
metrics['latent_dim']         = 10240

# ── save ──────────────────────────────────────────────────────────────
with open(OUT + 'dreamerv3_metrics_summary.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "="*50)
print("ALL METRICS COMPLETE")
print("="*50)
print(json.dumps(metrics, indent=2))
print(f"\nSaved to {OUT}dreamerv3_metrics_summary.json")

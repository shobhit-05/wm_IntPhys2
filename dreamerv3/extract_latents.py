import sys, pickle, csv, json
sys.path.insert(0, '/gpfs/projects/infoseeking/sgupta01/dreamerv3')
import numpy as np
import jax.numpy as jnp
import ninjax as nj
import elements
import imageio
from PIL import Image
from dreamerv3.rssm import RSSM, Encoder

CKPT = '/gpfs/projects/infoseeking/sgupta01/dreamerv3_logs/latest_atari100k_pong/ckpt/20260125T200341F803200/agent.pkl'
META = '/gpfs/projects/infoseeking/preiyalt/Main/metadata.csv'
VID_DIR = '/gpfs/projects/infoseeking/preiyalt/Main/Videos/'
OUT_DIR = '/gpfs/projects/infoseeking/maanyacb/rq3_outputs/dreamerv3/'
SEQ_LEN = 16
IMG_SIZE = 64
MAX_PER_SPLIT = 50  # small first run; increase later

print("Loading checkpoint...")
with open(CKPT, 'rb') as f:
    ckpt = pickle.load(f)
pretrained = ckpt['params']

act_space = {'action': elements.Space(np.int32, (), 0, 6)}
obs_space = {'image': elements.Space(np.uint8, (IMG_SIZE, IMG_SIZE, 3), 0, 255)}
encoder = Encoder(obs_space, depth=64, mults=(2,3,4,4), act='silu', norm='rms', kernel=5, name='enc')
rssm    = RSSM(act_space, deter=8192, hidden=1024, stoch=32, classes=64, blocks=8, act='silu', norm='rms', name='dyn')

# Init and load pretrained weights
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

def load_video_frames(path, seq_len, img_size):
    reader = imageio.get_reader(path)
    frames = []
    for i, frame in enumerate(reader):
        img = np.array(Image.fromarray(frame).resize((img_size, img_size), Image.BILINEAR))
        frames.append(img)
        if i >= seq_len - 1:
            break
    reader.close()
    if len(frames) < seq_len:
        return None
    return np.stack(frames).astype(np.uint8)  # [T, H, W, 3]

def run_model(frames, enc_state, rssm_state):
    imgs = jnp.array(frames)[None]  # [1, T, H, W, 3]
    obs  = {'image': imgs}
    acts = {'action': jnp.zeros((1, SEQ_LEN), dtype=jnp.int32)}
    reset = jnp.zeros((1, SEQ_LEN), dtype=bool).at[:, 0].set(True)

    enc_state, (_, _, tokens) = nj.pure(encoder.__call__)(enc_state, {}, obs, reset, training=False, seed=0)
    rssm_state2, carry0 = nj.pure(rssm.initial)(rssm_state, 1)
    rssm_state2, (_, _, feat) = nj.pure(rssm.observe)(rssm_state2, carry0, tokens, acts, reset, training=False, seed=0, create=True)

    deter = np.array(feat['deter'][0])           # [T, 8192]
    stoch = np.array(feat['stoch'][0])           # [T, 32, 64]
    stoch_flat = stoch.reshape(SEQ_LEN, -1)      # [T, 2048]
    latent = np.concatenate([deter, stoch_flat], axis=-1)  # [T, 10240]
    return latent

with open(META) as f:
    all_rows = list(csv.DictReader(f))

id_rows  = [r for r in all_rows if 'Possible'   in r['type']][:MAX_PER_SPLIT]
ood_rows = [r for r in all_rows if 'Impossible' in r['type']][:MAX_PER_SPLIT]

results = {}
for split_name, rows in [('id', id_rows), ('ood', ood_rows)]:
    print(f"\nProcessing {split_name.upper()} ({len(rows)} videos)...")
    latents, metadata, errors = [], [], []

    for i, row in enumerate(rows):
        vid_path = VID_DIR + row['file_name'].split('/')[-1]
        frames = load_video_frames(vid_path, SEQ_LEN, IMG_SIZE)
        if frames is None:
            continue

        latent = run_model(frames, enc_state, rssm_state)  # [T, 10240]

        # Reconstruction error: compare frame t prediction vs frame t+1
        # Use mean latent norm change as proxy (decoder not loaded)
        lat_diff = float(np.mean((latent[1:] - latent[:-1])**2))

        latents.append(latent)
        metadata.append({k: row[k] for k in ['name','game_name','condition','type','Difficulty']})
        errors.append(lat_diff)

        if (i+1) % 10 == 0 and errors:
            print(f"  {i+1}/{len(rows)} done | mean latent change: {np.mean(errors):.5f}")

    latents_arr = np.stack(latents)   # [N, T, 10240]
    errors_arr  = np.array(errors)

    print(f"[{split_name.upper()}] Latents: {latents_arr.shape} | Mean error: {errors_arr.mean():.5f}")
    results[split_name] = {'latents': latents_arr, 'errors': errors_arr, 'metadata': metadata}
    np.save(OUT_DIR + f'{split_name}_latents.npy', latents_arr)
    np.save(OUT_DIR + f'{split_name}_errors.npy',  errors_arr)
    with open(OUT_DIR + f'{split_name}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

gen_gap = float(results['ood']['errors'].mean() - results['id']['errors'].mean())
print(f"\n>>> Generalization Gap: {gen_gap:.6f}")
summary = {
    'generalization_gap': gen_gap,
    'id_mean':  float(results['id']['errors'].mean()),
    'ood_mean': float(results['ood']['errors'].mean()),
    'latent_dim': 10240,
    'n_id':  len(results['id']['metadata']),
    'n_ood': len(results['ood']['metadata']),
}
with open(OUT_DIR + 'extraction_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Done!", json.dumps(summary, indent=2))

#!/usr/bin/env python3
"""
Merge per-rank shard bundles from vjepa_extract_embeddings.py into a single bundle,
then hand off to vjepa_rq3_metrics.py.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--world-size', type=int, required=True)
args = parser.parse_args()

VJEPA2_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = VJEPA2_DIR / 'outputs'
WORLD_SIZE = args.world_size

shard_paths = [OUTPUT_DIR / f'intphys2_main1012_embeddings_bundle_shard{r}of{WORLD_SIZE}.pt'
               for r in range(WORLD_SIZE)]

for p in shard_paths:
    if not p.exists():
        sys.exit(f"ERROR: missing shard {p}")

print(f"Merging {WORLD_SIZE} shards...")
keys_list = ['embeddings', 'frame_embeddings']
meta_keys = ['video_ids', 'conditions', 'types', 'difficulties', 'cameras']

merged = {k: [] for k in keys_list + meta_keys}

for p in shard_paths:
    b = torch.load(str(p), map_location='cpu', weights_only=False)
    for k in keys_list:
        merged[k].append(b[k])
    for k in meta_keys:
        merged[k].extend(list(b[k]))
    print(f"  {p.name}: {b['embeddings'].shape[0]} videos")

bundle = {
    'embeddings':       torch.cat(merged['embeddings'],       dim=0),
    'frame_embeddings': torch.cat(merged['frame_embeddings'], dim=0),
    **{k: merged[k] for k in meta_keys},
}

out = OUTPUT_DIR / 'intphys2_main1012_embeddings_bundle.pt'
torch.save(bundle, str(out))
print(f"Merged bundle saved: {out}  shape={bundle['embeddings'].shape}")

# merge summary JSONs
errors_all = []
n_processed = 0
for r in range(WORLD_SIZE):
    sp = OUTPUT_DIR / f'extraction_summary_shard{r}of{WORLD_SIZE}.json'
    if sp.exists():
        with open(sp) as f:
            s = json.load(f)
        n_processed += s.get('n_processed', 0)
        errors_all.extend(s.get('errors', []))

with open(OUTPUT_DIR / 'extraction_summary.json', 'w') as f:
    json.dump({'n_processed': n_processed, 'n_errors': len(errors_all),
               'world_size': WORLD_SIZE, 'errors': errors_all}, f, indent=2)

print(f"Total processed: {n_processed} | errors: {len(errors_all)}")

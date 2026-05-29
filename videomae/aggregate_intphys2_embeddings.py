#!/usr/bin/env python3
"""Aggregate per-video embedding .pt files into a V-JEPA-compatible analysis bundle.

Joins conditions, types, difficulties, and cameras from the IntPhys2 Main metadata CSV
so the output bundle can be used directly by compute_intphys2_main1012_dci_cv.py.
"""

import argparse
import csv
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Aggregate IntPhys2 VideoMAE embeddings")
    parser.add_argument(
        "--summary_jsonl",
        type=str,
        required=True,
        help="Path to summary.jsonl produced by infer_intphys2_subset.py",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/IntPhys2/Main/metadata.csv",
        help="IntPhys2 Main metadata CSV (columns: name, condition, type, Difficulty, Camera).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for the aggregated bundle.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="intphys2_main1012",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    summary_path  = Path(args.summary_jsonl).resolve()
    metadata_path = Path(args.metadata_csv).resolve()
    output_dir    = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows found in {summary_path}")

    # build lookup: video_id (= name field in metadata) → metadata row
    meta_by_id = {r['name']: r for r in load_csv(metadata_path)}

    embeddings       = []
    video_ids        = []
    conditions       = []
    types            = []
    difficulties     = []
    cameras          = []
    unmatched        = []

    for row in rows:
        out_file = Path(row["output_file"])
        payload  = torch.load(out_file, map_location="cpu")
        emb = payload["embedding"]
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb[0]
        embeddings.append(emb.to(torch.float32))

        vid_id = payload["video_id"]
        video_ids.append(vid_id)

        meta = meta_by_id.get(vid_id)
        if meta is None:
            unmatched.append(vid_id)
            conditions.append('')
            types.append('')
            difficulties.append('')
            cameras.append('')
        else:
            conditions.append(meta.get('condition', ''))
            types.append(meta.get('type', ''))
            difficulties.append(meta.get('Difficulty', ''))
            cameras.append(meta.get('Camera', ''))

    if unmatched:
        print(f"WARNING: {len(unmatched)} video_ids not found in metadata CSV: {unmatched[:5]}{'...' if len(unmatched) > 5 else ''}")

    embeddings_tensor = torch.stack(embeddings, dim=0)   # [N, 768]

    bundle = {
        "embeddings":    embeddings_tensor,
        "video_ids":     video_ids,
        "conditions":    conditions,
        "types":         types,
        "difficulties":  difficulties,
        "cameras":       cameras,
        # VideoMAE outputs one embedding per clip — no per-frame temporal resolution
        "frame_embeddings": None,
    }

    bundle_path = output_dir / f"{args.output_prefix}_embeddings_bundle.pt"
    torch.save(bundle, bundle_path)
    print(f"Aggregated N={len(video_ids)} videos, D={embeddings_tensor.shape[1]}")
    print(f"Saved bundle: {bundle_path}")

    n_pos = sum(1 for t in types if 'Possible'   in t)
    n_imp = sum(1 for t in types if 'Impossible' in t)
    print(f"  possible={n_pos}  impossible={n_imp}  unmatched={len(unmatched)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate per-video embedding .pt files into a single analysis bundle."""

import argparse
import csv
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Aggregate IntPhys2 embedding outputs")
    parser.add_argument(
        "--summary_jsonl",
        type=str,
        required=True,
        help="Path to summary.jsonl produced by infer_intphys2_subset.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for aggregated outputs.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="intphys2_subset64",
        help="Prefix for output filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    rows = []
    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows found in summary file: {summary_path}")

    embeddings = []
    video_ids = []
    video_paths = []
    sampled_indices = []
    shape_rows = []

    for row in rows:
        out_file = Path(row["output_file"])
        payload = torch.load(out_file, map_location="cpu")
        emb = payload["embedding"]
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb[0]
        embeddings.append(emb.to(torch.float32))
        video_ids.append(payload["video_id"])
        video_paths.append(payload["video_path"])
        sampled_indices.append(payload["sampled_frame_indices"])
        shape_rows.append(payload.get("shapes", {}))

    embeddings_tensor = torch.stack(embeddings, dim=0)  # [N, D]

    bundle = {
        "embeddings": embeddings_tensor,
        "video_ids": video_ids,
        "video_paths": video_paths,
        "sampled_frame_indices": sampled_indices,
        "source_summary_jsonl": str(summary_path),
    }
    bundle_path = output_dir / f"{args.output_prefix}_embeddings_bundle.pt"
    torch.save(bundle, bundle_path)

    meta_jsonl_path = output_dir / f"{args.output_prefix}_metadata.jsonl"
    with meta_jsonl_path.open("w", encoding="utf-8") as f:
        for i in range(len(video_ids)):
            rec = {
                "video_id": video_ids[i],
                "video_path": video_paths[i],
                "sampled_frame_indices": sampled_indices[i],
                "embedding_dim": int(embeddings_tensor.shape[1]),
                "shapes": shape_rows[i],
            }
            f.write(json.dumps(rec) + "\n")

    meta_csv_path = output_dir / f"{args.output_prefix}_metadata.csv"
    with meta_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "video_path",
                "embedding_dim",
                "sampled_frame_indices",
                "raw_frames_thwc",
                "model_input_bcthw",
                "embedding_shape",
            ],
        )
        writer.writeheader()
        for i in range(len(video_ids)):
            shapes = shape_rows[i] if isinstance(shape_rows[i], dict) else {}
            writer.writerow(
                {
                    "video_id": video_ids[i],
                    "video_path": video_paths[i],
                    "embedding_dim": int(embeddings_tensor.shape[1]),
                    "sampled_frame_indices": json.dumps(sampled_indices[i]),
                    "raw_frames_thwc": json.dumps(shapes.get("raw_frames_thwc")),
                    "model_input_bcthw": json.dumps(shapes.get("model_input_bcthw")),
                    "embedding_shape": json.dumps(shapes.get("embedding")),
                }
            )

    print(f"Aggregated N={len(video_ids)} videos, D={embeddings_tensor.shape[1]}")
    print(f"Saved bundle:   {bundle_path}")
    print(f"Saved metadata: {meta_jsonl_path}")
    print(f"Saved metadata: {meta_csv_path}")


if __name__ == "__main__":
    main()

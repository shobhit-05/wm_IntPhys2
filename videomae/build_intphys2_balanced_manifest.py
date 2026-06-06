#!/usr/bin/env python3
"""Build a balanced IntPhys2 manifest (per condition, per ID/OOD class)."""

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build balanced IntPhys2 manifest")
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/IntPhys2/Main/metadata.csv",
        help="Path to IntPhys2 metadata.csv",
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/IntPhys2/Main/Videos",
        help="Directory containing video files named {name}.mp4",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        default="manifests/intphys2_balanced200.txt",
        help="Output manifest path (one absolute video path per line)",
    )
    parser.add_argument(
        "--n_per_type_per_condition",
        type=int,
        default=25,
        help="How many Possible and Impossible videos to sample per condition",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_possible_impossible(type_value: str) -> str:
    # Expected examples: 1_Possible, 2_Impossible
    suffix = type_value.split("_", 1)[-1].strip()
    if suffix not in {"Possible", "Impossible"}:
        raise ValueError(f"Unexpected type value: {type_value}")
    return suffix


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    metadata_csv = Path(args.metadata_csv).resolve()
    videos_dir = Path(args.videos_dir).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for required in ("name", "condition", "type"):
            if required not in fieldnames:
                raise ValueError(
                    f"Missing required column '{required}' in metadata. Found: {fieldnames}"
                )
        rows = list(reader)

    grouped = defaultdict(list)  # key: (condition, Possible|Impossible) -> [name, ...]
    for row in rows:
        cond = row["condition"].strip()
        cls = parse_possible_impossible(row["type"])
        grouped[(cond, cls)].append(row["name"].strip())

    conditions = sorted({k[0] for k in grouped.keys()})
    classes = ["Possible", "Impossible"]

    selected_names = []
    selected_counts = Counter()
    selected_cond_counts = Counter()

    for cond in conditions:
        for cls in classes:
            candidates = grouped[(cond, cls)]
            need = args.n_per_type_per_condition
            if len(candidates) < need:
                raise ValueError(
                    f"Not enough samples for (condition={cond}, class={cls}). "
                    f"Need {need}, found {len(candidates)}"
                )
            picks = rng.sample(candidates, need)
            selected_names.extend(picks)
            selected_counts[cls] += len(picks)
            selected_cond_counts[cond] += len(picks)

    # Keep manifest order deterministic but shuffled by fixed seed.
    rng.shuffle(selected_names)

    manifest_paths = []
    missing_files = []
    for name in selected_names:
        video_path = videos_dir / f"{name}.mp4"
        if not video_path.exists():
            missing_files.append(str(video_path))
            continue
        manifest_paths.append(str(video_path))

    if missing_files:
        raise FileNotFoundError(
            f"Missing {len(missing_files)} sampled video files. "
            f"First missing: {missing_files[0]}"
        )

    with output_manifest.open("w", encoding="utf-8") as f:
        for p in manifest_paths:
            f.write(p + "\n")

    print("Balanced manifest written:", output_manifest)
    print("Metadata file:", metadata_csv)
    print("Videos dir:", videos_dir)
    print("Seed:", args.seed)
    print("n_per_type_per_condition:", args.n_per_type_per_condition)
    print("Total rows:", len(manifest_paths))
    print("Counts by Possible/Impossible:", dict(selected_counts))
    print("Counts by condition:", dict(selected_cond_counts))


if __name__ == "__main__":
    main()

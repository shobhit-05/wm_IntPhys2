#!/usr/bin/env python3
"""Join IntPhys2 metadata labels onto subset embedding metadata."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Join IntPhys2 labels with subset metadata")
    parser.add_argument(
        "--intphys_metadata_csv",
        type=str,
        default="/mmfs1/gscratch/astro/klinjin/wm_IntPhys2/IntPhys2/Main/metadata.csv",
    )
    parser.add_argument(
        "--subset_metadata_csv",
        type=str,
        default="/gpfs/projects/infoseeking/sgupta01/VideoMAEv2/outputs/intphys2_subset64_pipeline/aggregated/intphys2_subset64_metadata.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/gpfs/projects/infoseeking/sgupta01/VideoMAEv2/outputs/intphys2_subset64_pipeline/aggregated",
    )
    parser.add_argument("--output_prefix", type=str, default="intphys2_subset64_joined")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_type_to_labels(type_value: str) -> tuple[str, str]:
    # Examples: 1_Possible, 2_Impossible
    possible_or_impossible = type_value.split("_", 1)[-1].strip()
    if possible_or_impossible == "Possible":
        id_ood = "ID"
    elif possible_or_impossible == "Impossible":
        id_ood = "OOD"
    else:
        id_ood = "UNKNOWN"
    return possible_or_impossible, id_ood


def main() -> None:
    args = parse_args()
    intphys_csv = Path(args.intphys_metadata_csv).resolve()
    subset_csv = Path(args.subset_metadata_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not intphys_csv.exists():
        raise FileNotFoundError(f"IntPhys metadata CSV not found: {intphys_csv}")
    if not subset_csv.exists():
        raise FileNotFoundError(f"Subset metadata CSV not found: {subset_csv}")

    intphys_rows = load_csv(intphys_csv)
    subset_rows = load_csv(subset_csv)

    by_name = {}
    for r in intphys_rows:
        by_name[r["name"]] = r

    joined_rows = []
    unmatched = []
    for r in subset_rows:
        video_id = r["video_id"]
        label = by_name.get(video_id)
        if label is None:
            unmatched.append(video_id)
            continue

        possible_or_impossible, id_ood = parse_type_to_labels(label["type"])
        joined = {
            "video_id": video_id,
            "source_path": r["video_path"],
            "possible_or_impossible": possible_or_impossible,
            "id_ood": id_ood,
            "condition": label["condition"],
            "scene_index": label["SceneIndex"],
            "type_raw": label["type"],
            "file_name": label["file_name"],
            "game_name": label["game_name"],
            "env": label["env"],
            "occluder": label["occluder"],
            "difficulty": label["Difficulty"],
            "camera": label["Camera"],
            "embedding_dim": r["embedding_dim"],
            "sampled_frame_indices": r["sampled_frame_indices"],
        }
        joined_rows.append(joined)

    id_counter = Counter(row["id_ood"] for row in joined_rows)
    condition_counter = Counter(row["condition"] for row in joined_rows)

    summary = {
        "intphys_rows_total": len(intphys_rows),
        "subset_rows_total": len(subset_rows),
        "matched_rows": len(joined_rows),
        "unmatched_rows": len(unmatched),
        "unmatched_video_ids": unmatched,
        "id_ood_counts": dict(id_counter),
        "condition_counts": dict(condition_counter),
        "join_key": "subset.video_id == intphys_metadata.name",
        "label_source_file": str(intphys_csv),
        "subset_source_file": str(subset_csv),
    }

    csv_out = output_dir / f"{args.output_prefix}.csv"
    jsonl_out = output_dir / f"{args.output_prefix}.jsonl"
    summary_out = output_dir / f"{args.output_prefix}_validation_summary.json"

    if joined_rows:
        fieldnames = list(joined_rows[0].keys())
        with csv_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(joined_rows)
    else:
        with csv_out.open("w", encoding="utf-8", newline="") as f:
            f.write("")

    with jsonl_out.open("w", encoding="utf-8") as f:
        for row in joined_rows:
            f.write(json.dumps(row) + "\n")

    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Matched {summary['matched_rows']} / {summary['subset_rows_total']}")
    print(f"ID/OOD counts: {summary['id_ood_counts']}")
    print(f"Condition counts: {summary['condition_counts']}")
    print(f"Wrote: {csv_out}")
    print(f"Wrote: {jsonl_out}")
    print(f"Wrote: {summary_out}")


if __name__ == "__main__":
    main()
